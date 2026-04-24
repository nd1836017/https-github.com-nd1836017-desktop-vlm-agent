"""Main agent loop: tasks.txt -> VLM -> parse -> execute -> verify.

Includes:
- Rolling short-term memory of recent (step, action, verdict) records,
  injected into each plan prompt.
- Replan-on-failure: when a verifier returns FAIL, the agent replans with
  the failure reason, up to `max_replans_per_step` times before halting.
- Checkpoint + resume: after each verified step the progress is saved to
  a JSON state file so a long run can be resumed after a crash/abort.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import ArtifactWriter
from .config import Config
from .cost import RpdGuard
from .executor import execute, execute_click_pixels
from .history import History, render_command
from .ocr import find_text_center
from .parser import ClickCommand, ClickTextCommand, PauseCommand, parse_command
from .screen import (
    ScreenGeometry,
    annotate_candidates,
    capture_screenshot,
    crop_around,
    detect_geometry,
)
from .state import AgentState, load_state, save_state
from .vlm import GeminiClient, VerificationResult


@dataclass
class ReplanCounter:
    """Tracks total replans across the whole run (not just per-step).

    When ``total_max > 0`` and ``total_used >= total_max`` the agent halts
    with a clear message. A counter value of 0 means "unlimited" — matching
    the PR #2 behaviour for backwards compat.
    """

    total_max: int = 0
    total_used: int = 0
    budget_exhausted: bool = field(default=False)

    def can_replan(self) -> bool:
        if self.total_max <= 0:
            return True
        return self.total_used < self.total_max

    def record_replan(self) -> None:
        self.total_used += 1
        if self.total_max > 0 and self.total_used >= self.total_max:
            self.budget_exhausted = True

log = logging.getLogger(__name__)


class AgentHalt(RuntimeError):
    """Raised when the verifier reports the screen state does not match the goal."""


@dataclass
class PauseRequested:
    """Sentinel returned by ``_attempt_step`` when the planner emitted PAUSE.

    The run loop unwraps this and prompts the human; it is not propagated
    further as a verdict.
    """

    reason: str
    raw: str


def _handle_pause(reason: str) -> bool:
    """Block on stdin until the user signals readiness to resume.

    Returns True if the user wants to continue, False if they want to abort.
    Visible to tests via the ``input`` builtin patch.
    """
    log.warning("Agent paused — waiting for human: %s", reason)
    print(
        "\n" + "=" * 60
        + f"\n[!] PAUSE: {reason}\n"
        + "    Resolve the prompt above (e.g. approve on your phone, solve\n"
        + "    the captcha, etc.) and then press Enter to resume.\n"
        + "    Type 'q' + Enter to abort the run instead.\n"
        + "=" * 60,
        file=sys.stderr,
    )
    try:
        answer = input(">>> Resume? [Enter / q]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer != "q"


def read_tasks(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
    steps: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        steps.append(line)
    return steps


def _execute_click_two_stage(
    step: str,
    cmd: ClickCommand,
    screenshot: object,  # PIL.Image.Image; typed broadly to avoid extra import
    vlm: GeminiClient,
    geometry: ScreenGeometry,
    animation_buffer: float,
    crop_size_px: int,
    max_candidates: int,
    click_min_delay_seconds: float = 0.0,
    click_max_delay_seconds: float = 0.0,
) -> tuple[bool, str]:
    """Execute a CLICK using the two-stage (crop -> refine -> disambiguate) path.

    Returns `(ok, action_text)`. On `ok=False` the caller treats the step as
    a verify-like FAIL (eligible for replan) with the reason embedded in
    `action_text` — so the agent does NOT blindly click the coarse point.
    """
    crop = crop_around(screenshot, geometry, cmd.x, cmd.y, crop_size_px)
    log.info(
        "Two-stage CLICK: coarse norm=(%d,%d); crop origin=%s size=%s",
        cmd.x,
        cmd.y,
        crop.origin_px,
        crop.size_px,
    )

    candidates = vlm.refine_click(step, crop.image, max_candidates=max_candidates)
    log.info("Refine found %d candidate(s): %s", len(candidates), candidates)

    if not candidates:
        return (
            False,
            f"<refine-found-none coarse=[{cmd.x},{cmd.y}]>",
        )

    if len(candidates) == 1:
        chosen = candidates[0]
        log.info("Single candidate; clicking it directly.")
    else:
        annotated = annotate_candidates(screenshot, crop, candidates)
        pick = vlm.disambiguate_candidates(step, annotated, len(candidates))
        log.info("Disambiguator picked index %d / %d", pick, len(candidates))
        if pick <= 0:
            return (
                False,
                f"<disambig-picked-none N={len(candidates)} coarse=[{cmd.x},{cmd.y}]>",
            )
        chosen = candidates[pick - 1]

    px, py = crop.crop_norm_to_full_pixel(*chosen)
    action_text = (
        f"CLICK_PX [{px},{py}] (2-stage from CLICK [{cmd.x},{cmd.y}], "
        f"{len(candidates)} cand)"
    )
    log.info("Action: %s", action_text)
    execute_click_pixels(
        px,
        py,
        animation_buffer_seconds=animation_buffer,
        click_min_delay_seconds=click_min_delay_seconds,
        click_max_delay_seconds=click_max_delay_seconds,
    )
    return True, action_text


def _execute_click_text(
    cmd: ClickTextCommand,
    screenshot,  # PIL.Image.Image
    geometry: ScreenGeometry,
    animation_buffer: float,
    click_min_delay_seconds: float = 0.0,
    click_max_delay_seconds: float = 0.0,
) -> tuple[bool, str]:
    """Locate ``cmd.label`` via OCR and click its center.

    Returns `(ok, action_text)`. On a miss we return `ok=False` with a
    reason embedded in `action_text` so the agent loop treats the step as
    a verify FAIL (replan path), rather than blindly clicking the wrong
    spot. Uses `agent.ocr.find_text_center` which is tesseract-backed.
    """
    match = find_text_center(screenshot, cmd.label)
    if match is None:
        reason = f"CLICK_TEXT [{cmd.label}] (no OCR match)"
        log.warning("%s", reason)
        return False, reason

    px, py = match.center()
    # Geometry's size may differ from the screenshot size (e.g. HiDPI). For
    # now we assume screenshot == screen, which is how capture_screenshot()
    # behaves on this codebase. Map safely if that invariant changes.
    w, h = screenshot.size
    if (w, h) != (geometry.width, geometry.height):
        # Rescale pixel coordinates to the physical screen.
        px = int(round(px * geometry.width / w))
        py = int(round(py * geometry.height / h))

    action_text = (
        f"CLICK_TEXT [{cmd.label}] -> pixels=({px},{py}) "
        f"(match={match.text!r}, score={match.score:.2f}, conf={match.confidence:.0f})"
    )
    log.info("Action: %s", action_text)
    execute_click_pixels(
        px,
        py,
        animation_buffer_seconds=animation_buffer,
        click_min_delay_seconds=click_min_delay_seconds,
        click_max_delay_seconds=click_max_delay_seconds,
    )
    return True, action_text


def _attempt_step(
    step: str,
    vlm: GeminiClient,
    geometry: ScreenGeometry,
    animation_buffer: float,
    max_parse_retries: int,
    history_summary: str,
    previous_failure: str,
    enable_two_stage_click: bool,
    two_stage_crop_size_px: int,
    max_click_candidates: int,
    click_min_delay_seconds: float = 0.0,
    click_max_delay_seconds: float = 0.0,
    type_min_interval_seconds: float = 0.02,
    type_max_interval_seconds: float = 0.02,
    log_redact_type: bool = True,
    artifact_writer: ArtifactWriter | None = None,
    step_idx: int = 0,
) -> tuple[VerificationResult | PauseRequested, str]:
    """Run one plan/execute/verify attempt and return (verdict, action_text).

    Handles parse-failure retry internally. `action_text` is the rendered
    command that was executed, or a synthetic marker if we never got a
    parseable command.
    """
    attempts = 0
    last_parse_error = ""
    while attempts <= max_parse_retries:
        attempts += 1
        screenshot = capture_screenshot()
        if artifact_writer is not None:
            artifact_writer.save_before(step_idx, screenshot)
        raw, cmd = vlm.plan_action(
            step,
            screenshot,
            history_summary=history_summary,
            previous_failure=previous_failure,
        )
        if cmd is None:
            # JSON-mode failed or disabled — fall back to regex parse.
            cmd = parse_command(raw)
        if cmd is None:
            last_parse_error = f"Could not parse VLM response: {raw!r}"
            log.warning(
                "%s — %s",
                last_parse_error,
                "retrying" if attempts <= max_parse_retries else "giving up",
            )
            continue

        if isinstance(cmd, PauseCommand):
            log.info("Planner emitted PAUSE: %s", cmd.reason)
            if artifact_writer is not None:
                artifact_writer.save_plan(step_idx, raw, f"PAUSE [{cmd.reason}]")
            return PauseRequested(reason=cmd.reason, raw=raw), f"PAUSE [{cmd.reason}]"

        if isinstance(cmd, ClickTextCommand):
            ok, action_text = _execute_click_text(
                cmd=cmd,
                screenshot=screenshot,
                geometry=geometry,
                animation_buffer=animation_buffer,
                click_min_delay_seconds=click_min_delay_seconds,
                click_max_delay_seconds=click_max_delay_seconds,
            )
            if not ok:
                return (
                    VerificationResult(
                        passed=False,
                        reason=f"CLICK_TEXT failed: {action_text}",
                    ),
                    action_text,
                )
        elif isinstance(cmd, ClickCommand) and enable_two_stage_click:
            ok, action_text = _execute_click_two_stage(
                step=step,
                cmd=cmd,
                screenshot=screenshot,
                vlm=vlm,
                geometry=geometry,
                animation_buffer=animation_buffer,
                crop_size_px=two_stage_crop_size_px,
                max_candidates=max_click_candidates,
                click_min_delay_seconds=click_min_delay_seconds,
                click_max_delay_seconds=click_max_delay_seconds,
            )
            if not ok:
                # Refinement couldn't find a target — synthesize a FAIL verdict
                # so the replan loop can try a different action.
                return (
                    VerificationResult(
                        passed=False,
                        reason=f"Two-stage CLICK refinement failed: {action_text}",
                    ),
                    action_text,
                )
        else:
            action_text = render_command(cmd, redact_type=log_redact_type)
            log.info("Action: %s", action_text)
            execute(
                cmd,
                geometry,
                animation_buffer_seconds=animation_buffer,
                click_min_delay_seconds=click_min_delay_seconds,
                click_max_delay_seconds=click_max_delay_seconds,
                type_min_interval_seconds=type_min_interval_seconds,
                type_max_interval_seconds=type_max_interval_seconds,
                log_redact_type=log_redact_type,
            )

        post_screenshot = capture_screenshot()
        if artifact_writer is not None:
            artifact_writer.save_after(step_idx, post_screenshot)
            artifact_writer.save_plan(step_idx, raw, action_text)
        verdict = vlm.verify(step, post_screenshot)
        log.info("Verify: %s", verdict.reason)
        if artifact_writer is not None:
            artifact_writer.save_verdict(step_idx, verdict.passed, verdict.reason)
        return verdict, action_text

    return (
        VerificationResult(
            passed=False,
            reason=f"Parse failure after {max_parse_retries + 1} attempts: {last_parse_error}",
        ),
        "<parse-failed>",
    )


def run_step(
    step: str,
    vlm: GeminiClient,
    geometry: ScreenGeometry,
    animation_buffer: float,
    max_parse_retries: int,
    max_replans: int,
    history: History,
    enable_two_stage_click: bool = True,
    two_stage_crop_size_px: int = 300,
    max_click_candidates: int = 5,
    click_min_delay_seconds: float = 0.0,
    click_max_delay_seconds: float = 0.0,
    type_min_interval_seconds: float = 0.02,
    type_max_interval_seconds: float = 0.02,
    log_redact_type: bool = True,
    replan_counter: ReplanCounter | None = None,
    artifact_writer: ArtifactWriter | None = None,
    step_idx: int = 0,
    pause_handler: Callable[[str], bool] | None = None,
) -> VerificationResult:
    """Run one step with up to `max_replans` replans on verify FAIL.

    The plan/verify loop is:
        attempt 1              -> if PASS, return
                               -> if FAIL, replan #1
        attempt 2 (replan #1)  -> if PASS, return
                               -> if FAIL, replan #2
        ...
        after max_replans FAILs, return the final FAIL verdict.
    """
    total_attempts = max_replans + 1
    previous_failure = ""
    last_verdict: VerificationResult | None = None
    last_action_text = "<no-action>"
    attempt_idx = 0
    pauses_so_far = 0
    # Hard safety: a single step shouldn't require more than this many human
    # interventions; otherwise we're probably stuck in a PAUSE loop.
    max_pauses_per_step = 10

    while attempt_idx < total_attempts:
        attempt_idx += 1
        # Global replan guard. The first attempt is NOT a replan, so only
        # consult the budget on attempts 2+. The budget is *consumed* below
        # AFTER we know the attempt wasn't a PAUSE — PAUSE iterations roll
        # `attempt_idx` back and must not permanently drain the global
        # counter (see PR #7 review).
        is_replan_attempt = attempt_idx > 1
        if (
            is_replan_attempt
            and replan_counter is not None
            and not replan_counter.can_replan()
        ):
            log.error(
                "Global replan budget exhausted (%d/%d used); halting.",
                replan_counter.total_used,
                replan_counter.total_max,
            )
            halted = VerificationResult(
                passed=False,
                reason=(
                    f"Global replan budget exhausted "
                    f"({replan_counter.total_used}/{replan_counter.total_max}). "
                    f"Last failure: {last_verdict.reason if last_verdict else 'n/a'}"
                ),
            )
            history.record(step, last_action_text, passed=False, reason=halted.reason)
            return halted

        log.info(
            "Step attempt %d/%d (replan budget: %d remaining)",
            attempt_idx,
            total_attempts,
            total_attempts - attempt_idx,
        )

        verdict, action_text = _attempt_step(
            step=step,
            vlm=vlm,
            geometry=geometry,
            animation_buffer=animation_buffer,
            max_parse_retries=max_parse_retries,
            history_summary=history.summary(),
            previous_failure=previous_failure,
            enable_two_stage_click=enable_two_stage_click,
            two_stage_crop_size_px=two_stage_crop_size_px,
            max_click_candidates=max_click_candidates,
            click_min_delay_seconds=click_min_delay_seconds,
            click_max_delay_seconds=click_max_delay_seconds,
            type_min_interval_seconds=type_min_interval_seconds,
            type_max_interval_seconds=type_max_interval_seconds,
            log_redact_type=log_redact_type,
            artifact_writer=artifact_writer,
            step_idx=step_idx,
        )

        # PAUSE: ask the human, then loop back without consuming the
        # replan budget — the screen state has changed (because the human
        # acted), so a fresh plan attempt is the right thing.
        if isinstance(verdict, PauseRequested):
            pauses_so_far += 1
            if pauses_so_far > max_pauses_per_step:
                return VerificationResult(
                    passed=False,
                    reason=(
                        f"Exceeded max PAUSE rounds ({max_pauses_per_step}) "
                        f"on this step; last reason: {verdict.reason}"
                    ),
                )
            handler = pause_handler or _handle_pause
            should_continue = handler(verdict.reason)
            history.record(
                step,
                action_text,
                passed=False,
                reason=f"PAUSE handled: {verdict.reason}",
            )
            if not should_continue:
                return VerificationResult(
                    passed=False,
                    reason=f"User aborted at PAUSE: {verdict.reason}",
                )
            # Don't count this attempt against the replan budget — replan
            # only fires on verifier FAILs, and PAUSE is neither. The
            # global replan counter has NOT yet been incremented for this
            # iteration (it's consumed below only on non-PAUSE attempts),
            # so no rollback is needed.
            attempt_idx -= 1
            continue

        # Non-PAUSE outcome: if this was a replan attempt (2+), consume
        # one slot of the global replan budget now — exactly once per
        # real replan iteration, regardless of PAUSE storms above.
        if is_replan_attempt and replan_counter is not None:
            replan_counter.record_replan()

        last_verdict = verdict
        last_action_text = action_text

        if verdict.passed:
            history.record(step, action_text, passed=True, reason=verdict.reason)
            return verdict

        # FAIL — prepare for replan.
        previous_failure = f"action `{action_text}` -> {verdict.reason}"
        log.warning(
            "Step verification failed on attempt %d/%d: %s",
            attempt_idx,
            total_attempts,
            verdict.reason,
        )

    # Budget exhausted.
    final = last_verdict or VerificationResult(
        passed=False, reason="No attempts were made (internal error)."
    )
    history.record(step, last_action_text, passed=False, reason=final.reason)
    return final


def run(config: Config, resume: bool = False) -> int:
    steps = read_tasks(config.tasks_file)
    if not steps:
        log.error("No steps found in %s", config.tasks_file)
        return 2

    # Checkpoint handling.
    existing_state = load_state(config.state_file) if resume else None
    start_idx = 1
    if existing_state is not None:
        if (
            existing_state.tasks_file == str(config.tasks_file)
            and existing_state.total_steps == len(steps)
            and 0 <= existing_state.last_completed_step < len(steps)
        ):
            start_idx = existing_state.last_completed_step + 1
            log.info(
                "Resuming from checkpoint: step %d/%d (file=%s)",
                start_idx,
                len(steps),
                config.state_file,
            )
            state = existing_state
        else:
            log.warning(
                "Ignoring stale checkpoint at %s (tasks file or step count changed).",
                config.state_file,
            )
            state = AgentState.initial(config.tasks_file, len(steps))
    else:
        state = AgentState.initial(config.tasks_file, len(steps))

    geometry = detect_geometry()
    rpd_guard = RpdGuard(
        rpd_limit=config.rpd_limit,
        warn_threshold=config.rpd_warn_threshold,
        halt_threshold=config.rpd_halt_threshold,
    )
    vlm = GeminiClient(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        retry_max_attempts=config.gemini_retry_max_attempts,
        retry_base_delay_seconds=config.gemini_retry_base_delay_seconds,
        retry_max_delay_seconds=config.gemini_retry_max_delay_seconds,
        enable_json_output=config.enable_json_output,
        rpd_guard=rpd_guard,
    )
    history = History(window=config.history_window)
    replan_counter = ReplanCounter(total_max=config.max_total_replans)
    artifact_writer = ArtifactWriter.create(
        enabled=config.save_run_artifacts,
        base_dir=config.run_artifacts_dir,
    )

    log.info(
        "Loaded %d step(s) from %s (starting at step %d)",
        len(steps),
        config.tasks_file,
        start_idx,
    )

    for idx in range(start_idx, len(steps) + 1):
        step = steps[idx - 1]
        log.info("=" * 60)
        log.info("Step %d/%d: %s", idx, len(steps), step)
        log.info("=" * 60)

        result = run_step(
            step=step,
            vlm=vlm,
            geometry=geometry,
            animation_buffer=config.animation_buffer_seconds,
            max_parse_retries=config.max_step_retries,
            max_replans=config.max_replans_per_step,
            history=history,
            enable_two_stage_click=config.enable_two_stage_click,
            two_stage_crop_size_px=config.two_stage_crop_size_px,
            max_click_candidates=config.max_click_candidates,
            click_min_delay_seconds=config.click_min_delay_seconds,
            click_max_delay_seconds=config.click_max_delay_seconds,
            type_min_interval_seconds=config.type_min_interval_seconds,
            type_max_interval_seconds=config.type_max_interval_seconds,
            log_redact_type=config.log_redact_type,
            replan_counter=replan_counter,
            artifact_writer=artifact_writer,
            step_idx=idx,
        )

        # action_text is resolved inside the writer from the most recent
        # save_plan() call for this step — no placeholder needed.
        artifact_writer.append_summary(
            step_idx=idx,
            step_text=step,
            passed=result.passed,
            reason=result.reason,
        )

        if not result.passed:
            msg = (
                f"\n[!] HALT at step {idx}/{len(steps)}: {step}\n"
                f"    Reason: {result.reason}\n"
                f"    (Exhausted replan budget of {config.max_replans_per_step}.)\n"
                f"    Run `python -m agent --resume` after fixing the blocker to continue.\n"
                "    The agent has stopped to prevent runaway actions."
            )
            print(msg, file=sys.stderr)
            log.error("Halting execution: %s", result.reason)
            return 1

        # Commit progress to disk before moving on.
        state = state.advance()
        try:
            save_state(config.state_file, state)
        except OSError as exc:
            log.warning("Failed to save checkpoint to %s: %s", config.state_file, exc)

        # RPD guard: halt cleanly between steps so the checkpoint is consistent.
        if rpd_guard.should_halt():
            print("\n[!] " + rpd_guard.halt_message(), file=sys.stderr)
            log.error("Halting execution: %s", rpd_guard.halt_message())
            return 1

    log.info("All %d step(s) completed successfully.", len(steps))
    print(f"[ok] All {len(steps)} step(s) completed successfully.")
    return 0
