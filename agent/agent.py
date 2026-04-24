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
from pathlib import Path

from .config import Config
from .executor import execute, execute_click_pixels
from .history import History, render_command
from .parser import ClickCommand, parse_command
from .screen import (
    ScreenGeometry,
    annotate_candidates,
    capture_screenshot,
    crop_around,
    detect_geometry,
)
from .state import AgentState, load_state, save_state
from .vlm import GeminiClient, VerificationResult

log = logging.getLogger(__name__)


class AgentHalt(RuntimeError):
    """Raised when the verifier reports the screen state does not match the goal."""


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
) -> tuple[VerificationResult, str]:
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
        raw = vlm.plan_action(
            step,
            screenshot,
            history_summary=history_summary,
            previous_failure=previous_failure,
        )
        cmd = parse_command(raw)
        if cmd is None:
            last_parse_error = f"Could not parse VLM response: {raw!r}"
            log.warning(
                "%s — %s",
                last_parse_error,
                "retrying" if attempts <= max_parse_retries else "giving up",
            )
            continue

        if isinstance(cmd, ClickCommand) and enable_two_stage_click:
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
        verdict = vlm.verify(step, post_screenshot)
        log.info("Verify: %s", verdict.reason)
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

    for attempt_idx in range(1, total_attempts + 1):
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
        )
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
    vlm = GeminiClient(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        retry_max_attempts=config.gemini_retry_max_attempts,
        retry_base_delay_seconds=config.gemini_retry_base_delay_seconds,
        retry_max_delay_seconds=config.gemini_retry_max_delay_seconds,
    )
    history = History(window=config.history_window)

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

    log.info("All %d step(s) completed successfully.", len(steps))
    print(f"[ok] All {len(steps)} step(s) completed successfully.")
    return 0
