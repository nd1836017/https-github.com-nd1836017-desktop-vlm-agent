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
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import ArtifactWriter
from .browser_bridge import BrowserBridge
from .config import Config
from .cost import RpdGuard
from .executor import execute, execute_click_pixels
from .files import (
    FileMode,
    FileWorkspace,
    execute_attach_file,
    execute_capture_for_ai,
    execute_download,
    format_features_summary,
    inspect_features,
    resolve_mode,
)
from .history import History, render_command
from .ocr import find_text_center
from .parser import (
    AttachFileCommand,
    BrowserClickCommand,
    BrowserFillCommand,
    BrowserGoCommand,
    CaptureForAiCommand,
    ClickCommand,
    ClickTextCommand,
    DownloadCommand,
    PauseCommand,
    RecallCommand,
    RememberCommand,
    TypeCommand,
    parse_command,
)
from .screen import (
    ScreenGeometry,
    annotate_candidates,
    capture_screenshot,
    crop_around,
    detect_geometry,
    image_signature,
)
from .state import AgentState, load_state, save_state
from .task_decomposer import (
    apply_decomposer,
)
from .task_decomposer import (
    parse_mode as parse_decomposition_mode,
)
from .task_router import (
    RoutingMode,
    apply_router,
)
from .task_router import (
    parse_mode as parse_routing_mode,
)
from .tasks_loader import (
    TasksLoadError,
    TaskStep,
    attach_routing_hints,
    load_steps,
    load_tasks,
)
from .variables import (
    UnknownVariableError,
    VariableStore,
    substitute_variables,
    text_uses_variables,
)
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


def _find_if_begin_index(steps: Sequence[TaskStep], block_id: int) -> int | None:
    """Return the 1-based index of the ``if_begin`` step for ``block_id``.

    Used by the resume path to rewind into an IF block so the condition
    is re-evaluated. Returns ``None`` when no matching if_begin exists,
    which would only happen on a malformed step list (programmer error
    rather than a parse error — the loader rejects unmatched IF / END_IF
    at parse time).
    """
    for idx, step in enumerate(steps, start=1):
        if step.control_kind == "if_begin" and step.block_id == block_id:
            return idx
    return None


def _maybe_handle_control(
    *,
    task_step: TaskStep,
    idx: int,
    total_steps: int,
    vlm: GeminiClient,
    geometry: ScreenGeometry,
    branch_decisions: dict[int, bool],
    wait_until_timeout_seconds: float,
    wait_until_poll_seconds: float,
) -> VerificationResult | None:
    """Handle control-flow directives and skipped-branch steps.

    Returns:
        ``None`` if this is a normal action step that should continue
        through the planner/executor path.
        ``VerificationResult`` (passed=True) for control directives or
        steps inside a non-taken branch — the run loop should treat the
        step as completed without calling the planner.
        ``VerificationResult`` (passed=False) only for WAIT_UNTIL on
        timeout, which becomes a HALT.
    """
    kind = task_step.control_kind

    # IF [text] THEN — evaluate the condition and store the decision.
    if kind == "if_begin":
        assert task_step.block_id is not None
        assert task_step.condition_text is not None
        log.info(
            "IF [%s]: evaluating against current screen…",
            task_step.condition_text,
        )
        decision = _evaluate_condition_via_screenshot(
            text=task_step.condition_text,
            vlm=vlm,
            geometry=geometry,
            label=f"IF (step {idx}/{total_steps})",
        )
        branch_decisions[task_step.block_id] = decision
        log.info(
            "IF [%s] → %s branch.",
            task_step.condition_text,
            "THEN" if decision else "ELSE",
        )
        return VerificationResult(
            passed=True,
            reason=(
                f"IF [{task_step.condition_text}] -> "
                f"{'THEN' if decision else 'ELSE'}"
            ),
        )

    # ELSE / END_IF — pure markers, no work. They MUST always process
    # (regardless of branch) because they're the structural skeleton of
    # the block; skipping them would leave branch_decisions stale on
    # next iteration.
    if kind == "if_else":
        return VerificationResult(passed=True, reason="ELSE marker")
    if kind == "if_end":
        return VerificationResult(passed=True, reason="END_IF marker")

    # Branch-skip check applies to BOTH plain action steps and
    # WAIT_UNTIL. A WAIT_UNTIL inside the non-taken branch must NOT
    # poll the screen — that would burn API quota and could halt the
    # run on timeout for a directive the user never intended to reach.
    block_id = task_step.active_block_id
    if block_id is not None and task_step.branch is not None:
        decision = branch_decisions.get(block_id)
        if decision is None:
            # No decision recorded — should never happen since IF runs
            # before any of its inner steps. Defensive: log + execute.
            log.warning(
                "step %d in block %d has no recorded branch decision; "
                "executing anyway.",
                idx,
                block_id,
            )
        else:
            taken_branch = "then" if decision else "else"
            if task_step.branch != taken_branch:
                log.info(
                    "Skipping step %d: branch %s not taken (IF block %d).",
                    idx,
                    task_step.branch,
                    block_id,
                )
                return VerificationResult(
                    passed=True,
                    reason=(
                        f"skipped: branch '{task_step.branch}' not taken "
                        f"(IF block {block_id})"
                    ),
                )

    # WAIT_UNTIL [text] — poll the screen until the condition is true
    # or the timeout expires.
    if kind == "wait_until":
        assert task_step.condition_text is not None
        timeout = (
            task_step.wait_timeout_seconds
            if task_step.wait_timeout_seconds is not None
            else wait_until_timeout_seconds
        )
        poll = (
            task_step.wait_poll_seconds
            if task_step.wait_poll_seconds is not None
            else wait_until_poll_seconds
        )
        deadline = time.monotonic() + timeout
        attempt = 0
        log.info(
            "WAIT_UNTIL [%s]: polling every %.1fs (timeout %.1fs).",
            task_step.condition_text,
            poll,
            timeout,
        )
        while True:
            attempt += 1
            label = (
                f"WAIT_UNTIL (step {idx}/{total_steps}, attempt {attempt})"
            )
            if _evaluate_condition_via_screenshot(
                text=task_step.condition_text,
                vlm=vlm,
                geometry=geometry,
                label=label,
            ):
                log.info(
                    "WAIT_UNTIL [%s] satisfied after %d attempt(s).",
                    task_step.condition_text,
                    attempt,
                )
                return VerificationResult(
                    passed=True,
                    reason=(
                        f"WAIT_UNTIL [{task_step.condition_text}] "
                        f"satisfied after {attempt} attempt(s)"
                    ),
                )
            if time.monotonic() >= deadline:
                return VerificationResult(
                    passed=False,
                    reason=(
                        f"WAIT_UNTIL [{task_step.condition_text}] timed out "
                        f"after {timeout:.1f}s ({attempt} attempt(s))"
                    ),
                )
            time.sleep(poll)

    # Plain action step that survived the branch-skip check above —
    # let the run loop hand it to the planner.
    return None


def _evaluate_condition_via_screenshot(
    *,
    text: str,
    vlm: GeminiClient,
    geometry: ScreenGeometry,
    label: str,
) -> bool:
    """Take one screenshot and ask the VLM if ``text`` is on screen.

    Pure helper so IF and WAIT_UNTIL can share the same code path.
    Returns False on any VLM error and logs a warning — better to take
    the False branch than to crash the run.
    """
    try:
        image = capture_screenshot()
        return vlm.check_condition(text, image)
    except Exception as exc:  # pragma: no cover — defensive guard
        log.warning(
            "%s: condition check failed (%s); treating as False.",
            label,
            exc,
        )
        return False


def _log_routing_summary(
    steps: list[TaskStep],
    *,
    mode: RoutingMode,
) -> None:
    """Print a one-shot breakdown of router decisions to the run log.

    Quiet by default — emits a single INFO line with counts plus a
    DEBUG line per annotated step. Goal: make it obvious in the log
    whether the router did what you expected, without spamming.
    """
    counts: dict[str, int] = {
        "browser-fast": 0,
        "browser-vlm": 0,
        "desktop-vlm": 0,
        "unrouted": 0,
    }
    manual_count = 0
    for step in steps:
        hint = step.routing_hint
        if hint is None:
            counts["unrouted"] += 1
            continue
        counts[hint.complexity.value] = counts.get(hint.complexity.value, 0) + 1
        if hint.source == "manual":
            manual_count += 1

    log.info(
        "Task router (%s): %d browser-fast, %d browser-vlm, %d desktop-vlm, "
        "%d unrouted (%d from manual annotations)",
        mode.value,
        counts["browser-fast"],
        counts["browser-vlm"],
        counts["desktop-vlm"],
        counts["unrouted"],
        manual_count,
    )
    for idx, step in enumerate(steps, start=1):
        if step.routing_hint is None:
            continue
        log.debug(
            "  step %d [%s/%s]%s: %s",
            idx,
            step.routing_hint.complexity.value,
            step.routing_hint.source,
            (
                f" -> {step.routing_hint.suggested_command}"
                if step.routing_hint.suggested_command
                else ""
            ),
            step.text[:80],
        )


def read_tasks(path: Path, csv_override: Path | None = None) -> list[str]:
    """Load and expand a tasks file (delegates to ``tasks_loader.load_tasks``).

    Kept as a thin wrapper for backwards compatibility — earlier code/tests
    imported ``read_tasks`` directly from ``agent.agent``.
    """
    return load_tasks(path, csv_override=csv_override)


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


def _execute_remember(
    *,
    cmd: RememberCommand,
    screenshot,
    vlm: GeminiClient,
    variables: VariableStore,
    step_text: str,
) -> tuple[bool, str]:
    """Run a REMEMBER command. Returns ``(ok, action_text)``.

    Two forms:
      - ``REMEMBER [name = literal]`` — store ``literal`` directly.
      - ``REMEMBER [name]`` — ask the VLM to extract the value of ``name``
        from the current screen via ``GeminiClient.extract_value``.
    """
    if not cmd.name:
        return False, "REMEMBER missing variable name"
    if not cmd.from_screen:
        # Literal form: deterministic, no VLM call.
        try:
            variables.set(cmd.name, cmd.literal_value)
        except (ValueError, TypeError) as exc:
            return False, f"REMEMBER [{cmd.name}] failed: {exc}"
        return (
            True,
            f"REMEMBER [{cmd.name} = literal({len(cmd.literal_value)} chars)]",
        )

    # Screen-extract form: hand the planner's full step as a hint so the
    # extractor picks up surrounding context (e.g. "the order ID at the
    # top of the confirmation page"), but pass the bare variable name as
    # the primary identifier.
    hint = step_text.strip() if step_text else ""
    try:
        result = vlm.extract_value(cmd.name, screenshot, hint=hint)
    except Exception as exc:  # pragma: no cover - defensive
        # extract_value already retries on 5xx via _call_with_retry; if
        # we get here it's a genuine error worth surfacing as a FAIL.
        return False, f"REMEMBER [{cmd.name}] extract_value error: {exc}"
    if not result.found or not result.value:
        return (
            False,
            f"REMEMBER [{cmd.name}] — value not found on screen",
        )
    try:
        variables.set(cmd.name, result.value)
    except (ValueError, TypeError) as exc:
        return False, f"REMEMBER [{cmd.name}] store failed: {exc}"
    return (
        True,
        f"REMEMBER [{cmd.name}] = <extracted {len(result.value)} chars>",
    )


def _execute_recall(
    *,
    cmd: RecallCommand,
    geometry: ScreenGeometry,
    variables: VariableStore,
    animation_buffer: float,
    type_min_interval: float,
    type_max_interval: float,
    log_redact_type: bool,
) -> tuple[bool, str]:
    """Run a RECALL command by typing the stored value into the focused field.

    Equivalent to ``TYPE [<value>]`` but more explicit. If the variable
    is unset we fail loudly so the replan loop can recover.
    """
    if not cmd.name:
        return False, "RECALL missing variable name"
    if cmd.name not in variables:
        return (
            False,
            f"RECALL [{cmd.name}] — variable is unset (available: "
            f"{', '.join(sorted(variables.names())) or '(none)'})",
        )
    value = variables.get(cmd.name)
    synthetic = TypeCommand(text=value)
    execute(
        synthetic,
        geometry,
        animation_buffer_seconds=animation_buffer,
        type_min_interval_seconds=type_min_interval,
        type_max_interval_seconds=type_max_interval,
        log_redact_type=log_redact_type,
    )
    return (
        True,
        f"RECALL [{cmd.name}] -> typed {len(value)} chars",
    )


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
    workspace: FileWorkspace | None = None,
    step_idx: int = 0,
    extra_images: list[bytes] | None = None,
    variables: VariableStore | None = None,
    browser_bridge: BrowserBridge | None = None,
    routing_hint: str = "",
) -> tuple[VerificationResult | PauseRequested, str]:
    """Run one plan/execute/verify attempt and return (verdict, action_text).

    Handles parse-failure retry internally. `action_text` is the rendered
    command that was executed, or a synthetic marker if we never got a
    parseable command.

    `extra_images` is owned by ``run_step`` (which drains the workspace's
    feed buffer ONCE per step) and passed through to every call here so
    parse retries AND replan attempts on the same step see the same
    images. Passing ``None`` is equivalent to ``[]``.
    """
    if extra_images is None:
        extra_images = []
    # Capture the baseline screenshot ONCE for this attempt — every
    # parse retry must see the same visual state the original plan was
    # made for. Re-capturing on each retry meant a tooltip / animation
    # / cursor blink between attempts would silently change the screen
    # the next plan was generated from, so the verifier was sometimes
    # checking a different scene than the planner saw.
    screenshot = capture_screenshot()
    if artifact_writer is not None:
        artifact_writer.save_before(step_idx, screenshot)
    attempts = 0
    last_parse_error = ""
    while attempts <= max_parse_retries:
        attempts += 1
        raw, cmd = vlm.plan_action(
            step,
            screenshot,
            history_summary=history_summary,
            previous_failure=previous_failure,
            extra_images=extra_images,
            routing_hint=routing_hint,
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

        # File primitives: synthesize PASS/FAIL based on the command's own
        # success — skip the visual verifier because DOWNLOAD/CAPTURE_FOR_AI
        # don't change the screen and ATTACH_FILE's effect (the file path
        # text in a dialog) the verifier already covers via the next step.
        if isinstance(cmd, (DownloadCommand, AttachFileCommand, CaptureForAiCommand)):
            if workspace is None:
                return (
                    VerificationResult(
                        passed=False,
                        reason=(
                            "Internal error: file command emitted but no "
                            "FileWorkspace was set up for this run."
                        ),
                    ),
                    f"<{cmd.kind} blocked — no workspace>",
                )
            if isinstance(cmd, DownloadCommand):
                ok, action_text = execute_download(cmd, workspace)
            elif isinstance(cmd, AttachFileCommand):
                ok, action_text = execute_attach_file(cmd, workspace)
            else:
                ok, action_text = execute_capture_for_ai(
                    cmd, workspace, screenshot=screenshot
                )
            log.info("Action: %s", action_text)
            if artifact_writer is not None:
                # Take a post-action screenshot for the artifact bundle so
                # the file-command path matches the regular action path
                # (which captures both before/after). DOWNLOAD and
                # CAPTURE_FOR_AI rarely change the screen, but ATTACH_FILE
                # does (the path appears in the file-picker), and even
                # for the no-change cases an `after` snapshot lets a
                # postmortem reader confirm the screen didn't change
                # unexpectedly during the network/IO operation.
                try:
                    post_screenshot = capture_screenshot()
                    artifact_writer.save_after(step_idx, post_screenshot)
                except Exception as exc:  # pragma: no cover - defensive
                    # Artifact saving must never crash the agent. The
                    # writer's own `_write_image` already swallows OSError
                    # but a screen-capture failure would bubble up here.
                    log.warning(
                        "Failed to capture post-action screenshot for "
                        "file command on step %d: %s",
                        step_idx,
                        exc,
                    )
                artifact_writer.save_plan(step_idx, raw, action_text)
                artifact_writer.save_verdict(step_idx, ok, action_text)
            return (
                VerificationResult(passed=ok, reason=action_text),
                action_text,
            )

        # Memory primitives: REMEMBER stores a value into the run's
        # VariableStore; RECALL types a stored value into the focused field
        # (executed as a synthetic TYPE so it benefits from the standard
        # animation buffer + redaction). Like the file-command path we
        # synthesize PASS/FAIL based on the operation succeeding rather
        # than calling the visual verifier — REMEMBER doesn't change the
        # screen and RECALL's effect is just text appearing in a textbox,
        # which is verified by the next step.
        if isinstance(cmd, (RememberCommand, RecallCommand)):
            if variables is None:
                return (
                    VerificationResult(
                        passed=False,
                        reason=(
                            "Internal error: memory command emitted but no "
                            "VariableStore was set up for this run."
                        ),
                    ),
                    f"<{cmd.kind} blocked — no variable store>",
                )
            if isinstance(cmd, RememberCommand):
                ok, action_text = _execute_remember(
                    cmd=cmd,
                    screenshot=screenshot,
                    vlm=vlm,
                    variables=variables,
                    step_text=step,
                )
            else:
                ok, action_text = _execute_recall(
                    cmd=cmd,
                    geometry=geometry,
                    variables=variables,
                    animation_buffer=animation_buffer,
                    type_min_interval=type_min_interval_seconds,
                    type_max_interval=type_max_interval_seconds,
                    log_redact_type=log_redact_type,
                )
            log.info("Action: %s", action_text)
            if artifact_writer is not None:
                try:
                    post_screenshot = capture_screenshot()
                    artifact_writer.save_after(step_idx, post_screenshot)
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "Failed to capture post-action screenshot for "
                        "memory command on step %d: %s",
                        step_idx,
                        exc,
                    )
                artifact_writer.save_plan(step_idx, raw, action_text)
                artifact_writer.save_verdict(step_idx, ok, action_text)
            return (
                VerificationResult(passed=ok, reason=action_text),
                action_text,
            )

        # Browser fast-path primitives: drive Chrome via CDP. Like file /
        # memory commands these synthesize PASS/FAIL based on the bridge's
        # success rather than calling the visual verifier — BROWSER_GO and
        # BROWSER_FILL definitively change the page when they succeed
        # (Page.navigate fires; setting an input's value + dispatching
        # input/change events is observable to React/Vue), and the next
        # natural-language step's verifier pass will catch any mismatch
        # against the user's intent.
        if isinstance(
            cmd, (BrowserGoCommand, BrowserClickCommand, BrowserFillCommand)
        ):
            if browser_bridge is None or not browser_bridge.is_connected():
                # Two failure modes are both real:
                #   - BROWSER_FAST_PATH=false (bridge wasn't created)
                #   - bridge created but couldn't connect to CDP at run start
                # In both cases we fail this attempt with a clear reason; the
                # planner sees previous_failure and will replan to the visual
                # equivalent (CLICK / TYPE / etc.) on the next attempt.
                return (
                    VerificationResult(
                        passed=False,
                        reason=(
                            "Browser fast-path command emitted but Chrome "
                            "CDP bridge is not available. Use the visual "
                            "primitives (CLICK / TYPE / PRESS) instead, or "
                            "launch Chrome with --remote-debugging-port=29229 "
                            "and set BROWSER_FAST_PATH=true."
                        ),
                    ),
                    f"<{cmd.kind} blocked — bridge unavailable>",
                )
            if isinstance(cmd, BrowserGoCommand):
                ok, action_text = browser_bridge.navigate(cmd.url)
            elif isinstance(cmd, BrowserClickCommand):
                ok, action_text = browser_bridge.click(cmd.selector)
            else:
                ok, action_text = browser_bridge.fill(cmd.selector, cmd.value)
            log.info("Action: %s", action_text)
            if artifact_writer is not None:
                try:
                    post_screenshot = capture_screenshot()
                    artifact_writer.save_after(step_idx, post_screenshot)
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "Failed to capture post-action screenshot for "
                        "browser command on step %d: %s",
                        step_idx,
                        exc,
                    )
                artifact_writer.save_plan(step_idx, raw, action_text)
                artifact_writer.save_verdict(step_idx, ok, action_text)
            return (
                VerificationResult(passed=ok, reason=action_text),
                action_text,
            )

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

    parse_fail_reason = (
        f"Parse failure after {max_parse_retries + 1} attempts: {last_parse_error}"
    )
    if artifact_writer is not None:
        # Every other return path from _attempt_step writes both
        # save_plan() and save_verdict(), so a postmortem reader can
        # rely on the artifact bundle being symmetric. The parse-fail
        # path used to skip both, leaving a `before` screenshot with
        # no plan/verdict — confusing when triaging a stuck run.
        # Synthesize placeholders that match the verdict we return.
        artifact_writer.save_plan(
            step_idx, "<parse-failed>", "<parse-failed>"
        )
        artifact_writer.save_verdict(step_idx, False, parse_fail_reason)
    return (
        VerificationResult(passed=False, reason=parse_fail_reason),
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
    workspace: FileWorkspace | None = None,
    step_idx: int = 0,
    pause_handler: Callable[[str], bool] | None = None,
    variables: VariableStore | None = None,
    step_timeout_seconds: float = 0.0,
    stuck_step_threshold: int = 0,
    browser_bridge: BrowserBridge | None = None,
    routing_hint: str = "",
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
    # Drain the workspace's feed buffer ONCE per step. consume_feed() is
    # destructive, so we own the bytes here and pass them down to every
    # _attempt_step call — parse retries AND replan attempts all see the
    # same captured images. The buffer would otherwise empty after the
    # first attempt, silently losing CAPTURE_FOR_AI context on replans.
    extra_images = workspace.consume_feed() if workspace is not None else []
    # Number of real (non-PAUSE) attempts that have been consumed so far.
    # PAUSE iterations do NOT advance this counter — a PAUSE means "human
    # intervention changed the screen, so let me try again fresh" and must
    # not eat into the replan budget. The old implementation incremented
    # then decremented on PAUSE, which was fragile (an off-by-one near the
    # final slot could make `while attempt_idx < total_attempts` exit one
    # iteration early and starve the last replan).
    attempts_used = 0
    pauses_so_far = 0
    # Hard safety: a single step shouldn't require more than this many human
    # interventions; otherwise we're probably stuck in a PAUSE loop.
    max_pauses_per_step = 10

    # Tier 4 reliability state: track a wall-clock deadline for the whole
    # step and a rolling history of post-action screenshot fingerprints.
    # When the same screen appears N times in a row, the agent is stuck
    # and should fail fast instead of burning the whole replan budget on
    # an unresponsive UI.
    step_deadline: float | None = None
    if step_timeout_seconds and step_timeout_seconds > 0:
        step_deadline = time.monotonic() + step_timeout_seconds
    recent_signatures: list[str] = []

    while attempts_used < total_attempts:
        # Tier 4 — wall-clock timeout: if the user gave us a per-step budget
        # and we've already spent it, stop here so the next step can run.
        if step_deadline is not None and time.monotonic() >= step_deadline:
            timeout = VerificationResult(
                passed=False,
                reason=(
                    f"Step exceeded {step_timeout_seconds:.0f}s wall-clock "
                    f"budget after {attempts_used} attempt(s). "
                    f"Last failure: "
                    f"{last_verdict.reason if last_verdict else 'n/a'}"
                ),
            )
            log.error("%s", timeout.reason)
            history.record(
                step, last_action_text, passed=False, reason=timeout.reason
            )
            return timeout
        # `attempt_idx` is the 1-indexed number of the attempt we're about
        # to make (for logging / budget messages). The first attempt is
        # not a replan; attempts 2+ are.
        attempt_idx = attempts_used + 1
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

        # Log the attempt number plus an unambiguous replan-budget label.
        # ``total_attempts = max_replans + 1`` (the initial attempt is not
        # a replan). On attempt 1 we have all replans available; on
        # attempt 2 we're using replan #1; etc. We use a local
        # ``replan_budget`` for log labels (not ``max_replans``) so the
        # function parameter name doesn't get shadowed — code-smell only,
        # caught by Devin Review.
        replan_budget = total_attempts - 1
        if not is_replan_attempt:
            log.info(
                "Step attempt %d/%d (initial — %d replan(s) available if this fails)",
                attempt_idx,
                total_attempts,
                replan_budget,
            )
        else:
            replan_number = attempt_idx - 1
            log.info(
                "Step attempt %d/%d (replan %d/%d — %d replan(s) left after this)",
                attempt_idx,
                total_attempts,
                replan_number,
                replan_budget,
                replan_budget - replan_number,
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
            workspace=workspace,
            step_idx=step_idx,
            extra_images=extra_images,
            variables=variables,
            browser_bridge=browser_bridge,
            routing_hint=routing_hint,
        )

        # PAUSE: ask the human, then loop back without consuming the
        # replan budget — the screen state has changed (because the human
        # acted), so a fresh plan attempt is the right thing. Crucially,
        # `attempts_used` is NOT incremented on PAUSE, so the `while`
        # condition stays fresh and we don't starve the last replan slot
        # (only `pauses_so_far` / `max_pauses_per_step` bounds the loop
        # when PAUSE storms would otherwise spin forever).
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
            continue

        # Non-PAUSE outcome: this was a real attempt. Consume one slot
        # of both the per-step budget (via `attempts_used`) and — when
        # applicable — the global replan budget.
        attempts_used += 1
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

        # Tier 4 — stuck-step detection: if the post-action screenshot
        # looks identical to the previous N attempts on this step, our
        # actions aren't moving the UI. Bail early instead of burning
        # the rest of the replan budget on the same dead-end.
        if stuck_step_threshold and stuck_step_threshold > 0:
            try:
                signature_img = capture_screenshot()
                signature = image_signature(signature_img)
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("stuck-step: screenshot signature failed: %s", exc)
            else:
                recent_signatures.append(signature)
                # Keep the trailing window so memory doesn't grow.
                if len(recent_signatures) > stuck_step_threshold:
                    recent_signatures = recent_signatures[-stuck_step_threshold:]
                if (
                    len(recent_signatures) >= stuck_step_threshold
                    and len(set(recent_signatures)) == 1
                ):
                    # Include the action the planner kept emitting so a
                    # postmortem reader can see WHAT the agent tried,
                    # not just "it failed". The two most common failure
                    # modes are: (a) TYPE without a focused field —
                    # action_text says TYPE [...], (b) CLICK on the
                    # wrong coordinates — action_text says CLICK [x,y].
                    # Either way, surfacing the command makes the
                    # next-action diagnosis obvious.
                    stuck = VerificationResult(
                        passed=False,
                        reason=(
                            f"Step appears stuck — last "
                            f"{stuck_step_threshold} attempts produced "
                            f"identical screen state. "
                            f"Last action attempted: {action_text}. "
                            f"Last failure: {verdict.reason}"
                        ),
                    )
                    log.error("%s", stuck.reason)
                    history.record(
                        step,
                        last_action_text,
                        passed=False,
                        reason=stuck.reason,
                    )
                    return stuck

    # Budget exhausted.
    final = last_verdict or VerificationResult(
        passed=False, reason="No attempts were made (internal error)."
    )
    history.record(step, last_action_text, passed=False, reason=final.reason)
    return final


def run(
    config: Config,
    resume: bool = False,
    csv_override: Path | None = None,
    cli_file_mode: FileMode | None = None,
    cli_workdir: Path | None = None,
    pause_handler: Callable[[str], bool] | None = None,
) -> int:
    try:
        steps = load_steps(
            config.tasks_file,
            csv_override=csv_override,
            skills_dir=config.skills_dir,
        )
    except (TasksLoadError, FileNotFoundError) as exc:
        log.error("Failed to load tasks file %s: %s", config.tasks_file, exc)
        print(f"[tasks error] {exc}", file=sys.stderr)
        return 2
    if not steps:
        log.error("No steps found in %s", config.tasks_file)
        return 2

    # Inspect what's actually in the tasks file so we only prompt for
    # features the run actually needs. A simple "press Win, type Notepad"
    # task should not have to answer questions about file modes.
    features = inspect_features(steps)
    summary = format_features_summary(features, total_steps=len(steps))
    print(summary)
    log.info("%s", summary)

    workspace: FileWorkspace | None = None
    if features.uses_files:
        # Only ask about download persistence when the tasks file actually
        # uses file primitives. CLI / env values still take precedence so
        # unattended (.exe) runs never prompt.
        file_mode, workdir = resolve_mode(
            cli_mode=cli_file_mode,
            cli_workdir=cli_workdir,
            env_mode=config.file_mode,
            env_workdir=config.workdir,
            interactive=True,
        )
        try:
            workspace = FileWorkspace.create(mode=file_mode, workdir=workdir)
        except (OSError, ValueError) as exc:
            log.error("Could not set up file workspace: %s", exc)
            print(f"[workspace error] {exc}", file=sys.stderr)
            return 2
        log.info(
            "File mode: %s%s",
            workspace.mode.value,
            f" (workdir={workspace.root})" if workspace.root else "",
        )

    # Variable store: gated on tasks-file feature inspection so a simple
    # "press Win, type Notepad" run never sets one up. When the user
    # resumes from a checkpoint, the store is rehydrated from the
    # serialized snapshot so values learned before the crash survive.
    variables: VariableStore | None = None
    if features.uses_variables:
        variables = VariableStore()
        log.info(
            "Variable store enabled "
            "(REMEMBER=%d, RECALL=%d, {{var.*}}=%d)",
            features.remember_count,
            features.recall_count,
            features.var_placeholder_count,
        )

    # Browser fast-path bridge: connect to Chrome's CDP debug port so
    # BROWSER_GO / BROWSER_CLICK / BROWSER_FILL primitives can drive the
    # active tab directly. Gated on BOTH ``BROWSER_FAST_PATH=true`` (env
    # opt-in) AND a feature hint in the tasks file — connecting to CDP
    # for a tasks file that never asks for it would just add startup
    # latency. Failure to connect is non-fatal: every browser command
    # then synthesizes a clear FAIL with a message telling the planner
    # to use the visual primitives instead.
    browser_bridge: BrowserBridge | None = None
    if config.browser_fast_path and features.uses_browser_fast_path:
        browser_bridge = BrowserBridge(
            host=config.browser_cdp_host,
            port=config.browser_cdp_port,
        )
        if browser_bridge.connect():
            log.info(
                "Browser fast-path enabled "
                "(BROWSER_GO=%d, BROWSER_CLICK=%d, BROWSER_FILL=%d)",
                features.browser_go_count,
                features.browser_click_count,
                features.browser_fill_count,
            )
        else:
            log.warning(
                "BROWSER_FAST_PATH=true but Chrome CDP at %s:%d is not "
                "reachable. Browser commands will fail; use the visual "
                "primitives instead, or relaunch Chrome with "
                "--remote-debugging-port=%d.",
                config.browser_cdp_host,
                config.browser_cdp_port,
                config.browser_cdp_port,
            )
            # Keep the (unconnected) bridge around so the executor's
            # is_connected() check fails fast with a clear message
            # rather than crashing on a None reference. close() is a
            # no-op when never connected.

    # Construct VLM + supporting infra BEFORE checkpoint validation,
    # because the decomposer (which expands the step list) needs the
    # VLM client and MUST run before checkpoint validation. If we
    # validated the checkpoint against the pre-decomposition step
    # count, every resume after a decomposition-expanded run would be
    # rejected as "stale" (the saved total_steps is the post-decomp
    # count). See PR #25 review feedback.
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
        image_max_dim=config.vlm_image_max_dim,
        image_quality=config.vlm_image_quality,
        skip_identical_frames=config.vlm_skip_identical_frames,
        # Only advertise BROWSER_* commands to the planner when the
        # bridge is actually connected. Otherwise the planner would
        # emit BROWSER_GO on every navigation step, every attempt
        # would FAIL, and replan budget would burn on doomed commands.
        enable_browser_fast_path=(
            browser_bridge is not None and browser_bridge.is_connected()
        ),
    )
    history = History(window=config.history_window)
    replan_counter = ReplanCounter(total_max=config.max_total_replans)
    artifact_writer = ArtifactWriter.create(
        enabled=config.save_run_artifacts,
        base_dir=config.run_artifacts_dir,
    )

    # Task decomposition: one Gemini call at run start that splits any
    # compound natural-language steps into atomic substeps. Runs BEFORE
    # checkpoint validation so the validation compares against the
    # post-decomposition step count — matching what was saved on the
    # previous run. Falls back gracefully on error (original step list
    # returned, run continues).
    decomposition_mode = parse_decomposition_mode(
        config.task_decomposition_mode
    )
    steps = apply_decomposer(
        steps,
        mode=decomposition_mode,
        client=vlm,
    )

    # Checkpoint handling — runs against the POST-decomposition step
    # count so a previous run's saved total_steps lines up.
    existing_state = load_state(config.state_file) if resume else None
    start_idx = 1
    if existing_state is not None:
        if (
            existing_state.tasks_file == str(config.tasks_file)
            and existing_state.total_steps == len(steps)
            and 0 <= existing_state.last_completed_step < len(steps)
        ):
            start_idx = existing_state.last_completed_step + 1
            # If we're about to resume INSIDE an IF/ELSE/END_IF block,
            # rewind to that block's if_begin so the condition is
            # re-evaluated and the branch_decisions table is repopulated.
            # Without this the agent would have no record of which branch
            # to take and would run BOTH branches as if no IF existed.
            if start_idx <= len(steps):
                resumed_step = steps[start_idx - 1]
                active = resumed_step.active_block_id
                if active is not None:
                    rewind_idx = _find_if_begin_index(steps, active)
                    if rewind_idx is not None and rewind_idx < start_idx:
                        log.info(
                            "Resume point (step %d) is inside IF block "
                            "%d; rewinding to step %d (the IF) so the "
                            "condition can be re-evaluated.",
                            start_idx,
                            active,
                            rewind_idx,
                        )
                        start_idx = rewind_idx
                        # Reset last_completed_step so subsequent
                        # advance() calls track the rewound position.
                        # Without this, the checkpoint counter drifts
                        # ahead of the step index across multi-resume
                        # cycles and steps get silently skipped.
                        existing_state = AgentState(
                            version=existing_state.version,
                            tasks_file=existing_state.tasks_file,
                            total_steps=existing_state.total_steps,
                            last_completed_step=rewind_idx - 1,
                            variables=dict(existing_state.variables),
                        )
            log.info(
                "Resuming from checkpoint: step %d/%d (file=%s)",
                start_idx,
                len(steps),
                config.state_file,
            )
            state = existing_state
            # Rehydrate variables from the checkpoint when this run uses
            # them. Skipping the rehydrate when ``features.uses_variables``
            # is False keeps the unused store empty + drops any stale
            # entries that no longer correspond to a REMEMBER step.
            if variables is not None and existing_state.variables is not None:
                # Use ``is not None`` rather than truthy: an empty
                # ``{}`` is a valid checkpoint (the previous run hadn't
                # called REMEMBER yet) and rehydrating it preserves
                # the run's identity. The truthy check skipped this
                # legitimate case and silently dropped the persisted
                # store.
                variables = VariableStore.from_dict(existing_state.variables)
                log.info(
                    "Restored %d variable(s) from checkpoint: %s",
                    len(variables),
                    variables.summary(),
                )
        else:
            # Step-count mismatch on resume usually means either the
            # tasks file changed OR the decomposer's non-deterministic
            # expansion came out to a different count than last run.
            # Either way the saved offset can't be trusted; we restart
            # from step 1 with a clear log line.
            log.warning(
                "Ignoring stale checkpoint at %s (tasks file or step "
                "count changed: was total_steps=%d, now %d).",
                config.state_file,
                existing_state.total_steps,
                len(steps),
            )
            state = AgentState.initial(config.tasks_file, len(steps))
    else:
        state = AgentState.initial(config.tasks_file, len(steps))

    # Smart task router: one Gemini call to classify every step's
    # complexity and (where possible) suggest a literal command. The
    # planner then sees the hint as advisory context. Manual ``[tag]``
    # annotations on tasks-file lines are already attached to the
    # TaskSteps at load time and are NEVER overwritten by the auto
    # router — the user's explicit annotation always wins.
    routing_mode = parse_routing_mode(config.task_routing_mode)
    if routing_mode == RoutingMode.AUTO:
        # Only AUTO actually calls Gemini. MANUAL mode's hints were
        # already attached at load time and ``apply_router`` is a no-op
        # there — calling it just iterates the steps to produce a list
        # of ``None`` hints we'd then re-attach. Skip the round-trip
        # entirely; manual annotations stay intact on the TaskStep
        # objects we got from ``load_steps``.
        bridge_ready = (
            browser_bridge is not None and browser_bridge.is_connected()
        )
        auto_hints = apply_router(
            [s.text for s in steps],
            mode=routing_mode,
            client=vlm,
            enable_browser_fast_path=bridge_ready,
        )
        steps = attach_routing_hints(steps, auto_hints)
    if routing_mode != RoutingMode.OFF:
        # Log the resolved routing summary in both AUTO and MANUAL
        # so the user can see what the planner is going to receive
        # — even when the router itself didn't run.
        _log_routing_summary(steps, mode=routing_mode)

    log.info(
        "Loaded %d step(s) from %s (starting at step %d)",
        len(steps),
        config.tasks_file,
        start_idx,
    )

    # Track which side of each IF block was taken. Built up as if_begin
    # steps run; consulted by every step inside the block to decide
    # whether to execute or skip. Skipped steps still tick the step
    # counter so total_steps math (checkpoints, "x/N") stays consistent.
    branch_decisions: dict[int, bool] = {}

    success = False
    try:
        for idx in range(start_idx, len(steps) + 1):
            task_step: TaskStep = steps[idx - 1]
            step_text = task_step.text
            # Substitute ``{{var.<name>}}`` placeholders just before
            # running the step, NOT at load time — variables are populated
            # as the run progresses, and a step further down may reference
            # one set by a step above. This is intentionally separate from
            # the ``{{row.<field>}}`` substitution which IS done at load
            # time (the CSV is fully known up-front).
            if variables is not None and text_uses_variables(step_text):
                try:
                    step_text = substitute_variables(step_text, variables)
                except UnknownVariableError as exc:
                    msg = (
                        f"\n[!] HALT at step {idx}/{len(steps)}: "
                        f"unresolved variable in step text: {exc}\n"
                        f"    Step: {task_step.text}\n"
                        f"    Add a REMEMBER step before this one, or "
                        f"provide a default like {{{{var.name|default}}}}.\n"
                    )
                    print(msg, file=sys.stderr)
                    log.error("Unresolved variable: %s", exc)
                    return 1
            log.info("=" * 60)
            if task_step.row_index is not None:
                log.info(
                    "Step %d/%d (row %d of %s): %s",
                    idx,
                    len(steps),
                    task_step.row_index,
                    task_step.csv_name,
                    step_text,
                )
            else:
                log.info("Step %d/%d: %s", idx, len(steps), step_text)
            log.info("=" * 60)
            if workspace is not None:
                workspace.begin_step(
                    row_index=task_step.row_index,
                    csv_name=task_step.csv_name,
                )

            # Control-flow directives (IF / ELSE / END_IF) and step-skip
            # decisions are handled BEFORE the planner runs — they are
            # synthetic "non-action" steps that consume a step counter
            # slot but never invoke run_step.
            control_result = _maybe_handle_control(
                task_step=task_step,
                idx=idx,
                total_steps=len(steps),
                vlm=vlm,
                geometry=geometry,
                branch_decisions=branch_decisions,
                wait_until_timeout_seconds=config.wait_until_timeout_seconds,
                wait_until_poll_seconds=config.wait_until_poll_seconds,
            )
            if control_result is not None:
                # Always record the verdict in the run summary, even
                # for failed control steps (so the artifact bundle
                # shows the timeout reason).
                artifact_writer.append_summary(
                    step_idx=idx,
                    step_text=step_text,
                    passed=control_result.passed,
                    reason=control_result.reason,
                )
                # Failed control directive (only WAIT_UNTIL timeout
                # can produce this) must HALT *without* advancing the
                # checkpoint — otherwise --resume would skip past the
                # timed-out step. Mirrors the normal-step failure path
                # below, which also halts pre-advance.
                if not control_result.passed:
                    msg = (
                        f"\n[!] HALT at step {idx}/{len(steps)}: "
                        f"{step_text}\n"
                        f"    Reason: {control_result.reason}\n"
                        "    The agent has stopped to prevent runaway actions."
                    )
                    print(msg, file=sys.stderr)
                    log.error("Halting execution: %s", control_result.reason)
                    return 1
                # Passed control / skipped step: persist progress and
                # move on.
                state = state.advance()
                if variables is not None:
                    state = state.with_variables(variables.to_dict())
                try:
                    save_state(config.state_file, state)
                except OSError as exc:
                    log.warning(
                        "Failed to save checkpoint to %s: %s",
                        config.state_file,
                        exc,
                    )
                continue

            # Routing hint is per-step and frozen at run start; pass the
            # rendered string so the planner sees it embedded in its
            # user prompt. Empty string when no hint is attached.
            hint_text = (
                task_step.routing_hint.render_for_prompt()
                if task_step.routing_hint is not None
                else ""
            )

            result = run_step(
                step=step_text,
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
                workspace=workspace,
                step_idx=idx,
                variables=variables,
                step_timeout_seconds=config.step_timeout_seconds,
                stuck_step_threshold=config.stuck_step_threshold,
                browser_bridge=browser_bridge,
                routing_hint=hint_text,
                pause_handler=pause_handler,
            )

            # action_text is resolved inside the writer from the most recent
            # save_plan() call for this step — no placeholder needed.
            artifact_writer.append_summary(
                step_idx=idx,
                step_text=step_text,
                passed=result.passed,
                reason=result.reason,
            )

            if not result.passed:
                msg = (
                    f"\n[!] HALT at step {idx}/{len(steps)}: {step_text}\n"
                    f"    Reason: {result.reason}\n"
                    f"    (Exhausted replan budget of {config.max_replans_per_step}.)\n"
                    f"    Run `python -m agent --resume` after fixing the blocker to continue.\n"
                    "    The agent has stopped to prevent runaway actions."
                )
                print(msg, file=sys.stderr)
                log.error("Halting execution: %s", result.reason)
                return 1

            # Commit progress to disk before moving on. Snapshot the
            # variable store too so a resume can pick up where we left
            # off — not just on which step number, but with the values
            # learned from REMEMBER calls so far.
            state = state.advance()
            if variables is not None:
                state = state.with_variables(variables.to_dict())
            try:
                save_state(config.state_file, state)
            except OSError as exc:
                log.warning(
                    "Failed to save checkpoint to %s: %s", config.state_file, exc
                )

            # RPD guard: halt cleanly between steps so the checkpoint is consistent.
            if rpd_guard.should_halt():
                print("\n[!] " + rpd_guard.halt_message(), file=sys.stderr)
                log.error("Halting execution: %s", rpd_guard.halt_message())
                return 1

        log.info("All %d step(s) completed successfully.", len(steps))
        print(f"[ok] All {len(steps)} step(s) completed successfully.")
        success = True
        return 0
    finally:
        # Workspace cleanup is the last thing we do — temp dirs are wiped on
        # success, kept on failure so the user can inspect downloads, and
        # save / feed modes are no-ops here. Skipped entirely when the
        # tasks file didn't use any file primitives.
        if workspace is not None:
            workspace.finalize(success=success)
        # Drop the CDP websocket. ``close()`` is a no-op if we never
        # connected, so this is safe whether or not the bridge was used.
        if browser_bridge is not None:
            browser_bridge.close()
