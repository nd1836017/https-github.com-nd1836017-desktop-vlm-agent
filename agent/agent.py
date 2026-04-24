"""Main agent loop: tasks.txt -> VLM -> parse -> execute -> verify."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import Config
from .executor import execute
from .parser import parse_command
from .screen import ScreenGeometry, capture_screenshot, detect_geometry
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


def run_step(
    step: str,
    vlm: GeminiClient,
    geometry: ScreenGeometry,
    animation_buffer: float,
    max_retries: int,
) -> VerificationResult:
    """Run one step, retrying up to `max_retries` times on parse failure."""
    attempts = 0
    last_error: str = ""
    while attempts <= max_retries:
        attempts += 1
        log.info("Step attempt %d/%d: %s", attempts, max_retries + 1, step)

        screenshot = capture_screenshot()
        raw = vlm.plan_action(step, screenshot)
        cmd = parse_command(raw)
        if cmd is None:
            last_error = f"Could not parse VLM response: {raw!r}"
            log.warning("%s — retrying" if attempts <= max_retries else "%s", last_error)
            continue

        execute(cmd, geometry, animation_buffer_seconds=animation_buffer)

        post_screenshot = capture_screenshot()
        verdict = vlm.verify(step, post_screenshot)
        log.info("Verify: %s", verdict.reason)
        return verdict

    # Exhausted retries without a parseable command.
    return VerificationResult(
        passed=False,
        reason=f"Parse failure after {max_retries + 1} attempts: {last_error}",
    )


def run(config: Config) -> int:
    steps = read_tasks(config.tasks_file)
    if not steps:
        log.error("No steps found in %s", config.tasks_file)
        return 2

    geometry = detect_geometry()
    vlm = GeminiClient(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
    )

    log.info("Loaded %d step(s) from %s", len(steps), config.tasks_file)

    for idx, step in enumerate(steps, start=1):
        log.info("=" * 60)
        log.info("Step %d/%d: %s", idx, len(steps), step)
        log.info("=" * 60)

        result = run_step(
            step=step,
            vlm=vlm,
            geometry=geometry,
            animation_buffer=config.animation_buffer_seconds,
            max_retries=config.max_step_retries,
        )

        if not result.passed:
            msg = (
                f"\n[!] HALT at step {idx}/{len(steps)}: {step}\n"
                f"    Reason: {result.reason}\n"
                "    The agent has stopped to prevent runaway actions."
            )
            print(msg, file=sys.stderr)
            log.error("Halting execution: %s", result.reason)
            return 1

    log.info("All %d step(s) completed successfully.", len(steps))
    print(f"[ok] All {len(steps)} step(s) completed successfully.")
    return 0
