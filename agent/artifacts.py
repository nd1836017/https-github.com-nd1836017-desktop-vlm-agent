"""Per-step run artifacts for postmortems.

When `SAVE_RUN_ARTIFACTS=true`, every run gets its own timestamped directory
under `RUN_ARTIFACTS_DIR` (default `runs/`). For each step we write:

    runs/<ts>/step_03_before.png      — screenshot before the action
    runs/<ts>/step_03_after.png       — screenshot after the action
    runs/<ts>/step_03_plan.txt        — VLM plan response (raw)
    runs/<ts>/step_03_verdict.txt     — VLM verify response + parsed verdict
    runs/<ts>/summary.json            — per-step pass/fail summary, written
                                        incrementally as the run progresses.

Disabled by default to avoid filling disk on long runs. The writer is
defensive — any IOError logs a WARNING and is otherwise swallowed; saving
artifacts must NEVER block or crash the agent.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class ArtifactWriter:
    """Best-effort writer for per-step artifacts."""

    enabled: bool = False
    base_dir: Path = field(default_factory=lambda: Path("runs"))
    run_dir: Path | None = None
    _summary: list[dict] = field(default_factory=list)
    # Remember the last action text seen for each step so `append_summary`
    # can record what actually ran, even when the caller doesn't have it
    # handy (the run-level `run()` only sees the verdict, not the action).
    _last_action_by_step: dict[int, str] = field(default_factory=dict)

    @classmethod
    def create(cls, *, enabled: bool, base_dir: str | Path) -> ArtifactWriter:
        writer = cls(enabled=enabled, base_dir=Path(base_dir).expanduser())
        if not enabled:
            return writer
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            writer.run_dir = writer.base_dir / ts
            writer.run_dir.mkdir(parents=True, exist_ok=True)
            log.info("Run artifacts will be written to %s", writer.run_dir)
        except OSError as exc:
            log.warning("Failed to create artifact dir %s: %s", writer.base_dir, exc)
            writer.enabled = False
        return writer

    def _write_text(self, name: str, content: str) -> None:
        if not self.enabled or self.run_dir is None:
            return
        try:
            (self.run_dir / name).write_text(content, encoding="utf-8")
        except OSError as exc:
            log.warning("Failed to write artifact %s: %s", name, exc)

    def _write_image(self, name: str, image: Image.Image) -> None:
        if not self.enabled or self.run_dir is None:
            return
        try:
            image.save(self.run_dir / name, format="PNG", optimize=True)
        except OSError as exc:
            log.warning("Failed to write artifact image %s: %s", name, exc)

    def save_before(self, step_idx: int, screenshot: Image.Image) -> None:
        self._write_image(f"step_{step_idx:03d}_before.png", screenshot)

    def save_after(self, step_idx: int, screenshot: Image.Image) -> None:
        self._write_image(f"step_{step_idx:03d}_after.png", screenshot)

    def save_plan(self, step_idx: int, raw_text: str, action_text: str) -> None:
        # Stash the action text so append_summary() can use it without the
        # run loop having to plumb it through. The latest call wins, which
        # matches "what actually executed for this step".
        self._last_action_by_step[step_idx] = action_text
        body = f"# action_text\n{action_text}\n\n# raw VLM response\n{raw_text}\n"
        self._write_text(f"step_{step_idx:03d}_plan.txt", body)

    def save_verdict(self, step_idx: int, passed: bool, reason: str) -> None:
        verdict = "PASS" if passed else "FAIL"
        body = f"VERDICT: {verdict}\nREASON: {reason}\n"
        self._write_text(f"step_{step_idx:03d}_verdict.txt", body)

    def append_summary(
        self,
        step_idx: int,
        step_text: str,
        passed: bool,
        reason: str,
        action_text: str | None = None,
    ) -> None:
        """Append one step's outcome to the rolling summary.json.

        If ``action_text`` is None, fall back to the last action seen for
        this step via ``save_plan``. This lets the run loop call
        ``append_summary`` without having to plumb action_text through
        ``run_step``'s return value — the writer already saw it when the
        plan was saved.
        """
        if not self.enabled or self.run_dir is None:
            return
        resolved_action = (
            action_text
            if action_text is not None
            else self._last_action_by_step.get(step_idx, "<no-action-recorded>")
        )
        self._summary.append(
            {
                "step_idx": step_idx,
                "step": step_text,
                "action": resolved_action,
                "passed": passed,
                "reason": reason,
            }
        )
        try:
            (self.run_dir / "summary.json").write_text(
                json.dumps(self._summary, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("Failed to write summary.json: %s", exc)
