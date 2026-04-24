"""Rolling short-term memory of recently executed steps.

The planner-VLM sees a compact summary of the last N `(step, action, verdict)`
records so it can reason about what has already happened and avoid loops.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from .parser import ClickCommand, Command, PressCommand, TypeCommand


@dataclass(frozen=True)
class StepRecord:
    step: str
    action_text: str
    passed: bool
    reason: str


def render_command(cmd: Command) -> str:
    """Render a parsed command back as its canonical `CLICK [...]` form."""
    if isinstance(cmd, ClickCommand):
        return f"CLICK [{cmd.x},{cmd.y}]"
    if isinstance(cmd, PressCommand):
        return f"PRESS [{cmd.key}]"
    if isinstance(cmd, TypeCommand):
        return f"TYPE [{cmd.text}]"
    return str(cmd)


class History:
    """Bounded FIFO of recent step records."""

    def __init__(self, window: int = 5) -> None:
        if window < 0:
            raise ValueError("history window must be non-negative")
        self._window = window
        self._records: deque[StepRecord] = deque(maxlen=window) if window > 0 else deque(maxlen=0)

    @property
    def window(self) -> int:
        return self._window

    def record(self, step: str, action_text: str, passed: bool, reason: str) -> None:
        if self._window == 0:
            return
        self._records.append(
            StepRecord(
                step=step,
                action_text=action_text,
                passed=passed,
                reason=reason.strip(),
            )
        )

    def __iter__(self) -> Iterable[StepRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def summary(self) -> str:
        """Render recent records as a compact text block for prompt injection.

        Returns an empty string when there is no history to report.
        """
        if not self._records:
            return ""
        lines: list[str] = []
        for idx, rec in enumerate(self._records, start=1):
            verdict = "PASS" if rec.passed else "FAIL"
            # Keep each line short; the VLM only needs the gist.
            reason = rec.reason[:120]
            lines.append(
                f"{idx}. step={rec.step!r} action={rec.action_text} -> {verdict}"
                + (f" ({reason})" if reason else "")
            )
        return "\n".join(lines)
