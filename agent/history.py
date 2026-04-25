"""Rolling short-term memory of recently executed steps.

The planner-VLM sees a compact summary of the last N `(step, action, verdict)`
records so it can reason about what has already happened and avoid loops.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from .parser import (
    AttachFileCommand,
    CaptureForAiCommand,
    ClickCommand,
    ClickTextCommand,
    Command,
    DoubleClickCommand,
    DownloadCommand,
    DragCommand,
    MoveToCommand,
    PauseCommand,
    PressCommand,
    RecallCommand,
    RememberCommand,
    RightClickCommand,
    ScrollCommand,
    TypeCommand,
    WaitCommand,
)


@dataclass(frozen=True)
class StepRecord:
    step: str
    action_text: str
    passed: bool
    reason: str


def render_command(cmd: Command, *, redact_type: bool = False) -> str:
    """Render a parsed command back as its canonical `COMMAND [...]` form.

    When ``redact_type`` is True, `TYPE` commands are rendered as
    ``TYPE [<REDACTED, N chars>]`` to avoid leaking sensitive strings into
    log files or history summaries. Non-TYPE commands are never affected.
    """
    if isinstance(cmd, ClickCommand):
        return f"CLICK [{cmd.x},{cmd.y}]"
    if isinstance(cmd, DoubleClickCommand):
        return f"DOUBLE_CLICK [{cmd.x},{cmd.y}]"
    if isinstance(cmd, RightClickCommand):
        return f"RIGHT_CLICK [{cmd.x},{cmd.y}]"
    if isinstance(cmd, PressCommand):
        return f"PRESS [{cmd.key}]"
    if isinstance(cmd, TypeCommand):
        if redact_type:
            return f"TYPE [<REDACTED, {len(cmd.text)} chars>]"
        return f"TYPE [{cmd.text}]"
    if isinstance(cmd, ScrollCommand):
        return f"SCROLL [{cmd.direction},{cmd.amount}]"
    if isinstance(cmd, DragCommand):
        return f"DRAG [{cmd.x1},{cmd.y1},{cmd.x2},{cmd.y2}]"
    if isinstance(cmd, MoveToCommand):
        return f"MOVE_TO [{cmd.x},{cmd.y}]"
    if isinstance(cmd, WaitCommand):
        return f"WAIT [{cmd.seconds}]"
    if isinstance(cmd, ClickTextCommand):
        return f"CLICK_TEXT [{cmd.label}]"
    if isinstance(cmd, PauseCommand):
        return f"PAUSE [{cmd.reason}]"
    if isinstance(cmd, DownloadCommand):
        if cmd.filename:
            return f"DOWNLOAD [{cmd.url}, {cmd.filename}]"
        return f"DOWNLOAD [{cmd.url}]"
    if isinstance(cmd, AttachFileCommand):
        return f"ATTACH_FILE [{cmd.filename}]"
    if isinstance(cmd, CaptureForAiCommand):
        if cmd.filename:
            return f"CAPTURE_FOR_AI [{cmd.filename}]"
        return "CAPTURE_FOR_AI"
    if isinstance(cmd, RememberCommand):
        # Two forms: REMEMBER [name = literal] (literal stored as-is)
        # and REMEMBER [name] (extracted from the screen via VLM).
        # Don't render the literal value if redact_type is set — it
        # may contain sensitive data the user pasted.
        if not cmd.from_screen:
            if redact_type:
                return (
                    f"REMEMBER [{cmd.name} = "
                    f"<REDACTED, {len(cmd.literal_value)} chars>]"
                )
            return f"REMEMBER [{cmd.name} = {cmd.literal_value}]"
        return f"REMEMBER [{cmd.name}]"
    if isinstance(cmd, RecallCommand):
        return f"RECALL [{cmd.name}]"
    return str(cmd)


class History:
    """Bounded FIFO of recent step records.

    ``window`` controls how many records are retained for prompt injection:

    - ``window > 0`` (default 5): rolling window; the deque keeps the last
      ``window`` records and drops older ones.
    - ``window == 0`` **disables short-term memory entirely** — ``record()``
      is a no-op, ``__iter__`` yields nothing, ``summary()`` returns an
      empty string. This is the documented opt-out from
      ``HISTORY_WINDOW=0`` in ``.env``; it is *not* an alias for
      "unlimited".
    - ``window < 0`` raises ``ValueError`` — there is no "unlimited" mode.

    The intentional silent no-op for ``window == 0`` is the difference
    callers should know about. If you want a rolling memory, use any
    positive integer; if you want to turn memory off, use 0.
    """

    def __init__(self, window: int = 5) -> None:
        if window < 0:
            raise ValueError(
                "history window must be non-negative; use 0 to disable"
            )
        self._window = window
        # When disabled, keep an empty deque rather than ``None`` so all
        # iteration / len / summary calls remain branch-free at the call
        # site. ``record()`` returns early so nothing is ever appended.
        self._records: deque[StepRecord] = deque(
            maxlen=window if window > 0 else 0
        )

    @property
    def window(self) -> int:
        return self._window

    @property
    def disabled(self) -> bool:
        """True when ``window == 0`` and short-term memory is off."""
        return self._window == 0

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

    def __iter__(self) -> Iterator[StepRecord]:
        # Short-circuit when disabled — no records can ever exist, but
        # being explicit makes the no-op behavior obvious to readers and
        # avoids relying on the empty-deque coincidence.
        if self._window == 0:
            return iter(())
        return iter(self._records)

    def __len__(self) -> int:
        if self._window == 0:
            return 0
        return len(self._records)

    def summary(self) -> str:
        """Render recent records as a compact text block for prompt injection.

        Returns an empty string when there is no history to report (which
        includes the ``window == 0`` "disabled" mode — see class docstring).
        """
        if self._window == 0 or not self._records:
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
