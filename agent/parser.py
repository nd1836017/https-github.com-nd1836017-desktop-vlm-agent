"""Regex-based command parser for VLM responses.

Supported commands:
    CLICK [X,Y]                 — click at normalized coordinates (0-1000 grid).
    DOUBLE_CLICK [X,Y]          — double-click at normalized coordinates.
    RIGHT_CLICK [X,Y]           — right-click at normalized coordinates.
    PRESS [KEY]                 — single key or '+'-separated hotkey.
    TYPE [TEXT]                 — literal text to type.
    SCROLL [DIR, AMOUNT]        — scroll up/down by N "clicks" (AMOUNT is a
                                  positive integer; DIR is up/down).
    DRAG [X1,Y1,X2,Y2]          — press at (X1,Y1), drag to (X2,Y2), release.
    MOVE_TO [X,Y]               — move mouse to (X,Y) without clicking (hover).
    WAIT [SECONDS]              — sleep N seconds (float). Capped by the
                                  executor at WAIT_MAX_SECONDS.
    CLICK_TEXT [LABEL]          — click the center of the on-screen text that
                                  best matches LABEL. Uses OCR and falls back
                                  to replan when no match is found.

The parser is defensive: it tolerates surrounding conversational text,
case variations, and a few common bracket omissions. It never raises on
malformed input — instead it returns None, and the caller is expected
to retry the step once.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

CommandType = Literal[
    "CLICK",
    "DOUBLE_CLICK",
    "RIGHT_CLICK",
    "PRESS",
    "TYPE",
    "SCROLL",
    "DRAG",
    "MOVE_TO",
    "WAIT",
    "CLICK_TEXT",
]


@dataclass(frozen=True)
class ClickCommand:
    kind: CommandType = "CLICK"
    x: int = 0
    y: int = 0


@dataclass(frozen=True)
class DoubleClickCommand:
    kind: CommandType = "DOUBLE_CLICK"
    x: int = 0
    y: int = 0


@dataclass(frozen=True)
class RightClickCommand:
    kind: CommandType = "RIGHT_CLICK"
    x: int = 0
    y: int = 0


@dataclass(frozen=True)
class PressCommand:
    kind: CommandType = "PRESS"
    key: str = ""


@dataclass(frozen=True)
class TypeCommand:
    kind: CommandType = "TYPE"
    text: str = ""


@dataclass(frozen=True)
class ScrollCommand:
    kind: CommandType = "SCROLL"
    direction: str = "down"  # "up" or "down"
    amount: int = 3


@dataclass(frozen=True)
class DragCommand:
    kind: CommandType = "DRAG"
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0


@dataclass(frozen=True)
class MoveToCommand:
    kind: CommandType = "MOVE_TO"
    x: int = 0
    y: int = 0


@dataclass(frozen=True)
class WaitCommand:
    kind: CommandType = "WAIT"
    seconds: float = 0.0


@dataclass(frozen=True)
class ClickTextCommand:
    kind: CommandType = "CLICK_TEXT"
    label: str = ""


Command = (
    ClickCommand
    | DoubleClickCommand
    | RightClickCommand
    | PressCommand
    | TypeCommand
    | ScrollCommand
    | DragCommand
    | MoveToCommand
    | WaitCommand
    | ClickTextCommand
)


# --- Strict patterns (bracketed form, preferred). -----------------------------

# Note the order matters: DOUBLE_CLICK / RIGHT_CLICK / MOVE_TO must be matched
# before CLICK, because CLICK's regex will also match "DOUBLE_CLICK [..,..]".
_DOUBLE_CLICK_RE = re.compile(
    r"DOUBLE[_\s]?CLICK\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)
_RIGHT_CLICK_RE = re.compile(
    r"RIGHT[_\s]?CLICK\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)
_MOVE_TO_RE = re.compile(
    r"MOVE[_\s]?TO\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)
_CLICK_RE = re.compile(
    r"(?<![A-Z_])CLICK\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)
_PRESS_RE = re.compile(r"PRESS\s*\[\s*([^\]\n]+?)\s*\]", re.IGNORECASE)
_TYPE_RE = re.compile(r"TYPE\s*\[\s*(.*?)\s*\]", re.IGNORECASE | re.DOTALL)
_SCROLL_RE = re.compile(
    r"SCROLL\s*\[\s*(up|down|UP|DOWN)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)
_DRAG_RE = re.compile(
    r"DRAG\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:,|->|=>|to)\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)
_WAIT_RE = re.compile(
    r"WAIT\s*\[\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*\]",
    re.IGNORECASE,
)
_CLICK_TEXT_RE = re.compile(
    r"CLICK[_\s]?TEXT\s*\[\s*(.*?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)


# --- Lenient fallbacks (missing brackets, common prose forms). ----------------

_CLICK_RE_LENIENT = re.compile(
    r"(?<![A-Z_])CLICK\s*\(?\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*\)?",
    re.IGNORECASE,
)
_PRESS_RE_LENIENT = re.compile(
    r"PRESS\s+(?:the\s+)?([A-Za-z0-9_+\-]+)(?:\s+key)?",
    re.IGNORECASE,
)
_TYPE_RE_LENIENT = re.compile(
    r'TYPE\s+"([^"\n]*)"|TYPE\s+\'([^\'\n]*)\'',
    re.IGNORECASE,
)


def parse_command(response: str) -> Command | None:
    """Extract the first recognized command from `response`.

    Returns None if no command can be parsed. Never raises.
    """
    if not response:
        return None

    try:
        # CLICK_TEXT must be tried before CLICK so the CLICK regex doesn't
        # mistakenly match prose inside a CLICK_TEXT label.
        m = _CLICK_TEXT_RE.search(response)
        if m:
            label = m.group(1).strip()
            if label:
                return ClickTextCommand(label=label)

        # Compound commands must be tried before CLICK so the CLICK pattern
        # doesn't eat "DOUBLE_CLICK [..]" or "RIGHT_CLICK [..]".
        m = _DOUBLE_CLICK_RE.search(response)
        if m:
            return DoubleClickCommand(x=int(m.group(1)), y=int(m.group(2)))

        m = _RIGHT_CLICK_RE.search(response)
        if m:
            return RightClickCommand(x=int(m.group(1)), y=int(m.group(2)))

        m = _MOVE_TO_RE.search(response)
        if m:
            return MoveToCommand(x=int(m.group(1)), y=int(m.group(2)))

        m = _CLICK_RE.search(response)
        if m:
            return ClickCommand(x=int(m.group(1)), y=int(m.group(2)))

        m = _SCROLL_RE.search(response)
        if m:
            direction = m.group(1).lower()
            amount = abs(int(m.group(2)))
            return ScrollCommand(direction=direction, amount=amount)

        m = _DRAG_RE.search(response)
        if m:
            return DragCommand(
                x1=int(m.group(1)),
                y1=int(m.group(2)),
                x2=int(m.group(3)),
                y2=int(m.group(4)),
            )

        m = _WAIT_RE.search(response)
        if m:
            return WaitCommand(seconds=float(m.group(1)))

        m = _PRESS_RE.search(response)
        if m:
            key = m.group(1).strip()
            if key:
                return PressCommand(key=key)

        m = _TYPE_RE.search(response)
        if m:
            return TypeCommand(text=m.group(1))

        # Lenient fallbacks.
        m = _CLICK_RE_LENIENT.search(response)
        if m:
            return ClickCommand(x=int(m.group(1)), y=int(m.group(2)))

        m = _PRESS_RE_LENIENT.search(response)
        if m:
            return PressCommand(key=m.group(1).strip())

        m = _TYPE_RE_LENIENT.search(response)
        if m:
            text = m.group(1) if m.group(1) is not None else m.group(2)
            return TypeCommand(text=text or "")

    except (IndexError, ValueError, AttributeError) as exc:
        # Defensive: never let a malformed response crash the agent.
        log.warning("parse_command: swallowed %s: %s", type(exc).__name__, exc)
        return None

    return None
