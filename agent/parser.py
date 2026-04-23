"""Regex-based command parser for VLM responses.

Supported commands:
    CLICK [X,Y]      — X and Y are integers on the 0-1000 normalized grid.
    PRESS [KEY]      — a single key name or '+'-separated hotkey (e.g. ctrl+c).
    TYPE [TEXT]      — literal text to type.

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

CommandType = Literal["CLICK", "PRESS", "TYPE"]


@dataclass(frozen=True)
class ClickCommand:
    kind: CommandType = "CLICK"
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


Command = ClickCommand | PressCommand | TypeCommand


# Primary patterns (strict: bracketed form).
_CLICK_RE = re.compile(
    r"CLICK\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)
_PRESS_RE = re.compile(r"PRESS\s*\[\s*([^\]\n]+?)\s*\]", re.IGNORECASE)
_TYPE_RE = re.compile(r"TYPE\s*\[\s*(.*?)\s*\]", re.IGNORECASE | re.DOTALL)

# Fallback patterns (lenient: missing brackets).
_CLICK_RE_LENIENT = re.compile(
    r"CLICK\s*\(?\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*\)?",
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
        # Strict forms first.
        m = _CLICK_RE.search(response)
        if m:
            return ClickCommand(x=int(m.group(1)), y=int(m.group(2)))

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
