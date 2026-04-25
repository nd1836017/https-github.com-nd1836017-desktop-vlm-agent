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
    PAUSE [REASON]              — halt and wait for a human to resolve REASON
                                  (e.g. a 2FA prompt, captcha) before resuming.
    DOWNLOAD [URL]              — fetch URL via HTTP and save to the run
                                  workspace; filename derived from the URL.
    DOWNLOAD [URL, FILENAME]    — same, with explicit filename. Inside a
                                  FOR_EACH_ROW block FILENAME gets a
                                  ``(rowN)`` suffix to avoid collisions.
    ATTACH_FILE [FILENAME]      — type the absolute path of FILENAME into
                                  the OS file-picker dialog (Ctrl+L on
                                  Windows file dialogs) and submit.
    CAPTURE_FOR_AI              — capture the current screenshot and feed
                                  it as an extra image to the next plan
                                  call (no disk write).
    CAPTURE_FOR_AI [FILENAME]   — same, but also reads bytes from
                                  FILENAME (useful for non-screen content).
    REMEMBER [name]             — read the value labeled ``name`` off the
                                  current screen and store it as a run
                                  variable, accessible later via the
                                  ``{{var.name}}`` placeholder. Uses the
                                  VLM to extract the value (no typing).
    REMEMBER [name = literal]   — store ``literal`` directly as the value
                                  of variable ``name``. No VLM call.
    RECALL [name]               — alias for ``TYPE [{{var.name}}]``: types
                                  the stored value of ``name`` into the
                                  currently focused field. Convenience
                                  primitive when you don't want to write
                                  the placeholder out.

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
    "PAUSE",
    "DOWNLOAD",
    "ATTACH_FILE",
    "CAPTURE_FOR_AI",
    "REMEMBER",
    "RECALL",
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


@dataclass(frozen=True)
class PauseCommand:
    kind: CommandType = "PAUSE"
    reason: str = ""


@dataclass(frozen=True)
class DownloadCommand:
    """Download URL via HTTP into the run workspace.

    ``filename`` is optional — when missing, derived from the URL path.
    """

    kind: CommandType = "DOWNLOAD"
    url: str = ""
    filename: str = ""


@dataclass(frozen=True)
class AttachFileCommand:
    """Type a file path into a native OS file-picker dialog and submit.

    ``filename`` is matched against files in the workspace; if not found,
    treated as an absolute / CWD-relative path.
    """

    kind: CommandType = "ATTACH_FILE"
    filename: str = ""


@dataclass(frozen=True)
class CaptureForAiCommand:
    """Feed an image (current screen or a file) to the next plan call.

    ``filename`` is optional — when missing, the executor uses the next
    fresh screenshot.
    """

    kind: CommandType = "CAPTURE_FOR_AI"
    filename: str = ""


@dataclass(frozen=True)
class RememberCommand:
    """Store a value into the run's variable store.

    Two forms:

    - ``REMEMBER [name = literal]`` — ``literal`` is stored as-is, with no
      VLM round-trip. ``literal`` is set on ``literal_value`` and
      ``from_screen`` is False.
    - ``REMEMBER [name]`` — the agent asks the VLM to extract the value
      labeled ``name`` from the current screen. ``literal_value`` is empty
      and ``from_screen`` is True.

    The stored value is accessible later via the ``{{var.<name>}}``
    placeholder, which is substituted at runtime before the next step is
    sent to the planner.
    """

    kind: CommandType = "REMEMBER"
    name: str = ""
    literal_value: str = ""
    from_screen: bool = False


@dataclass(frozen=True)
class RecallCommand:
    """Type the stored value of variable ``name`` into the focused field.

    Equivalent to ``TYPE [{{var.<name>}}]`` but more explicit.
    """

    kind: CommandType = "RECALL"
    name: str = ""


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
    | PauseCommand
    | DownloadCommand
    | AttachFileCommand
    | CaptureForAiCommand
    | RememberCommand
    | RecallCommand
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
_PAUSE_RE = re.compile(
    r"PAUSE\s*\[\s*(.*?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)
# DOWNLOAD accepts [URL] (filename auto-derived) or [URL, FILENAME].
# The URL/FILENAME delimiter is comma + AT LEAST ONE WHITESPACE so that URLs
# containing bare commas (e.g. ``?ids=1,2,3`` query params) aren't silently
# truncated. ``DOWNLOAD [https://x/y?a=1,2,3]`` keeps the whole URL intact;
# ``DOWNLOAD [https://x/y, out.pdf]`` splits at the comma+space. Filenames
# may not contain ``,`` or ``]`` themselves.
_DOWNLOAD_RE = re.compile(
    r"DOWNLOAD\s*\[\s*(.+?)(?:,\s+([^\],]+?))?\s*\]",
    re.IGNORECASE,
)
_ATTACH_FILE_RE = re.compile(
    r"ATTACH[_\s]?FILE\s*\[\s*(.+?)\s*\]",
    re.IGNORECASE,
)
# CAPTURE_FOR_AI ALWAYS requires brackets so the parser doesn't accidentally
# match prose like "...to capture for AI analysis, click here". The contents
# may be empty (``CAPTURE_FOR_AI []`` = grab the current screen).
_CAPTURE_FOR_AI_RE = re.compile(
    r"CAPTURE[_\s]?FOR[_\s]?AI\s*\[\s*(.*?)\s*\]",
    re.IGNORECASE,
)
# REMEMBER accepts two forms:
#   REMEMBER [name]              -> from_screen=True, no literal
#   REMEMBER [name = some text]  -> literal_value="some text"
# ``name`` is alphanumeric + underscore (variable identifier shape).
_REMEMBER_RE = re.compile(
    r"REMEMBER\s*\[\s*([A-Za-z_][\w]*)\s*(?:=\s*(.*?))?\s*\]",
    re.IGNORECASE | re.DOTALL,
)
_RECALL_RE = re.compile(
    r"RECALL\s*\[\s*([A-Za-z_][\w]*)\s*\]",
    re.IGNORECASE,
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
        m = _PAUSE_RE.search(response)
        if m:
            reason = m.group(1).strip()
            if reason:
                return PauseCommand(reason=reason)

        # File primitives. Tried before CLICK_TEXT because their bracket
        # args can contain spaces / dots that would otherwise read as a
        # CLICK_TEXT label fallback.
        m = _DOWNLOAD_RE.search(response)
        if m:
            url = m.group(1).strip()
            filename = (m.group(2) or "").strip()
            if url:
                return DownloadCommand(url=url, filename=filename)

        m = _ATTACH_FILE_RE.search(response)
        if m:
            fname = m.group(1).strip()
            if fname:
                return AttachFileCommand(filename=fname)

        m = _CAPTURE_FOR_AI_RE.search(response)
        if m:
            fname = (m.group(1) or "").strip()
            return CaptureForAiCommand(filename=fname)

        # REMEMBER / RECALL must be tried before CLICK_TEXT — otherwise the
        # CLICK_TEXT regex (which is very permissive about its label arg)
        # would happily eat ``REMEMBER [order_id]`` as a label.
        m = _REMEMBER_RE.search(response)
        if m:
            name = m.group(1).strip()
            literal = m.group(2)
            if literal is None:
                return RememberCommand(name=name, from_screen=True)
            return RememberCommand(
                name=name,
                literal_value=literal.strip(),
                from_screen=False,
            )

        m = _RECALL_RE.search(response)
        if m:
            name = m.group(1).strip()
            if name:
                return RecallCommand(name=name)

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
