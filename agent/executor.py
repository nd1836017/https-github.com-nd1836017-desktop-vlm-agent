"""Executes parsed commands via pyautogui with stability safeguards."""
from __future__ import annotations

import logging
import random
import time

from .parser import (
    ClickCommand,
    Command,
    DoubleClickCommand,
    DragCommand,
    MoveToCommand,
    PressCommand,
    RightClickCommand,
    ScrollCommand,
    TypeCommand,
    WaitCommand,
)
from .screen import ScreenGeometry

log = logging.getLogger(__name__)


# Safety cap on WAIT values — a hallucinated `WAIT [9999]` should not
# freeze the agent for hours.
WAIT_MAX_SECONDS = 60.0


def _pyautogui():
    """Lazy import so tests and non-desktop contexts don't need a display."""
    import pyautogui

    # Required by spec. Safe to set every call — the flag is module-global.
    pyautogui.FAILSAFE = True
    return pyautogui


# Keys that trigger OS-level UI animations and require a longer post-action buffer.
_ANIMATION_KEYS = {"win", "super", "cmd", "meta", "lwin", "rwin"}


def _is_animation_key(key: str) -> bool:
    k = key.lower().strip()
    if k in _ANIMATION_KEYS:
        return True
    # Hotkeys containing an animation key (e.g. "win+d") also trigger animation.
    parts = [p.strip().lower() for p in k.split("+")]
    return any(p in _ANIMATION_KEYS for p in parts)


def _human_pre_click_delay(
    min_seconds: float,
    max_seconds: float,
    *,
    sleep=None,
) -> float:
    """Sleep a random interval in [min, max] before a click.

    This breaks up perfectly-regular click cadence that bot-detection
    heuristics watch for. Returns the actual delay slept (mostly useful
    for tests). If the range is non-positive the call is a no-op.
    """
    lo = max(0.0, float(min_seconds))
    hi = max(lo, float(max_seconds))
    if hi <= 0.0:
        return 0.0
    delay = random.uniform(lo, hi)
    log.info("Pre-click human delay: %.2fs", delay)
    # Resolve ``time.sleep`` lazily so tests that patch ``agent.executor.time.sleep``
    # can observe the call.
    (sleep or time.sleep)(delay)
    return delay


def _human_type_interval(
    min_interval: float, max_interval: float
) -> float:
    """Return a random per-keystroke interval inside [min, max]."""
    lo = max(0.0, float(min_interval))
    hi = max(lo, float(max_interval))
    if hi <= 0.0:
        return 0.0
    return random.uniform(lo, hi)


def _is_typewrite_safe(char: str) -> bool:
    """True when ``pyautogui.typewrite`` can reliably emit ``char``.

    ``typewrite`` uses a US-keyboard key map and silently discards any
    character it can't map (anything with ord > 127, tabs, newlines, ...).
    We treat printable ASCII plus the couple of whitespace chars pyautogui
    actually handles as safe; anything else gets the clipboard-paste path.
    """
    if len(char) != 1:
        return False
    if char in ("\n", "\t"):
        return True
    return 32 <= ord(char) <= 126


def _paste_char(pyautogui_module, char: str, *, redact: bool) -> None:
    """Paste a single char via the clipboard + Ctrl+V.

    Falls back to a warning log if ``pyperclip`` isn't installed or the
    clipboard is unavailable (headless CI with no X11 clipboard, etc.).
    Never raises — typing is best-effort by design.
    """
    try:
        import pyperclip
    except ImportError:
        log.warning(
            "TYPE: non-ASCII char %s dropped (pyperclip not installed)",
            "<REDACTED>" if redact else repr(char),
        )
        return
    try:
        pyperclip.copy(char)
    except pyperclip.PyperclipException as exc:  # type: ignore[attr-defined]
        log.warning(
            "TYPE: non-ASCII char %s dropped (clipboard unavailable: %s)",
            "<REDACTED>" if redact else repr(char),
            exc,
        )
        return
    pyautogui_module.hotkey("ctrl", "v")


def execute_click_pixels(
    px: int,
    py: int,
    animation_buffer_seconds: float = 1.5,
    *,
    click_min_delay_seconds: float = 0.0,
    click_max_delay_seconds: float = 0.0,
) -> None:
    """Click at absolute screen pixel coordinates and sleep the animation buffer.

    Used by the two-stage CLICK path, where the refined pixel is computed
    directly from a cropped region and we want to bypass the normalized
    0-1000 round-trip (which would lose sub-unit precision). A human-like
    pre-click delay is applied first when configured.
    """
    pyautogui = _pyautogui()
    _human_pre_click_delay(click_min_delay_seconds, click_max_delay_seconds)
    log.info("CLICK pixels=(%d,%d)", px, py)
    pyautogui.click(px, py)
    time.sleep(animation_buffer_seconds)


def execute(
    cmd: Command,
    geometry: ScreenGeometry,
    animation_buffer_seconds: float = 1.5,
    *,
    click_min_delay_seconds: float = 0.0,
    click_max_delay_seconds: float = 0.0,
    type_min_interval_seconds: float = 0.02,
    type_max_interval_seconds: float = 0.02,
    log_redact_type: bool = True,
) -> None:
    """Execute a parsed command and sleep the animation buffer when required."""
    pyautogui = _pyautogui()

    if isinstance(cmd, ClickCommand):
        px, py = geometry.to_pixels(cmd.x, cmd.y)
        _human_pre_click_delay(click_min_delay_seconds, click_max_delay_seconds)
        log.info(
            "CLICK normalized=(%d,%d) -> pixels=(%d,%d)", cmd.x, cmd.y, px, py
        )
        pyautogui.click(px, py)
        time.sleep(animation_buffer_seconds)
        return

    if isinstance(cmd, DoubleClickCommand):
        px, py = geometry.to_pixels(cmd.x, cmd.y)
        _human_pre_click_delay(click_min_delay_seconds, click_max_delay_seconds)
        log.info(
            "DOUBLE_CLICK normalized=(%d,%d) -> pixels=(%d,%d)",
            cmd.x,
            cmd.y,
            px,
            py,
        )
        pyautogui.doubleClick(px, py)
        time.sleep(animation_buffer_seconds)
        return

    if isinstance(cmd, RightClickCommand):
        px, py = geometry.to_pixels(cmd.x, cmd.y)
        _human_pre_click_delay(click_min_delay_seconds, click_max_delay_seconds)
        log.info(
            "RIGHT_CLICK normalized=(%d,%d) -> pixels=(%d,%d)",
            cmd.x,
            cmd.y,
            px,
            py,
        )
        pyautogui.rightClick(px, py)
        time.sleep(animation_buffer_seconds)
        return

    if isinstance(cmd, MoveToCommand):
        px, py = geometry.to_pixels(cmd.x, cmd.y)
        log.info(
            "MOVE_TO normalized=(%d,%d) -> pixels=(%d,%d)", cmd.x, cmd.y, px, py
        )
        pyautogui.moveTo(px, py, duration=0.2)
        # Short settle so hover-triggered tooltips/menus can render before the
        # next screenshot is taken.
        time.sleep(0.5)
        return

    if isinstance(cmd, ScrollCommand):
        # pyautogui.scroll: positive = up, negative = down.
        amount = abs(int(cmd.amount))
        direction = cmd.direction.lower()
        clicks = amount if direction == "up" else -amount
        log.info("SCROLL direction=%s amount=%d (clicks=%d)", direction, amount, clicks)
        pyautogui.scroll(clicks)
        time.sleep(0.5)
        return

    if isinstance(cmd, DragCommand):
        px1, py1 = geometry.to_pixels(cmd.x1, cmd.y1)
        px2, py2 = geometry.to_pixels(cmd.x2, cmd.y2)
        _human_pre_click_delay(click_min_delay_seconds, click_max_delay_seconds)
        log.info(
            "DRAG normalized=(%d,%d)->(%d,%d) pixels=(%d,%d)->(%d,%d)",
            cmd.x1,
            cmd.y1,
            cmd.x2,
            cmd.y2,
            px1,
            py1,
            px2,
            py2,
        )
        pyautogui.moveTo(px1, py1, duration=0.2)
        pyautogui.dragTo(px2, py2, duration=0.5, button="left")
        time.sleep(animation_buffer_seconds)
        return

    if isinstance(cmd, WaitCommand):
        seconds = max(0.0, min(float(cmd.seconds), WAIT_MAX_SECONDS))
        log.info("WAIT %.2fs", seconds)
        time.sleep(seconds)
        return

    if isinstance(cmd, PressCommand):
        key = cmd.key.strip()
        log.info("PRESS %r", key)
        if "+" in key:
            parts = [p.strip() for p in key.split("+") if p.strip()]
            pyautogui.hotkey(*parts)
        else:
            pyautogui.press(key)
        if _is_animation_key(key):
            time.sleep(animation_buffer_seconds)
        else:
            # Still a small buffer so the UI settles before the next screenshot.
            time.sleep(0.25)
        return

    if isinstance(cmd, TypeCommand):
        if log_redact_type:
            log.info("TYPE <REDACTED, %d chars>", len(cmd.text))
        else:
            log.info("TYPE %r", cmd.text)
        # Per-character jittered interval to avoid perfectly-uniform typing
        # cadence that is trivial for bot detectors to flag. pyautogui's
        # typewrite only accepts a fixed interval, so we loop ourselves.
        # pyautogui.typewrite silently drops any char outside its
        # US-keyboard map (anything with ord > 127, plus some punctuation).
        # For those we copy the char to the clipboard and paste with
        # Ctrl+V so non-ASCII text (emoji, accented chars, CJK, etc.)
        # actually lands in the target field instead of vanishing.
        for char in cmd.text:
            if _is_typewrite_safe(char):
                pyautogui.typewrite(char, interval=0)
            else:
                _paste_char(pyautogui, char, redact=log_redact_type)
            interval = _human_type_interval(
                type_min_interval_seconds, type_max_interval_seconds
            )
            if interval > 0:
                time.sleep(interval)
        time.sleep(0.25)
        return

    raise TypeError(f"Unknown command type: {type(cmd).__name__}")
