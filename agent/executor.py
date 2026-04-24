"""Executes parsed commands via pyautogui with stability safeguards."""
from __future__ import annotations

import logging
import time

from .parser import ClickCommand, Command, PressCommand, TypeCommand
from .screen import ScreenGeometry

log = logging.getLogger(__name__)


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


def execute_click_pixels(
    px: int,
    py: int,
    animation_buffer_seconds: float = 1.5,
) -> None:
    """Click at absolute screen pixel coordinates and sleep the animation buffer.

    Used by the two-stage CLICK path, where the refined pixel is computed
    directly from a cropped region and we want to bypass the normalized
    0-1000 round-trip (which would lose sub-unit precision).
    """
    pyautogui = _pyautogui()
    log.info("CLICK pixels=(%d,%d)", px, py)
    pyautogui.click(px, py)
    time.sleep(animation_buffer_seconds)


def execute(
    cmd: Command,
    geometry: ScreenGeometry,
    animation_buffer_seconds: float = 1.5,
) -> None:
    """Execute a parsed command and sleep the animation buffer when required."""
    pyautogui = _pyautogui()

    if isinstance(cmd, ClickCommand):
        px, py = geometry.to_pixels(cmd.x, cmd.y)
        log.info(
            "CLICK normalized=(%d,%d) -> pixels=(%d,%d)", cmd.x, cmd.y, px, py
        )
        pyautogui.click(px, py)
        time.sleep(animation_buffer_seconds)
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
        log.info("TYPE %r", cmd.text)
        # Small interval between characters helps slower apps keep up.
        pyautogui.typewrite(cmd.text, interval=0.02)
        time.sleep(0.25)
        return

    raise TypeError(f"Unknown command type: {type(cmd).__name__}")
