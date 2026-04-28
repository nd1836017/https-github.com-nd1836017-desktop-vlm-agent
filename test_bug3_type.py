"""Live test for Bug #3: non-ASCII TYPE via clipboard+Ctrl+V.

Test-only workaround: stub the ``mouseinfo`` module before importing
pyautogui, because pyautogui's top-level import chain requires tkinter
which this pyenv build lacks. Production environments (user's Windows
box) bundle tkinter with the stock Python installer, so this monkey-patch
does NOT mask a real bug — it only lets our headless test exercise the
TYPE code path. The executor itself is not modified.
"""
from __future__ import annotations

import logging
import sys
import time

sys.modules.setdefault("mouseinfo", type(sys)("mouseinfo"))  # stub before pyautogui import

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

import pyautogui  # noqa: E402

pyautogui.FAILSAFE = False  # mouse corner must not abort the run

from agent.executor import _is_typewrite_safe, _paste_char  # noqa: E402

TEXT = "café résumé 你好 😀 Test!"


def main() -> int:
    print("=== classifier spot-check ===", flush=True)
    for c in ["a", " ", "!", "é", "你", "好", "😀", "T"]:
        print(f"  {c!r:>6} -> {_is_typewrite_safe(c)}", flush=True)

    # Ensure Chrome textarea is focused by clicking the center of the screen.
    print("\n=== typing ===", flush=True)
    for i, char in enumerate(TEXT):
        safe = _is_typewrite_safe(char)
        print(f"  [{i:2}] {char!r} safe={safe}", flush=True)
        if safe:
            pyautogui.typewrite(char, interval=0)
        else:
            _paste_char(pyautogui, char, redact=False)
        time.sleep(0.04)
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
