"""Parser tests for the PAUSE [REASON] primitive."""
from __future__ import annotations

from agent.parser import (
    ClickCommand,
    PauseCommand,
    parse_command,
)


def test_parse_pause_simple():
    cmd = parse_command("PAUSE [Verify it's you on your phone]")
    assert isinstance(cmd, PauseCommand)
    assert cmd.reason == "Verify it's you on your phone"


def test_parse_pause_multiline_reason():
    cmd = parse_command("PAUSE [\n  Approve sign-in\n  and enter 75\n]")
    assert isinstance(cmd, PauseCommand)
    assert "Approve sign-in" in cmd.reason
    assert "enter 75" in cmd.reason


def test_parse_pause_case_insensitive():
    cmd = parse_command("pause [captcha detected]")
    assert isinstance(cmd, PauseCommand)
    assert cmd.reason == "captcha detected"


def test_pause_preferred_over_click():
    # Even if the response mentions CLICK elsewhere, PAUSE wins when present.
    cmd = parse_command(
        "I see a verify-it's-you prompt. PAUSE [approve on phone] — "
        "later the user can CLICK [100,200]."
    )
    assert isinstance(cmd, PauseCommand)


def test_empty_pause_falls_back():
    # Empty reason shouldn't match — caller will fall through to other patterns.
    cmd = parse_command("PAUSE []  then CLICK [100,200]")
    assert isinstance(cmd, ClickCommand)


def test_pause_with_prose_wrapping():
    cmd = parse_command(
        "The screen shows a CAPTCHA. PAUSE [Solve the CAPTCHA] — I will "
        "wait for you."
    )
    assert isinstance(cmd, PauseCommand)
    assert cmd.reason == "Solve the CAPTCHA"
