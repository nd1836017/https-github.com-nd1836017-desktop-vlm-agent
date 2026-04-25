"""Tests for the regex command parser."""
from __future__ import annotations

from agent.parser import (
    ClickCommand,
    ClickTextCommand,
    DoubleClickCommand,
    DragCommand,
    MoveToCommand,
    PressCommand,
    RightClickCommand,
    ScrollCommand,
    TypeCommand,
    WaitCommand,
    parse_command,
)


def test_click_strict():
    cmd = parse_command("CLICK [500,250]")
    assert isinstance(cmd, ClickCommand)
    assert cmd.x == 500
    assert cmd.y == 250


def test_click_with_spaces():
    cmd = parse_command("CLICK [ 123 , 456 ]")
    assert isinstance(cmd, ClickCommand)
    assert (cmd.x, cmd.y) == (123, 456)


def test_click_embedded_in_prose():
    cmd = parse_command(
        "Sure! I'll click the button at CLICK [700,300] to open it."
    )
    assert isinstance(cmd, ClickCommand)
    assert (cmd.x, cmd.y) == (700, 300)


def test_click_lowercase():
    cmd = parse_command("click [10,20]")
    assert isinstance(cmd, ClickCommand)
    assert (cmd.x, cmd.y) == (10, 20)


def test_click_lenient_parens():
    cmd = parse_command("CLICK (100, 200)")
    assert isinstance(cmd, ClickCommand)
    assert (cmd.x, cmd.y) == (100, 200)


def test_click_lenient_no_brackets():
    cmd = parse_command("CLICK 100 200")
    assert isinstance(cmd, ClickCommand)
    assert (cmd.x, cmd.y) == (100, 200)


def test_press_strict():
    cmd = parse_command("PRESS [win]")
    assert isinstance(cmd, PressCommand)
    assert cmd.key == "win"


def test_press_hotkey():
    cmd = parse_command("PRESS [ctrl+c]")
    assert isinstance(cmd, PressCommand)
    assert cmd.key == "ctrl+c"


def test_press_lenient_no_brackets():
    cmd = parse_command("PRESS enter")
    assert isinstance(cmd, PressCommand)
    assert cmd.key == "enter"


def test_press_lenient_key_suffix():
    cmd = parse_command("PRESS the enter key")
    assert isinstance(cmd, PressCommand)
    assert cmd.key == "enter"


def test_type_strict():
    cmd = parse_command("TYPE [hello world]")
    assert isinstance(cmd, TypeCommand)
    assert cmd.text == "hello world"


def test_type_lenient_quotes():
    cmd = parse_command('TYPE "hello"')
    assert isinstance(cmd, TypeCommand)
    assert cmd.text == "hello"


def test_empty_response_returns_none():
    assert parse_command("") is None
    assert parse_command("   ") is None


def test_unrelated_prose_returns_none():
    assert parse_command("I don't know what to do here.") is None


def test_malformed_coords_returns_none():
    # No digits at all -> None, no crash.
    assert parse_command("CLICK [foo,bar]") is None


def test_first_command_wins_click_over_press():
    # If both are present, parser picks CLICK first (strict pattern is tried first).
    cmd = parse_command("CLICK [1,2] then PRESS [enter]")
    assert isinstance(cmd, ClickCommand)
    assert (cmd.x, cmd.y) == (1, 2)


def test_type_preserves_internal_brackets_naively():
    # Empty inner text is still a valid TYPE.
    cmd = parse_command("TYPE []")
    assert isinstance(cmd, TypeCommand)
    assert cmd.text == ""


# --- New primitives (SCROLL / DRAG / DOUBLE_CLICK / RIGHT_CLICK / WAIT / MOVE_TO). ---


def test_double_click_strict():
    cmd = parse_command("DOUBLE_CLICK [100,200]")
    assert isinstance(cmd, DoubleClickCommand)
    assert (cmd.x, cmd.y) == (100, 200)


def test_double_click_with_space():
    cmd = parse_command("DOUBLE CLICK [100,200]")
    assert isinstance(cmd, DoubleClickCommand)


def test_double_click_does_not_get_parsed_as_click():
    """CLICK regex must not eat DOUBLE_CLICK or RIGHT_CLICK."""
    cmd = parse_command("DOUBLE_CLICK [50,60]")
    assert isinstance(cmd, DoubleClickCommand)
    assert not isinstance(cmd, ClickCommand)


def test_right_click_strict():
    cmd = parse_command("RIGHT_CLICK [300,400]")
    assert isinstance(cmd, RightClickCommand)
    assert (cmd.x, cmd.y) == (300, 400)


def test_right_click_does_not_get_parsed_as_click():
    cmd = parse_command("RIGHT_CLICK [10,20]")
    assert isinstance(cmd, RightClickCommand)
    assert not isinstance(cmd, ClickCommand)


def test_move_to():
    cmd = parse_command("MOVE_TO [500,500]")
    assert isinstance(cmd, MoveToCommand)
    assert (cmd.x, cmd.y) == (500, 500)


def test_scroll_down():
    cmd = parse_command("SCROLL [down, 5]")
    assert isinstance(cmd, ScrollCommand)
    assert cmd.direction == "down"
    assert cmd.amount == 5


def test_scroll_up_case_insensitive():
    cmd = parse_command("scroll [UP, 3]")
    assert isinstance(cmd, ScrollCommand)
    assert cmd.direction == "up"
    assert cmd.amount == 3


def test_scroll_negative_amount_absolutized():
    """A negative AMOUNT is normalized to positive; direction carries the sign."""
    cmd = parse_command("SCROLL [down, -2]")
    assert isinstance(cmd, ScrollCommand)
    assert cmd.amount == 2


def test_drag_four_coords():
    cmd = parse_command("DRAG [10,20,30,40]")
    assert isinstance(cmd, DragCommand)
    assert (cmd.x1, cmd.y1, cmd.x2, cmd.y2) == (10, 20, 30, 40)


def test_drag_with_arrow():
    cmd = parse_command("DRAG [10,20 -> 30,40]")
    assert isinstance(cmd, DragCommand)
    assert (cmd.x1, cmd.y1, cmd.x2, cmd.y2) == (10, 20, 30, 40)


def test_drag_with_to():
    cmd = parse_command("DRAG [10,20 to 30,40]")
    assert isinstance(cmd, DragCommand)


def test_wait_integer():
    cmd = parse_command("WAIT [5]")
    assert isinstance(cmd, WaitCommand)
    assert cmd.seconds == 5.0


def test_wait_float():
    cmd = parse_command("WAIT [0.75]")
    assert isinstance(cmd, WaitCommand)
    assert cmd.seconds == 0.75


def test_wait_with_seconds_unit():
    cmd = parse_command("WAIT [2s]")
    assert isinstance(cmd, WaitCommand)
    assert cmd.seconds == 2.0


def test_click_text_simple():
    cmd = parse_command("CLICK_TEXT [Sign in]")
    assert isinstance(cmd, ClickTextCommand)
    assert cmd.label == "Sign in"


def test_click_text_with_punctuation():
    cmd = parse_command("CLICK_TEXT [Yes, continue]")
    assert isinstance(cmd, ClickTextCommand)
    assert cmd.label == "Yes, continue"


def test_click_text_is_not_eaten_by_click_regex():
    cmd = parse_command("CLICK_TEXT [Submit]")
    assert isinstance(cmd, ClickTextCommand)
    assert not isinstance(cmd, ClickCommand)


def test_click_text_empty_label_falls_through():
    # An empty label isn't useful — parser should reject and return None.
    assert parse_command("CLICK_TEXT []") is None


# ----------------------------------------------------- file-handling primitives


def test_parse_download_with_filename():
    from agent.parser import DownloadCommand

    cmd = parse_command("DOWNLOAD [https://example.com/foo.pdf, foo.pdf]")
    assert isinstance(cmd, DownloadCommand)
    assert cmd.url == "https://example.com/foo.pdf"
    assert cmd.filename == "foo.pdf"


def test_parse_download_url_only():
    from agent.parser import DownloadCommand

    cmd = parse_command("DOWNLOAD [https://example.com/foo.pdf]")
    assert isinstance(cmd, DownloadCommand)
    assert cmd.url == "https://example.com/foo.pdf"
    assert cmd.filename == ""


def test_parse_attach_file():
    from agent.parser import AttachFileCommand

    cmd = parse_command("ATTACH_FILE [invoice.pdf]")
    assert isinstance(cmd, AttachFileCommand)
    assert cmd.filename == "invoice.pdf"


def test_parse_attach_file_with_space_alias():
    from agent.parser import AttachFileCommand

    # Planner sometimes drops the underscore.
    cmd = parse_command("ATTACH FILE [resume.docx]")
    assert isinstance(cmd, AttachFileCommand)
    assert cmd.filename == "resume.docx"


def test_parse_capture_for_ai_no_filename():
    from agent.parser import CaptureForAiCommand

    cmd = parse_command("CAPTURE_FOR_AI")
    assert isinstance(cmd, CaptureForAiCommand)
    assert cmd.filename == ""


def test_parse_capture_for_ai_with_filename():
    from agent.parser import CaptureForAiCommand

    cmd = parse_command("CAPTURE_FOR_AI [snapshot.png]")
    assert isinstance(cmd, CaptureForAiCommand)
    assert cmd.filename == "snapshot.png"
