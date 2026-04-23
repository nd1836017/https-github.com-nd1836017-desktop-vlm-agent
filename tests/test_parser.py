"""Tests for the regex command parser."""
from __future__ import annotations

from agent.parser import (
    ClickCommand,
    PressCommand,
    TypeCommand,
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
