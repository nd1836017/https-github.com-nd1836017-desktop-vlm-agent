"""Unit tests for the new action primitives introduced in PR #5.

Covers SCROLL, DRAG, DOUBLE_CLICK, RIGHT_CLICK, WAIT, MOVE_TO — each
confirmed to invoke the right pyautogui call with the expected
arguments, and TYPE redaction of the log line.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from agent.executor import WAIT_MAX_SECONDS, execute
from agent.history import render_command
from agent.parser import (
    DoubleClickCommand,
    DragCommand,
    MoveToCommand,
    RightClickCommand,
    ScrollCommand,
    TypeCommand,
    WaitCommand,
)
from agent.screen import ScreenGeometry

_GEOM = ScreenGeometry(width=1000, height=1000)


def test_execute_double_click_invokes_pyautogui_doubleclick():
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep"),
        patch("agent.executor.random.uniform", return_value=0.5),
    ):
        execute(DoubleClickCommand(x=500, y=500), _GEOM)
    fake.doubleClick.assert_called_once()


def test_execute_right_click_invokes_pyautogui_rightclick():
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep"),
        patch("agent.executor.random.uniform", return_value=0.5),
    ):
        execute(RightClickCommand(x=100, y=100), _GEOM)
    fake.rightClick.assert_called_once()


def test_execute_move_to_calls_moveto_without_clicking():
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep"),
    ):
        execute(MoveToCommand(x=250, y=750), _GEOM)
    fake.moveTo.assert_called_once()
    fake.click.assert_not_called()
    fake.doubleClick.assert_not_called()
    fake.rightClick.assert_not_called()


def test_execute_scroll_down_passes_negative_clicks():
    """pyautogui.scroll: positive=up, negative=down."""
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep"),
    ):
        execute(ScrollCommand(direction="down", amount=3), _GEOM)
    fake.scroll.assert_called_once_with(-3)


def test_execute_scroll_up_passes_positive_clicks():
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep"),
    ):
        execute(ScrollCommand(direction="up", amount=5), _GEOM)
    fake.scroll.assert_called_once_with(5)


def test_execute_drag_moves_then_drags():
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep"),
        patch("agent.executor.random.uniform", return_value=0.5),
    ):
        execute(DragCommand(x1=100, y1=200, x2=500, y2=600), _GEOM)
    fake.moveTo.assert_called_once()
    fake.dragTo.assert_called_once()


def test_execute_wait_calls_time_sleep_with_requested_seconds():
    sleeps: list[float] = []
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
    ):
        execute(WaitCommand(seconds=2.5), _GEOM)
    assert 2.5 in sleeps


def test_execute_wait_clamps_to_max():
    sleeps: list[float] = []
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
    ):
        execute(WaitCommand(seconds=9999.0), _GEOM)
    assert WAIT_MAX_SECONDS in sleeps
    assert 9999.0 not in sleeps


def test_execute_type_redacted_logs_do_not_leak_text(caplog):
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep"),
        patch("agent.executor.random.uniform", return_value=0.05),
        caplog.at_level(logging.INFO, logger="agent.executor"),
    ):
        execute(
            TypeCommand(text="hunter2"),
            _GEOM,
            log_redact_type=True,
        )

    combined = "\n".join(caplog.messages)
    assert "hunter2" not in combined
    assert "REDACTED" in combined
    assert "7 chars" in combined


def test_execute_type_unredacted_logs_the_text_when_explicitly_disabled(caplog):
    fake = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake),
        patch("agent.executor.time.sleep"),
        patch("agent.executor.random.uniform", return_value=0.05),
        caplog.at_level(logging.INFO, logger="agent.executor"),
    ):
        execute(
            TypeCommand(text="hello"),
            _GEOM,
            log_redact_type=False,
        )

    combined = "\n".join(caplog.messages)
    assert "hello" in combined


def test_render_command_redacts_type():
    assert render_command(TypeCommand(text="secret"), redact_type=True) == (
        "TYPE [<REDACTED, 6 chars>]"
    )


def test_render_command_does_not_redact_non_type():
    # Redaction only ever affects TYPE commands.
    rendered = render_command(ScrollCommand(direction="up", amount=4), redact_type=True)
    assert rendered == "SCROLL [up,4]"
