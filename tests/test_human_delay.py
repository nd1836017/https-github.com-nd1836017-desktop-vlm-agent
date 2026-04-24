"""Unit tests for human-like click delay + typing jitter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.executor import (
    _human_pre_click_delay,
    _human_type_interval,
    execute,
    execute_click_pixels,
)
from agent.parser import ClickCommand, TypeCommand
from agent.screen import ScreenGeometry


def test_pre_click_delay_in_range():
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    for _ in range(100):
        delay = _human_pre_click_delay(0.5, 1.5, sleep=fake_sleep)
        assert 0.5 <= delay <= 1.5

    assert len(sleeps) == 100
    assert all(0.5 <= s <= 1.5 for s in sleeps)


def test_pre_click_delay_zero_range_no_sleep():
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    delay = _human_pre_click_delay(0.0, 0.0, sleep=fake_sleep)
    assert delay == 0.0
    assert sleeps == []


def test_pre_click_delay_handles_swapped_range():
    """If min > max, we clamp so we don't crash."""
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    delay = _human_pre_click_delay(2.0, 0.5, sleep=fake_sleep)
    # After clamping, hi == lo == 2.0.
    assert delay == 2.0
    assert sleeps == [2.0]


def test_type_interval_in_range():
    for _ in range(100):
        iv = _human_type_interval(0.02, 0.08)
        assert 0.02 <= iv <= 0.08


def test_type_interval_zero_disabled():
    assert _human_type_interval(0.0, 0.0) == 0.0


def test_execute_click_pixels_sleeps_pre_click():
    sleeps: list[float] = []

    fake_pyautogui = MagicMock()

    with (
        patch("agent.executor._pyautogui", return_value=fake_pyautogui),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
        patch("agent.executor.random.uniform", return_value=1.0),
    ):
        execute_click_pixels(
            100,
            200,
            animation_buffer_seconds=1.5,
            click_min_delay_seconds=0.5,
            click_max_delay_seconds=2.0,
        )

    fake_pyautogui.click.assert_called_once_with(100, 200)
    # First sleep is the human pre-click delay (1.0s from mocked uniform),
    # second is the animation buffer (1.5s).
    assert sleeps == [1.0, 1.5]


def test_execute_click_command_sleeps_pre_click():
    sleeps: list[float] = []
    fake_pyautogui = MagicMock()
    geometry = ScreenGeometry(width=1000, height=1000)

    with (
        patch("agent.executor._pyautogui", return_value=fake_pyautogui),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
        patch("agent.executor.random.uniform", return_value=0.7),
    ):
        execute(
            ClickCommand(x=500, y=500),
            geometry,
            animation_buffer_seconds=1.5,
            click_min_delay_seconds=0.3,
            click_max_delay_seconds=1.2,
        )

    fake_pyautogui.click.assert_called_once()
    assert sleeps[0] == 0.7  # human pre-click delay
    assert sleeps[-1] == 1.5  # animation buffer after the click


def test_execute_type_command_jitters_per_character():
    sleeps: list[float] = []
    fake_pyautogui = MagicMock()
    geometry = ScreenGeometry(width=1000, height=1000)

    # random.uniform returns 0.05 every time (between min=0.03 and max=0.12).
    with (
        patch("agent.executor._pyautogui", return_value=fake_pyautogui),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
        patch("agent.executor.random.uniform", return_value=0.05),
    ):
        execute(
            TypeCommand(text="abcd"),
            geometry,
            type_min_interval_seconds=0.03,
            type_max_interval_seconds=0.12,
        )

    # typewrite called once per char with interval=0
    assert fake_pyautogui.typewrite.call_count == 4
    for call in fake_pyautogui.typewrite.call_args_list:
        assert call.kwargs.get("interval") == 0

    # 4 jitter sleeps (0.05 each) + 1 trailing 0.25 settle sleep.
    assert sleeps == [0.05, 0.05, 0.05, 0.05, 0.25]


def test_execute_type_command_no_jitter_when_disabled():
    sleeps: list[float] = []
    fake_pyautogui = MagicMock()
    geometry = ScreenGeometry(width=1000, height=1000)

    with (
        patch("agent.executor._pyautogui", return_value=fake_pyautogui),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
    ):
        execute(
            TypeCommand(text="hi"),
            geometry,
            type_min_interval_seconds=0.0,
            type_max_interval_seconds=0.0,
        )

    assert fake_pyautogui.typewrite.call_count == 2
    # Only the trailing settle sleep — no per-char jitter.
    assert sleeps == [0.25]
