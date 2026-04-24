"""End-to-end tests for the CLICK_TEXT action primitive.

Verifies that when the VLM emits `CLICK_TEXT [label]`, the agent runs OCR
on the current screenshot, finds the label, and clicks its center.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest
from PIL import Image

from agent.agent import _execute_click_text
from agent.parser import ClickTextCommand
from agent.screen import ScreenGeometry


@pytest.fixture
def fake_screenshot():
    return Image.new("RGB", (800, 600), "white")


@pytest.fixture
def fake_pyautogui_patched():
    fake = mock.MagicMock()
    fake.FAILSAFE = False
    with mock.patch.dict(sys.modules, {"pyautogui": fake}):
        yield fake


def test_click_text_clicks_matched_center(fake_screenshot, fake_pyautogui_patched):
    match = mock.MagicMock()
    match.center.return_value = (120, 60)
    match.text = "Sign in"
    match.score = 1.0
    match.confidence = 90.0

    with mock.patch("agent.agent.find_text_center", return_value=match):
        ok, action_text = _execute_click_text(
            cmd=ClickTextCommand(label="Sign in"),
            screenshot=fake_screenshot,
            geometry=ScreenGeometry(width=800, height=600),
            animation_buffer=0.0,
        )

    assert ok is True
    fake_pyautogui_patched.click.assert_called_once_with(120, 60)
    assert "Sign in" in action_text


def test_click_text_returns_false_when_no_match(
    fake_screenshot, fake_pyautogui_patched
):
    with mock.patch("agent.agent.find_text_center", return_value=None):
        ok, action_text = _execute_click_text(
            cmd=ClickTextCommand(label="Sign in"),
            screenshot=fake_screenshot,
            geometry=ScreenGeometry(width=800, height=600),
            animation_buffer=0.0,
        )

    assert ok is False
    assert "no OCR match" in action_text
    fake_pyautogui_patched.click.assert_not_called()


def test_click_text_rescales_pixels_when_geometry_differs(
    fake_screenshot, fake_pyautogui_patched
):
    """When the screenshot resolution differs from the screen, coords are scaled."""
    match = mock.MagicMock()
    match.center.return_value = (400, 300)
    match.text = "OK"
    match.score = 1.0
    match.confidence = 90.0

    # Screenshot 800x600, screen 1600x1200 — coords double.
    with mock.patch("agent.agent.find_text_center", return_value=match):
        ok, _ = _execute_click_text(
            cmd=ClickTextCommand(label="OK"),
            screenshot=fake_screenshot,
            geometry=ScreenGeometry(width=1600, height=1200),
            animation_buffer=0.0,
        )

    assert ok is True
    fake_pyautogui_patched.click.assert_called_once_with(800, 600)
