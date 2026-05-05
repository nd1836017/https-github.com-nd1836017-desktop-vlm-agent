"""Tests for the global MAX_TOTAL_REPLANS cap."""
from __future__ import annotations

import sys
from unittest import mock

import pytest
from PIL import Image

from agent.agent import ReplanCounter, run_step
from agent.history import History
from agent.screen import ScreenGeometry
from agent.vlm import VerificationResult


class FakeClient:
    """Minimal stand-in for GeminiClient — always returns the same CLICK."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.plan_calls = 0

    def plan_action(self, step, screenshot, history_summary="", previous_failure="", extra_images=None, routing_hint="", prior_run_hint=""):
        self.plan_calls += 1
        return "CLICK [500,500]", None

    def verify(self, goal, screenshot):
        out = self._verdicts.pop(0)
        if isinstance(out, VerificationResult):
            return out
        passed, reason = out
        return VerificationResult(passed=passed, reason=reason)


@pytest.fixture(autouse=True)
def patch_pyautogui():
    fake_img = Image.new("RGB", (8, 8), color=(0, 0, 0))
    fake_pyautogui = mock.MagicMock()
    fake_pyautogui.size.return_value = mock.MagicMock(width=1920, height=1080)
    fake_pyautogui.screenshot.return_value = fake_img
    fake_pyautogui.FAILSAFE = False
    with mock.patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        yield fake_pyautogui


def _args(vlm, counter):
    return dict(
        step="do the thing",
        vlm=vlm,
        geometry=ScreenGeometry(width=1920, height=1080),
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=5,  # Per-step budget generous; global cap should trip first.
        history=History(window=5),
        enable_two_stage_click=False,
        replan_counter=counter,
    )


def test_global_budget_disabled_when_zero():
    """total_max=0 means unlimited (backwards-compat with pre-PR6 behavior)."""
    vlm = FakeClient(verdicts=[(False, "fail1"), (False, "fail2"), (True, "ok")])
    counter = ReplanCounter(total_max=0)
    result = run_step(**_args(vlm, counter))
    assert result.passed
    # 1 initial + 2 replans = 3 plan calls. Counter still tracks the replans
    # (useful for observability) but never trips a halt when total_max=0.
    assert vlm.plan_calls == 3
    assert counter.total_used == 2
    assert not counter.budget_exhausted


def test_global_budget_counts_replans_across_runs():
    vlm = FakeClient(verdicts=[(False, "fail1"), (True, "ok")])
    counter = ReplanCounter(total_max=5)
    result = run_step(**_args(vlm, counter))
    assert result.passed
    # One replan was consumed from the global budget.
    assert counter.total_used == 1


def test_global_budget_halts_when_exhausted():
    """Once total_used >= total_max, the next replan attempt halts immediately."""
    # Five FAIL verdicts consecutively. With total_max=2, only 2 replans are
    # permitted; the 3rd attempt should halt without a plan_action call.
    vlm = FakeClient(
        verdicts=[
            (False, "fail1"),
            (False, "fail2"),
            (False, "fail3"),
            (False, "fail4"),
            (False, "fail5"),
        ]
    )
    counter = ReplanCounter(total_max=2)
    result = run_step(**_args(vlm, counter))
    assert not result.passed
    assert "Global replan budget exhausted" in result.reason
    # 1 initial + 2 replans = 3 plan_action calls. Further replans blocked.
    assert vlm.plan_calls == 3
    assert counter.total_used == 2
    assert counter.budget_exhausted


def test_global_budget_shared_across_multiple_run_step_calls():
    counter = ReplanCounter(total_max=3)

    # First call consumes 2 replans.
    vlm1 = FakeClient(verdicts=[(False, "x"), (False, "y"), (True, "ok")])
    r1 = run_step(**_args(vlm1, counter))
    assert r1.passed
    assert counter.total_used == 2

    # Second call needs 2 replans but only 1 is left — should halt.
    vlm2 = FakeClient(
        verdicts=[(False, "x"), (False, "y"), (False, "z")]
    )
    r2 = run_step(**_args(vlm2, counter))
    assert not r2.passed
    assert "Global replan budget exhausted" in r2.reason
    assert counter.total_used == 3
