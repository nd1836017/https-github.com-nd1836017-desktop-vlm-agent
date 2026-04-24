"""Integration tests: PAUSE command inside the run_step loop.

Verifies that:
- When the planner emits PAUSE, the pause_handler is invoked.
- After the user signals continue, the agent re-plans without consuming the
  replan budget.
- If the user aborts at a PAUSE, the step returns a clean FAIL verdict.
- A PAUSE storm (>max_pauses_per_step) is bounded so we can't loop forever.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest
from PIL import Image

from agent.agent import ReplanCounter, run_step
from agent.history import History
from agent.screen import ScreenGeometry
from agent.vlm import VerificationResult


class _FakeClient:
    def __init__(self, plans, verdicts):
        self._plans = list(plans)
        self._verdicts = list(verdicts)
        self.plan_calls = 0
        self.verify_calls = 0

    def plan_action(self, step, screenshot, history_summary="", previous_failure=""):
        self.plan_calls += 1
        out = self._plans.pop(0)
        return (out, None) if not isinstance(out, tuple) else out

    def verify(self, goal, screenshot):
        self.verify_calls += 1
        passed, reason = self._verdicts.pop(0)
        return VerificationResult(passed=passed, reason=reason)


@pytest.fixture
def fake_geometry():
    return ScreenGeometry(width=1920, height=1080)


@pytest.fixture(autouse=True)
def patch_pyautogui():
    fake_img = Image.new("RGB", (8, 8))
    fake_pyautogui = mock.MagicMock()
    fake_pyautogui.size.return_value = mock.MagicMock(width=1920, height=1080)
    fake_pyautogui.screenshot.return_value = fake_img
    fake_pyautogui.FAILSAFE = False
    with mock.patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        yield fake_pyautogui


def test_pause_then_continue_succeeds(fake_geometry):
    client = _FakeClient(
        # 1. PAUSE -> user continues -> 2. real action that PASSes.
        plans=["PAUSE [Approve on phone]", "PRESS [enter]"],
        verdicts=[(True, "signed in")],
    )
    history = History(window=5)
    counter = ReplanCounter(total_max=10)

    pauses = []

    def handler(reason):
        pauses.append(reason)
        return True  # user says continue

    verdict = run_step(
        step="sign in",
        vlm=client,
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=2,
        history=history,
        enable_two_stage_click=False,
        replan_counter=counter,
        pause_handler=handler,
    )
    assert verdict.passed
    assert pauses == ["Approve on phone"]
    # The PAUSE loop iteration should NOT have counted against the replan budget.
    assert counter.total_used == 0


def test_pause_user_aborts(fake_geometry):
    client = _FakeClient(
        plans=["PAUSE [Solve captcha]"],
        verdicts=[],
    )
    verdict = run_step(
        step="verify step",
        vlm=client,
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=2,
        history=History(window=5),
        enable_two_stage_click=False,
        pause_handler=lambda _reason: False,  # user aborts
    )
    assert not verdict.passed
    assert "User aborted at PAUSE" in verdict.reason
    assert "Solve captcha" in verdict.reason


def test_pause_on_replan_does_not_double_count_global_replans(fake_geometry):
    # Attempt 1 verifier FAILs -> agent replans.
    # Attempt 2 planner emits PAUSE -> handler says continue -> attempt_idx
    # is rolled back so the agent can re-plan against the post-human state
    # without this PAUSE iteration permanently consuming a replan slot.
    # Attempt 2 (retried) emits a real action that PASSes.
    # End state: exactly ONE logical replan occurred (the FAIL -> retry).
    client = _FakeClient(
        plans=["PRESS [a]", "PAUSE [approve on phone]", "PRESS [b]"],
        verdicts=[(False, "first action missed"), (True, "ok")],
    )
    counter = ReplanCounter(total_max=10)
    verdict = run_step(
        step="do the thing",
        vlm=client,
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=2,
        history=History(window=5),
        enable_two_stage_click=False,
        replan_counter=counter,
        pause_handler=lambda _reason: True,
    )
    assert verdict.passed
    # Before the fix, PAUSE-during-replan double-consumed the counter so this
    # assertion would see total_used == 2.
    assert counter.total_used == 1, (
        f"PAUSE on a replan attempt must not double-consume the global replan "
        f"budget; expected total_used=1, got {counter.total_used}"
    )


def test_pause_storm_is_bounded(fake_geometry):
    # 50 consecutive PAUSEs — the step should halt cleanly after 10.
    client = _FakeClient(
        plans=["PAUSE [loop]"] * 50,
        verdicts=[],
    )
    verdict = run_step(
        step="stuck step",
        vlm=client,
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=2,
        history=History(window=5),
        enable_two_stage_click=False,
        pause_handler=lambda _reason: True,  # always "continue"
    )
    assert not verdict.passed
    assert "Exceeded max PAUSE rounds" in verdict.reason
    # 10 + 1 probe before we catch it.
    assert client.plan_calls <= 12
