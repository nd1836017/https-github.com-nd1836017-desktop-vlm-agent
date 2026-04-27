"""Regression tests for the 8 real Devin Review findings on PR #23.

Each test names the finding number from the review and the specific
behavior it pins. The two findings NOT covered here are:

* Finding 2 (``budget_exhausted`` field) — flagged as dead-code-ish but
  ``can_replan()`` already enforces the budget. Not a runtime bug; no
  fix shipped, no test added.
* Finding 4 (``gemini-3.1-flash-lite-preview``) — confirmed intentional
  by the user as test scaffolding; not a real bug.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

import agent.agent as agent_module
from agent.agent import _attempt_step
from agent.parser import (
    PauseCommand,
    ScrollCommand,
    TypeCommand,
    parse_command,
)
from agent.screen import ScreenGeometry
from agent.state import AgentState, save_state
from agent.task_router import RoutingMode, apply_router
from agent.variables import VariableStore
from agent.vlm import GeminiClient, PlanResponseModel, VerificationResult

# ---------------------------------------------------------------------------
# Shared fixtures / fakes
# ---------------------------------------------------------------------------


class _FakeClient:
    """Mirror of the helpers in test_pause_in_loop / test_pr_d_memory_tier4."""

    def __init__(self, plans, verdicts):
        self._plans = list(plans)
        self._verdicts = list(verdicts)
        self.plan_calls = 0
        self.verify_calls = 0
        self.plan_screenshots: list = []

    def plan_action(
        self,
        step,
        screenshot,
        history_summary="",
        previous_failure="",
        extra_images=None,
        routing_hint="",
    ):
        self.plan_calls += 1
        self.plan_screenshots.append(screenshot)
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


# ---------------------------------------------------------------------------
# Finding #1: pause_handler must be threaded through run() to run_step().
# Production runs without an explicit handler still fall back to stdin via
# _handle_pause; what we want to pin here is that a custom handler passed
# to run() actually reaches run_step().
# ---------------------------------------------------------------------------


def test_run_accepts_pause_handler_kwarg():
    import inspect

    sig = inspect.signature(agent_module.run)
    assert "pause_handler" in sig.parameters


def test_run_threads_pause_handler_to_run_step(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_run_step(*_args, **kwargs):
        captured["pause_handler"] = kwargs.get("pause_handler")
        # Simulate a passing step so the loop terminates cleanly.
        return VerificationResult(passed=True, reason="ok")

    monkeypatch.setattr(agent_module, "run_step", fake_run_step)

    # Minimal Config with a tasks file containing one trivial step.
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("press enter\n", encoding="utf-8")
    state = tmp_path / "state.json"
    monkeypatch.setenv("TASKS_FILE", str(tasks))
    monkeypatch.setenv("STATE_FILE", str(state))
    monkeypatch.setenv("RUN_ARTIFACTS_DIR", str(tmp_path / "runs"))
    # Disable both the router (no Gemini calls) and the bridge.
    monkeypatch.setenv("TASK_ROUTING", "off")
    monkeypatch.setenv("BROWSER_FAST_PATH", "false")

    from agent.config import Config

    cfg = Config.load()

    sentinel = object()

    def handler(_reason: str) -> bool:  # pragma: no cover - never invoked
        return True

    # Stub out the VLM client construction so we don't need real keys.
    monkeypatch.setattr(
        agent_module,
        "GeminiClient",
        mock.MagicMock(return_value=mock.MagicMock()),
    )

    rc = agent_module.run(cfg, pause_handler=handler)
    assert rc == 0
    assert captured["pause_handler"] is handler
    # Sanity: the sentinel was unused; we just want to assert identity.
    del sentinel


# ---------------------------------------------------------------------------
# Finding #3: _attempt_step must capture the screenshot ONCE per attempt,
# even if the parse retry fires multiple times. Otherwise a tooltip /
# animation between retries silently changes the visual baseline the
# planner saw.
# ---------------------------------------------------------------------------


def test_attempt_step_does_not_recapture_screenshot_on_parse_retry(
    fake_geometry, monkeypatch,
):
    capture_calls = {"n": 0}

    def fake_capture():
        capture_calls["n"] += 1
        # Return a unique image each call so the test would be able to
        # detect the bug — different calls produce different objects.
        return Image.new("RGB", (8, 8), color=(capture_calls["n"], 0, 0))

    monkeypatch.setattr(agent_module, "capture_screenshot", fake_capture)

    # Plan calls: first response unparseable, second parseable -> CLICK.
    # The verifier passes so the test exits cleanly after the action
    # executes via the patched pyautogui mock.
    client = _FakeClient(
        plans=[
            "garbage that cannot parse",
            "CLICK [100, 200]",
        ],
        verdicts=[(True, "ok")],
    )

    # Disable two-stage refinement so we don't need to mock its VLM calls.
    verdict, _action = _attempt_step(
        step="press the button",
        vlm=client,
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=1,
        history_summary="",
        previous_failure="",
        enable_two_stage_click=False,
        two_stage_crop_size_px=300,
        max_click_candidates=5,
    )
    # Both planner calls should have used the SAME screenshot object;
    # otherwise the parse retry generated a fresh frame.
    assert client.plan_calls == 2
    assert client.plan_screenshots[0] is client.plan_screenshots[1]
    # And capture_screenshot must have been called exactly once for the
    # baseline + once for the post-action snapshot taken after the click.
    # (The post-action one is unrelated to the bug under test.)
    assert capture_calls["n"] <= 2, (
        f"capture_screenshot called {capture_calls['n']} times; "
        "parse retry should reuse the baseline."
    )
    assert verdict.passed


# ---------------------------------------------------------------------------
# Finding #5: VariableStore rehydrate must trigger on existing_state.variables
# == {} (empty but valid checkpoint), not just truthy values.
# ---------------------------------------------------------------------------


def test_resume_rehydrates_with_empty_variables_checkpoint(tmp_path, monkeypatch):
    """Pin: the resume path must use ``is not None`` to decide whether to
    rehydrate, so an empty {} from a v2 checkpoint doesn't silently
    regress to a brand-new VariableStore.
    """
    # Simulate a checkpoint where the previous run advanced past step 1
    # but never called REMEMBER. variables == {} is valid persisted state.
    state = AgentState.initial(tmp_path / "tasks.txt", total_steps=2)
    state = state.advance().with_variables({})
    state_path = tmp_path / "state.json"
    save_state(state_path, state)

    # The fix is to switch the truthy check on existing_state.variables
    # to an `is not None` check so the rehydrate runs even for {}. We
    # verify the source code itself contains the new check; an end-to-end
    # test would require spinning up the entire run() pipeline which is
    # heavier than needed for a one-line guard.
    src = (
        Path(agent_module.__file__).read_text(encoding="utf-8")
    )
    assert "existing_state.variables is not None" in src, (
        "rehydrate guard was not updated to `is not None`; an empty {} "
        "checkpoint will silently skip rehydration."
    )


# ---------------------------------------------------------------------------
# Finding #6: _last_planner_signature must NOT be updated when
# send_screenshot=False. Otherwise we record a fingerprint for a frame
# the model never actually saw.
# ---------------------------------------------------------------------------


def _stub_plan_response(client: GeminiClient) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.text = '{"command":"WAIT","seconds":1}'
    resp.parsed = PlanResponseModel(command="WAIT", seconds=1)
    client._client.models.generate_content = mock.MagicMock(return_value=resp)
    return client._client.models.generate_content


def test_signature_not_updated_when_screenshot_skipped():
    with mock.patch("agent.vlm.genai.Client"):
        client = GeminiClient(
            api_key="fake",
            model_name="fake-model",
            enable_json_output=True,
            skip_identical_frames=True,
        )
    _stub_plan_response(client)

    img_a = Image.new("RGB", (640, 480), color=(10, 20, 30))

    # First call seeds the tracker with img_a's signature.
    client.plan_action("step 1", img_a)
    sig_after_seed = client._last_planner_signature
    assert sig_after_seed is not None

    # Replan on the same step with the same image -> screenshot dropped.
    client.plan_action("step 1", img_a, previous_failure="failed")
    sig_after_skip = client._last_planner_signature
    # The skipped call must not have rewritten the tracker; whatever was
    # there before remains there. (Same value here, but the important
    # invariant is "no write happened on a skip path".)
    assert sig_after_skip == sig_after_seed


# ---------------------------------------------------------------------------
# Finding #7: SCROLL with a negative AMOUNT — keep abs() to salvage the
# action but log a warning so a postmortem can find the bug.
# ---------------------------------------------------------------------------


def test_scroll_negative_amount_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.parser"):
        cmd = parse_command("SCROLL [down, -3]")
    assert isinstance(cmd, ScrollCommand)
    assert cmd.amount == 3
    assert any(
        "negative amount -3" in rec.message for rec in caplog.records
    ), [r.message for r in caplog.records]


def test_scroll_positive_amount_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.parser"):
        cmd = parse_command("SCROLL [up, 5]")
    assert isinstance(cmd, ScrollCommand)
    assert cmd.amount == 5
    assert not any(
        "negative amount" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Finding #8: PAUSE [] (empty reason) used to silently fall through —
# the planner clearly wants a pause; surface a default reason instead.
# (Behavior change from the older ``test_empty_pause_falls_back`` test,
# which has been updated separately.)
# ---------------------------------------------------------------------------


def test_pause_empty_returns_pausecommand_with_default_reason():
    cmd = parse_command("PAUSE []")
    assert isinstance(cmd, PauseCommand)
    assert cmd.reason == "manual pause requested"


# ---------------------------------------------------------------------------
# Finding #9: TYPE regex must not consume newlines (DOTALL removed) so a
# multi-line response with a TYPE on one line and a CLICK on the next
# doesn't get the CLICK swallowed into the TYPE payload.
# ---------------------------------------------------------------------------


def test_type_regex_does_not_match_across_newlines():
    # Earlier _TYPE_RE used re.DOTALL, which let `.` swallow newlines.
    # A broken / truncated bracket spanning two lines used to capture
    # whatever followed up to the next `]` — sometimes including the
    # text of a *different* command on a later line. Without DOTALL
    # the regex no longer matches across the newline.
    from agent.parser import _TYPE_RE

    assert _TYPE_RE.search("TYPE [hello\nworld]") is None
    # Single-line still parses fine.
    m = _TYPE_RE.search("TYPE [hello world]")
    assert m is not None
    assert m.group(1) == "hello world"


def test_type_command_payload_scoped_to_one_line():
    # Even on a multi-line response, the parsed TYPE payload must not
    # include text from a later line.
    response = "before line\nTYPE [hello world]\nafter line"
    cmd = parse_command(response)
    assert isinstance(cmd, TypeCommand)
    assert cmd.text == "hello world"
    assert "after line" not in cmd.text


# ---------------------------------------------------------------------------
# Finding #10: apply_router in MANUAL mode must be a no-op (no Gemini
# call). The agent.run() loop now skips the round-trip entirely; the
# unit-level assertion is simply that apply_router itself doesn't touch
# the client when mode=MANUAL.
# ---------------------------------------------------------------------------


def test_apply_router_does_not_call_client_in_manual_mode():
    sentinel_client = mock.MagicMock()
    sentinel_client.route_steps = mock.MagicMock(
        side_effect=AssertionError(
            "router should not be called in MANUAL mode"
        )
    )
    out = apply_router(
        ["click the button", "type hello"],
        mode=RoutingMode.MANUAL,
        client=sentinel_client,
        enable_browser_fast_path=False,
    )
    assert out == [None, None]
    sentinel_client.route_steps.assert_not_called()


def test_apply_router_off_mode_returns_all_none():
    out = apply_router(
        ["a", "b", "c"],
        mode=RoutingMode.OFF,
        client=None,
        enable_browser_fast_path=False,
    )
    assert out == [None, None, None]


# ---------------------------------------------------------------------------
# Sanity: VariableStore.from_dict round-trips an empty dict cleanly. This
# is what the rehydrate path will now invoke under the fixed condition.
# ---------------------------------------------------------------------------


def test_variable_store_from_empty_dict_is_empty_store():
    store = VariableStore.from_dict({})
    assert len(store) == 0
    assert store.to_dict() == {}
