"""End-to-end tests for the agent's run loop.

These tests replace GeminiClient with a scripted fake so we exercise
parse-retry, replan-on-verification-failure, checkpoint + resume, and
short-term memory injection — all without hitting the real API or a
display server.
"""
from __future__ import annotations

import sys
from dataclasses import replace as dataclasses_replace
from typing import cast
from unittest import mock

import pytest
from PIL import Image

from agent.agent import run, run_step
from agent.config import Config
from agent.history import History
from agent.screen import ScreenGeometry
from agent.state import load_state
from agent.vlm import VerificationResult


class FakeClient:
    """Drop-in replacement for GeminiClient; reads plans/verdicts from queues."""

    def __init__(self, plan_outputs, verify_outputs):
        self._plan_outputs = list(plan_outputs)
        self._verify_outputs = list(verify_outputs)
        self.plan_calls: list[dict] = []
        self.verify_calls = 0

    def plan_action(
        self,
        step,
        screenshot,
        history_summary: str = "",
        previous_failure: str = "",
        extra_images=None,
    ):
        self.plan_calls.append(
            {
                "step": step,
                "history_summary": history_summary,
                "previous_failure": previous_failure,
                "extra_images": list(extra_images or []),
            }
        )
        out = self._plan_outputs.pop(0)
        # FakeClient stays text-only — it always returns (raw, None) so the
        # agent falls back to the regex parser just like the legacy path.
        if isinstance(out, tuple):
            return out
        return out, None

    def verify(self, goal, screenshot):
        self.verify_calls += 1
        out = self._verify_outputs.pop(0)
        if isinstance(out, VerificationResult):
            return out
        passed, reason = out
        return VerificationResult(passed=passed, reason=reason)


@pytest.fixture
def fake_geometry():
    return ScreenGeometry(width=1920, height=1080)


@pytest.fixture(autouse=True)
def patch_pyautogui():
    """Replace pyautogui wherever it's lazily imported."""
    fake_img = Image.new("RGB", (8, 8), color=(0, 0, 0))
    fake_pyautogui = mock.MagicMock()
    fake_pyautogui.size.return_value = mock.MagicMock(width=1920, height=1080)
    fake_pyautogui.screenshot.return_value = fake_img
    fake_pyautogui.FAILSAFE = False
    with mock.patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        yield fake_pyautogui


def _cfg(tmp_path, tasks_text: str, **overrides) -> Config:
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text(tasks_text, encoding="utf-8")
    defaults = dict(
        gemini_api_key="fake",
        gemini_model="fake-model",
        tasks_file=tasks_file,
        animation_buffer_seconds=0.0,
        max_step_retries=1,
        max_replans_per_step=2,
        history_window=5,
        state_file=tmp_path / ".agent_state.json",
        # Legacy-loop tests default to two-stage OFF so the FakeClient doesn't
        # need to implement the refine/disambiguate surface. Two-stage gets
        # its own dedicated test module.
        enable_two_stage_click=False,
        two_stage_crop_size_px=300,
        max_click_candidates=5,
        click_min_delay_seconds=0.0,
        click_max_delay_seconds=0.0,
        type_min_interval_seconds=0.0,
        type_max_interval_seconds=0.0,
        gemini_retry_max_attempts=1,
        gemini_retry_base_delay_seconds=0.0,
        gemini_retry_max_delay_seconds=0.0,
        log_redact_type=True,
        enable_json_output=False,
        max_total_replans=0,
        save_run_artifacts=False,
        run_artifacts_dir=tmp_path / "runs",
        rpd_limit=0,
        rpd_warn_threshold=0.75,
        rpd_halt_threshold=0.95,
        file_mode=None,
        workdir=None,
        log_level="INFO",
    )
    defaults.update(overrides)
    return Config(**defaults)


# -----------------------------------------------------------------------------
# Parse-retry behaviour (pre-existing spec)
# -----------------------------------------------------------------------------
def test_retry_on_parse_failure(fake_geometry, patch_pyautogui):
    client = FakeClient(
        plan_outputs=["I'm not sure, maybe click?", "CLICK [100,200]"],
        verify_outputs=[VerificationResult(passed=True, reason="VERDICT: PASS — ok")],
    )
    history = History(window=5)
    result = run_step(
        step="click the thing",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=1,
        max_replans=0,
        history=history,
        enable_two_stage_click=False,
    )
    assert result.passed is True
    assert len(client.plan_calls) == 2
    # Scaled from normalized 100,200 on a 1920x1080 screen.
    assert patch_pyautogui.click.call_args.args == (192, 216)


def test_parse_failure_exhausted_returns_fail(fake_geometry, patch_pyautogui):
    client = FakeClient(
        plan_outputs=["prose one", "prose two"],
        verify_outputs=[],
    )
    result = run_step(
        step="click",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=1,
        max_replans=0,
        history=History(window=0),
        enable_two_stage_click=False,
    )
    assert result.passed is False
    assert "parse failure" in result.reason.lower()
    assert patch_pyautogui.click.call_count == 0


# -----------------------------------------------------------------------------
# Replan-on-failure with budget
# -----------------------------------------------------------------------------
def test_replan_succeeds_after_one_verification_failure(
    fake_geometry, patch_pyautogui
):
    """Verifier fails once, then passes. Replan budget 2 → overall success."""
    client = FakeClient(
        plan_outputs=["CLICK [100,100]", "CLICK [500,500]"],
        verify_outputs=[
            VerificationResult(passed=False, reason="VERDICT: FAIL — missed target"),
            VerificationResult(passed=True, reason="VERDICT: PASS — hit"),
        ],
    )
    history = History(window=5)
    result = run_step(
        step="click the button",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=2,
        history=history,
        enable_two_stage_click=False,
    )
    assert result.passed is True
    assert len(client.plan_calls) == 2
    # Second plan call must have been primed with the failure reason.
    assert "missed target" in client.plan_calls[1]["previous_failure"]
    # History records the final successful attempt.
    records = list(history)
    assert len(records) == 1
    assert records[0].passed is True
    assert records[0].action_text == "CLICK [500,500]"


def test_replan_budget_exhausts_and_halts(fake_geometry, patch_pyautogui):
    """All 3 attempts fail. Replan budget 2 + initial attempt = 3 plan calls, then FAIL."""
    client = FakeClient(
        plan_outputs=["CLICK [1,1]", "CLICK [2,2]", "CLICK [3,3]"],
        verify_outputs=[
            VerificationResult(passed=False, reason="VERDICT: FAIL — nope"),
            VerificationResult(passed=False, reason="VERDICT: FAIL — still no"),
            VerificationResult(passed=False, reason="VERDICT: FAIL — give up"),
        ],
    )
    history = History(window=5)
    result = run_step(
        step="click the button",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=2,
        history=history,
        enable_two_stage_click=False,
    )
    assert result.passed is False
    assert len(client.plan_calls) == 3, "expected 1 initial + 2 replan attempts"
    records = list(history)
    assert len(records) == 1
    assert records[0].passed is False


# -----------------------------------------------------------------------------
# History is injected into subsequent plan calls
# -----------------------------------------------------------------------------
def test_history_injected_into_next_step_plan(
    fake_geometry, patch_pyautogui, tmp_path
):
    cfg = _cfg(
        tmp_path,
        "first step\nsecond step\n",
        max_replans_per_step=0,
    )
    client = FakeClient(
        plan_outputs=["CLICK [1,1]", "CLICK [2,2]"],
        verify_outputs=[
            VerificationResult(passed=True, reason="VERDICT: PASS — ok1"),
            VerificationResult(passed=True, reason="VERDICT: PASS — ok2"),
        ],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code = run(cfg)

    assert exit_code == 0
    # First plan call has empty history; second plan call has the first step's record.
    assert client.plan_calls[0]["history_summary"] == ""
    second_summary = client.plan_calls[1]["history_summary"]
    assert "first step" in second_summary
    assert "CLICK [1,1]" in second_summary
    assert "PASS" in second_summary


# -----------------------------------------------------------------------------
# Checkpoint + resume
# -----------------------------------------------------------------------------
def test_checkpoint_written_after_each_verified_step(
    fake_geometry, patch_pyautogui, tmp_path
):
    cfg = _cfg(tmp_path, "a\nb\nc\n", max_replans_per_step=0)
    client = FakeClient(
        plan_outputs=["CLICK [1,1]", "CLICK [2,2]", "CLICK [3,3]"],
        verify_outputs=[
            VerificationResult(passed=True, reason="PASS"),
            VerificationResult(passed=True, reason="PASS"),
            VerificationResult(passed=True, reason="PASS"),
        ],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code = run(cfg)

    assert exit_code == 0
    state = load_state(cfg.state_file)
    assert state is not None
    assert state.last_completed_step == 3
    assert state.total_steps == 3


def test_halt_preserves_checkpoint_of_last_success(
    fake_geometry, patch_pyautogui, tmp_path, capsys
):
    cfg = _cfg(tmp_path, "a\nb\nc\n", max_replans_per_step=0)
    client = FakeClient(
        plan_outputs=["CLICK [1,1]", "CLICK [2,2]"],
        verify_outputs=[
            VerificationResult(passed=True, reason="PASS"),
            VerificationResult(passed=False, reason="VERDICT: FAIL — blocked"),
        ],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code = run(cfg)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "HALT at step 2/3" in captured.err
    assert "--resume" in captured.err
    state = load_state(cfg.state_file)
    assert state is not None
    assert state.last_completed_step == 1  # only step 1 succeeded


def test_resume_skips_already_completed_steps(
    fake_geometry, patch_pyautogui, tmp_path
):
    cfg = _cfg(tmp_path, "a\nb\nc\n", max_replans_per_step=0)
    # First run: complete steps 1 and 2, fail on step 3.
    client1 = FakeClient(
        plan_outputs=["CLICK [1,1]", "CLICK [2,2]", "CLICK [3,3]"],
        verify_outputs=[
            VerificationResult(passed=True, reason="PASS"),
            VerificationResult(passed=True, reason="PASS"),
            VerificationResult(passed=False, reason="FAIL"),
        ],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client1),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code1 = run(cfg)
    assert exit_code1 == 1

    # Second run with --resume: only step 3 should execute.
    client2 = FakeClient(
        plan_outputs=["CLICK [9,9]"],
        verify_outputs=[VerificationResult(passed=True, reason="PASS")],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client2),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code2 = run(cfg, resume=True)

    assert exit_code2 == 0
    assert len(client2.plan_calls) == 1
    assert client2.plan_calls[0]["step"] == "c"
    state = load_state(cfg.state_file)
    assert state is not None
    assert state.last_completed_step == 3


def test_resume_ignored_if_tasks_file_changed(
    fake_geometry, patch_pyautogui, tmp_path
):
    """A stale checkpoint (different total_steps) must be ignored."""
    cfg = _cfg(tmp_path, "a\nb\nc\n")
    # Write a stale state claiming step 2 of 5 is done; total_steps won't match.
    from agent.state import AgentState, save_state

    stale = AgentState(
        version=1,
        tasks_file=str(cfg.tasks_file),
        total_steps=5,
        last_completed_step=2,
    )
    save_state(cfg.state_file, stale)

    client = FakeClient(
        plan_outputs=["CLICK [1,1]", "CLICK [2,2]", "CLICK [3,3]"],
        verify_outputs=[
            VerificationResult(passed=True, reason="PASS"),
            VerificationResult(passed=True, reason="PASS"),
            VerificationResult(passed=True, reason="PASS"),
        ],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code = run(cfg, resume=True)

    assert exit_code == 0
    # All 3 steps must have been re-run from the top.
    assert len(client.plan_calls) == 3


# -----------------------------------------------------------------------------
# Graceful error handling when the tasks file is missing (regression test).
# -----------------------------------------------------------------------------
def test_run_returns_exit_code_2_when_tasks_file_missing(tmp_path, capsys):
    """``run()`` must catch FileNotFoundError and exit cleanly.

    Previously only ``TasksLoadError`` was caught, leaving the most common
    failure mode (typo'd path) crashing with a raw traceback.
    """
    cfg = _cfg(tmp_path, "Step one\n")
    # Point the tasks file at something that doesn't exist.
    cfg = dataclasses_replace(cfg, tasks_file=tmp_path / "does-not-exist.txt")

    exit_code = run(cfg)
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "[tasks error]" in captured.err
    assert "does-not-exist.txt" in captured.err


# -----------------------------------------------------------------------------
# CAPTURE_FOR_AI / FEED-mode wiring: verify consume_feed() → plan_action.
# -----------------------------------------------------------------------------
def test_capture_for_ai_feeds_next_plan_call(tmp_path, fake_geometry):
    """A CAPTURE_FOR_AI step must drain the feed buffer into the next
    plan_action call's ``extra_images``. Regression test for Devin Review
    finding that the buffer was populated but never consumed.
    """
    from agent.files import FileMode

    tasks_text = "CAPTURE_FOR_AI []\nClick the OK button\n"
    cfg = _cfg(tmp_path, tasks_text, file_mode=FileMode.FEED, workdir=None)

    client = FakeClient(
        plan_outputs=[
            "CAPTURE_FOR_AI []",  # step 1: emit the capture command
            "CLICK [500,500]",     # step 2: planner sees the captured image
        ],
        verify_outputs=[
            # Step 1 (CAPTURE_FOR_AI) bypasses the verifier — synthesises
            # its own verdict — so only step 2 needs a verify entry.
            VerificationResult(passed=True, reason="PASS"),
        ],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code = run(cfg)
    assert exit_code == 0

    # Two plan calls: step 1 (no images) and step 2 (1 image from buffer).
    assert len(client.plan_calls) == 2
    assert client.plan_calls[0]["extra_images"] == []
    assert len(client.plan_calls[1]["extra_images"]) == 1, (
        "step 2's plan_action should have been handed the screenshot bytes "
        "buffered by step 1's CAPTURE_FOR_AI"
    )


def test_capture_for_ai_buffer_survives_parse_retry(tmp_path, fake_geometry):
    """Parse-retries within a single step must reuse the same feed buffer.

    Regression test: consume_feed() is destructive, so if it were called
    inside the parse-retry loop the second attempt would see no images.
    """
    from agent.files import FileMode

    tasks_text = "CAPTURE_FOR_AI []\nDo the thing\n"
    cfg = _cfg(tmp_path, tasks_text, file_mode=FileMode.FEED, workdir=None)

    client = FakeClient(
        plan_outputs=[
            "CAPTURE_FOR_AI []",     # step 1
            "this isn't a command",  # step 2 attempt 1: unparseable
            "CLICK [10,10]",          # step 2 attempt 2: succeeds
        ],
        verify_outputs=[
            VerificationResult(passed=True, reason="PASS"),
        ],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code = run(cfg)
    assert exit_code == 0
    # 3 plan calls; both step-2 attempts must have seen the captured image.
    assert len(client.plan_calls) == 3
    assert client.plan_calls[0]["extra_images"] == []  # step 1 had no buffer
    assert len(client.plan_calls[1]["extra_images"]) == 1, (
        "step 2 attempt 1 should see the captured image"
    )
    assert len(client.plan_calls[2]["extra_images"]) == 1, (
        "step 2 attempt 2 (parse retry) should ALSO see the captured image; "
        "consume_feed() must not be called inside the parse-retry loop"
    )


def test_capture_for_ai_buffer_survives_replan(tmp_path, fake_geometry):
    """Replan attempts (verifier FAIL → new _attempt_step) must reuse the
    same feed buffer. Regression test: previously consume_feed() lived in
    _attempt_step so the second invocation drained nothing.
    """
    from agent.files import FileMode

    tasks_text = "CAPTURE_FOR_AI []\nClick the OK button\n"
    cfg = _cfg(
        tmp_path,
        tasks_text,
        file_mode=FileMode.FEED,
        workdir=None,
        max_replans_per_step=2,
    )

    client = FakeClient(
        plan_outputs=[
            "CAPTURE_FOR_AI []",  # step 1
            "CLICK [10,10]",       # step 2 attempt 1
            "CLICK [20,20]",       # step 2 replan #1 (after FAIL)
        ],
        verify_outputs=[
            VerificationResult(passed=False, reason="wrong button"),
            VerificationResult(passed=True, reason="ok now"),
        ],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=client),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        exit_code = run(cfg)
    assert exit_code == 0
    assert len(client.plan_calls) == 3
    # Step 1 ran without buffer.
    assert client.plan_calls[0]["extra_images"] == []
    # Step 2 attempt 1 received the captured image.
    assert len(client.plan_calls[1]["extra_images"]) == 1
    # Step 2 replan ALSO received the captured image (would have been [] before).
    assert len(client.plan_calls[2]["extra_images"]) == 1, (
        "step 2 replan attempt should still see the captured image; "
        "consume_feed must be hoisted to run_step level"
    )
