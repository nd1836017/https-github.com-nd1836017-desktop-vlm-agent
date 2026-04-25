"""Regression tests for 3 post-merge bugs reported on the CSV/files rollup.

Each test guards a specific failure mode pinpointed by Devin Review against
main after PR #13 (CSV loops + file primitives) merged.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from agent.artifacts import ArtifactWriter
from agent.parser import (
    AttachFileCommand,
    CaptureForAiCommand,
    DownloadCommand,
)
from agent.vlm import (
    _RETRYABLE_STATUS,
    _call_with_retry,
    _status_code_of,
)


# ---------------------------------------------------------------------------
# Bug #2: file commands skipped save_after on the artifact writer.
#
# The regular CLICK / TYPE / etc. branch in _attempt_step takes a
# post-action screenshot and calls artifact_writer.save_after(...). The
# file-command branch (DOWNLOAD / ATTACH_FILE / CAPTURE_FOR_AI) used to
# return without ever calling save_after — so the after.png artifact for
# those steps was always missing in the postmortem bundle.
# ---------------------------------------------------------------------------
def _stub_workspace():
    """Minimal mock of FileWorkspace for the file-command path."""
    ws = MagicMock()
    return ws


def _run_attempt_step_with_file_command(
    cmd, *, artifact_writer
) -> tuple:
    """Drive _attempt_step for a single file-primitive command.

    Mocks out the VLM (so it returns ``cmd``), the file executors (so they
    succeed without doing real I/O), and the screenshot tools (so we don't
    need an X server). Returns whatever ``_attempt_step`` returned.
    """
    from agent.agent import _attempt_step
    from agent.screen import ScreenGeometry

    geometry = ScreenGeometry(width=1920, height=1080)
    vlm = MagicMock()
    vlm.plan_action.return_value = ("<raw>", cmd)
    workspace = _stub_workspace()

    fake_image = MagicMock()
    fake_image.size = (1920, 1080)

    with (
        patch("agent.agent.capture_screenshot", return_value=fake_image),
        patch("agent.agent.execute_download", return_value=(True, f"{cmd.kind} ok")),
        patch("agent.agent.execute_attach_file", return_value=(True, f"{cmd.kind} ok")),
        patch("agent.agent.execute_capture_for_ai", return_value=(True, f"{cmd.kind} ok")),
    ):
        return _attempt_step(
            step="dummy step",
            vlm=vlm,
            geometry=geometry,
            animation_buffer=0.0,
            max_parse_retries=0,
            history_summary="",
            previous_failure="",
            enable_two_stage_click=False,
            two_stage_crop_size_px=300,
            max_click_candidates=5,
            artifact_writer=artifact_writer,
            workspace=workspace,
            step_idx=7,
            extra_images=[],
        )


@pytest.mark.parametrize(
    "cmd",
    [
        DownloadCommand(url="https://example.com/x.pdf", filename="x.pdf"),
        AttachFileCommand(filename="x.pdf"),
        CaptureForAiCommand(filename=""),
    ],
)
def test_file_commands_save_after_screenshot_artifact(cmd):
    """All three file-primitive commands must call ``save_after`` on the
    artifact writer so the postmortem bundle has matching before/after
    screenshots, just like the regular action path. Pre-fix: ``save_after``
    was never called for these commands and ``step_NNN_after.png`` was
    always missing.
    """
    writer = MagicMock(spec=ArtifactWriter)

    verdict, _action = _run_attempt_step_with_file_command(
        cmd, artifact_writer=writer
    )
    assert verdict.passed is True

    # Both before and after must be saved for postmortem parity.
    assert writer.save_before.called, (
        f"{cmd.kind}: save_before must be called once before plan_action"
    )
    assert writer.save_after.called, (
        f"{cmd.kind}: save_after must be called after the file action — "
        f"was never invoked pre-fix"
    )
    # And the before/after must reference the same step index we passed in.
    before_args, _ = writer.save_before.call_args
    after_args, _ = writer.save_after.call_args
    assert before_args[0] == 7
    assert after_args[0] == 7


def test_file_commands_save_after_does_not_crash_when_capture_fails(caplog):
    """If the post-action ``capture_screenshot()`` itself fails, the agent
    must log a warning and continue — saving artifacts must NEVER block or
    crash the run.
    """
    writer = MagicMock(spec=ArtifactWriter)

    from agent.agent import _attempt_step
    from agent.screen import ScreenGeometry

    cmd = DownloadCommand(url="https://x/y.pdf", filename="y.pdf")
    geometry = ScreenGeometry(width=1920, height=1080)
    vlm = MagicMock()
    vlm.plan_action.return_value = ("<raw>", cmd)
    fake_image = MagicMock()
    fake_image.size = (1920, 1080)

    # First call returns fake image (the pre-action screenshot inside the
    # parse loop). Second call raises (the post-action capture for save_after).
    capture_results = [fake_image, RuntimeError("display gone")]

    def capture_side_effect(*_a, **_kw):
        out = capture_results.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    with (
        patch("agent.agent.capture_screenshot", side_effect=capture_side_effect),
        patch("agent.agent.execute_download", return_value=(True, "ok")),
    ):
        verdict, _ = _attempt_step(
            step="dummy",
            vlm=vlm,
            geometry=geometry,
            animation_buffer=0.0,
            max_parse_retries=0,
            history_summary="",
            previous_failure="",
            enable_two_stage_click=False,
            two_stage_crop_size_px=300,
            max_click_candidates=5,
            artifact_writer=writer,
            workspace=_stub_workspace(),
            step_idx=3,
            extra_images=[],
        )

    # The verdict still PASSes — file action ran and the failed snapshot
    # is just a missing artifact, not a step failure.
    assert verdict.passed is True
    # save_after was never called (capture failed first), but save_plan and
    # save_verdict still ran.
    writer.save_after.assert_not_called()
    writer.save_plan.assert_called_once()
    writer.save_verdict.assert_called_once()


# ---------------------------------------------------------------------------
# Bug #3: replan-budget log was misleading on the very first attempt.
#
# Pre-fix:
#   "Step attempt 1/3 (replan budget: 2 remaining)"  <- ambiguous, attempt 1
#                                                      isn't a replan, so
#                                                      "replan budget" reads
#                                                      as off-by-one.
#
# Post-fix the message is unambiguous about what's the initial attempt vs.
# which replan number we're using.
# ---------------------------------------------------------------------------
def test_replan_log_is_clear_on_initial_attempt(caplog):
    """First attempt should log as 'initial' with N replans available."""
    from agent.agent import run_step

    fake_screenshot = MagicMock()
    fake_screenshot.size = (1920, 1080)
    vlm = MagicMock()
    # PASS on first try — only one log line emitted.
    vlm.plan_action.return_value = ("WAIT [0.5]", None)
    vlm.verify.return_value.passed = True
    vlm.verify.return_value.reason = "ok"

    from agent.history import History
    from agent.screen import ScreenGeometry

    with (
        patch("agent.agent.capture_screenshot", return_value=fake_screenshot),
        patch("agent.agent.parse_command") as parse_mock,
        patch("agent.agent.execute"),
        caplog.at_level("INFO", logger="agent.agent"),
    ):
        from agent.parser import WaitCommand
        parse_mock.return_value = WaitCommand(seconds=0.5)
        run_step(
            step="dummy",
            vlm=vlm,
            geometry=ScreenGeometry(width=1920, height=1080),
            animation_buffer=0.0,
            max_parse_retries=0,
            max_replans=2,
            history=History(window=3),
            enable_two_stage_click=False,
            step_idx=1,
        )

    initial_logs = [
        r.getMessage() for r in caplog.records
        if "Step attempt 1/3" in r.getMessage()
    ]
    assert initial_logs, "Initial attempt must log 'Step attempt 1/3'"
    msg = initial_logs[0]
    # New format: "(initial — N replan(s) available if this fails)"
    assert "initial" in msg, f"expected 'initial' marker, got: {msg!r}"
    assert "2 replan" in msg, f"expected '2 replan(s) available', got: {msg!r}"


def test_replan_log_is_clear_on_replan_attempts(caplog):
    """Replan attempts should log as 'replan K/max' with remaining count."""
    from agent.agent import run_step
    from agent.history import History
    from agent.parser import WaitCommand
    from agent.screen import ScreenGeometry
    from agent.vlm import VerificationResult

    fake_screenshot = MagicMock()
    fake_screenshot.size = (1920, 1080)
    vlm = MagicMock()
    vlm.plan_action.return_value = ("WAIT [0.5]", None)
    # FAIL twice, PASS on the third.
    vlm.verify.side_effect = [
        VerificationResult(passed=False, reason="nope"),
        VerificationResult(passed=False, reason="nope"),
        VerificationResult(passed=True, reason="ok"),
    ]

    with (
        patch("agent.agent.capture_screenshot", return_value=fake_screenshot),
        patch("agent.agent.parse_command", return_value=WaitCommand(seconds=0.5)),
        patch("agent.agent.execute"),
        caplog.at_level("INFO", logger="agent.agent"),
    ):
        run_step(
            step="dummy",
            vlm=vlm,
            geometry=ScreenGeometry(width=1920, height=1080),
            animation_buffer=0.0,
            max_parse_retries=0,
            max_replans=2,
            history=History(window=3),
            enable_two_stage_click=False,
            step_idx=1,
        )

    msgs = [r.getMessage() for r in caplog.records]
    replan1 = [m for m in msgs if "Step attempt 2/3" in m]
    replan2 = [m for m in msgs if "Step attempt 3/3" in m]
    assert replan1 and "replan 1/2" in replan1[0]
    assert replan1 and "1 replan" in replan1[0], replan1[0]
    assert replan2 and "replan 2/2" in replan2[0]
    assert replan2 and "0 replan" in replan2[0], replan2[0]


# ---------------------------------------------------------------------------
# Bug #4: _call_with_retry status-code lookup was fragile.
#
# Pre-fix:  status = getattr(exc, "status_code", getattr(exc, "code", None))
#
# google-genai's APIError exposes the HTTP status as ``code``, not
# ``status_code``. The previous code worked TODAY because of the inner
# fallback, but if a future SDK version moves the attribute (or wraps the
# exception in a way that drops both), we'd miss legitimate retryable
# errors. The new ``_status_code_of`` helper tries multiple attrs AND
# falls back to parsing the leading status from str(exc).
# ---------------------------------------------------------------------------
class _ExceptionWithCode(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"{code} ServerError")
        self.code = code


class _ExceptionWithStatusCode(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"{code} ServerError")
        self.status_code = code


class _ExceptionWithHttpStatus(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"{code} ServerError")
        self.http_status = code


class _ExceptionStringOnly(Exception):
    """No structured attribute — only the leading code in the message."""


def test_status_code_of_reads_code_attribute():
    assert _status_code_of(_ExceptionWithCode(503)) == 503
    assert _status_code_of(_ExceptionWithCode(429)) == 429


def test_status_code_of_reads_status_code_attribute():
    """When ``code`` is missing but ``status_code`` is set."""
    assert _status_code_of(_ExceptionWithStatusCode(500)) == 500


def test_status_code_of_reads_http_status_attribute():
    """When neither ``code`` nor ``status_code`` exists but ``http_status`` does."""
    assert _status_code_of(_ExceptionWithHttpStatus(502)) == 502


def test_status_code_of_falls_back_to_message_string():
    """Last-resort: parse leading 3-digit code from the exception message.

    APIError formats its message as ``"{code} {status}. {details}"`` —
    even if ALL structured attributes are dropped (e.g. by a wrapper),
    we can still extract the status from the message and route correctly.
    """
    e = _ExceptionStringOnly("503 UNAVAILABLE. Service is overloaded.")
    assert _status_code_of(e) == 503


def test_status_code_of_returns_none_for_unknown_shape():
    """Returns None when no attr is set and the message has no leading code."""
    e = _ExceptionStringOnly("connection reset by peer")
    assert _status_code_of(e) is None


def test_status_code_of_handles_string_attribute_value():
    """Some SDK shims store the code as a string. Accept that too."""

    class _StrCode(Exception):
        def __init__(self) -> None:
            super().__init__("err")
            self.code = "503"

    assert _status_code_of(_StrCode()) == 503


def test_call_with_retry_retries_when_status_only_in_message():
    """Even when neither ``code`` nor ``status_code`` is set, the retry guard
    must still recognize a retryable status from the message and retry.

    Pre-fix this would re-raise immediately because both attrs were None,
    losing legitimate transient retries on SDK versions that drop them.
    """
    calls = {"n": 0}

    class _BareServerError(genai_errors.ServerError):
        # Override __init__ so neither code nor status_code is set on the
        # instance directly — only the message string carries the status.
        def __init__(self) -> None:
            # Bypass APIError.__init__ — store nothing structurally.
            Exception.__init__(self, "503 UNAVAILABLE. The model is overloaded.")

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _BareServerError()
        return "OK"

    with patch("agent.vlm.time.sleep"):
        result = _call_with_retry(
            fn,
            label="t",
            max_attempts=3,
            base_delay_seconds=0.01,
            max_delay_seconds=0.5,
        )
    assert result == "OK"
    assert calls["n"] == 2


def test_call_with_retry_still_raises_on_400_with_string_status():
    """A non-retryable 400 with only the message-string code must still raise
    immediately (no retries) — this proves the new helper doesn't loosen the
    safety net by, say, accidentally matching ``200`` somewhere in the body.
    """

    class _Bare400(genai_errors.ClientError):
        def __init__(self) -> None:
            Exception.__init__(self, "400 INVALID_ARGUMENT. Bad request.")

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _Bare400()

    with patch("agent.vlm.time.sleep"), pytest.raises(genai_errors.ClientError):
        _call_with_retry(
            fn,
            label="t",
            max_attempts=4,
            base_delay_seconds=0.01,
            max_delay_seconds=0.5,
        )
    assert calls["n"] == 1, "400 must NOT trigger retry"


def test_retryable_status_set_includes_expected_codes():
    """Sanity check: the retryable set didn't accidentally narrow."""
    assert {429, 500, 502, 503, 504} == _RETRYABLE_STATUS


# ---------------------------------------------------------------------------
# Bug #3 (PR S round 2): parse-failure path skipped save_plan / save_verdict.
#
# Every other return path from _attempt_step writes both ``save_plan`` and
# ``save_verdict`` so the artifact bundle is symmetric. The parse-failure
# fall-through used to skip both, leaving a ``before.png`` with no plan or
# verdict — confusing when triaging a stuck run. The fix synthesizes
# ``<parse-failed>`` placeholders so the bundle stays consistent.
# ---------------------------------------------------------------------------


def test_parse_failure_writes_synthetic_plan_and_verdict():
    from unittest.mock import MagicMock, patch

    from agent.agent import _attempt_step
    from agent.artifacts import ArtifactWriter
    from agent.screen import ScreenGeometry

    writer = MagicMock(spec=ArtifactWriter)
    fake_image = MagicMock()
    fake_image.size = (1920, 1080)
    geometry = ScreenGeometry(width=1920, height=1080)
    vlm = MagicMock()
    vlm.plan_action.return_value = ("garbage VLM output", None)

    with (
        patch("agent.agent.capture_screenshot", return_value=fake_image),
        patch("agent.agent.parse_command", return_value=None),
    ):
        verdict, action_text = _attempt_step(
            step="step that will never parse",
            vlm=vlm,
            geometry=geometry,
            animation_buffer=0.0,
            max_parse_retries=1,
            history_summary="",
            previous_failure="",
            enable_two_stage_click=False,
            two_stage_crop_size_px=300,
            max_click_candidates=5,
            artifact_writer=writer,
            workspace=None,
            step_idx=42,
            extra_images=[],
        )

    assert verdict.passed is False
    assert "Parse failure" in verdict.reason
    assert action_text == "<parse-failed>"

    # The whole point of this regression test: bundle is symmetric.
    assert writer.save_plan.called, "save_plan must run on parse failure"
    assert writer.save_verdict.called, "save_verdict must run on parse failure"

    plan_args, _ = writer.save_plan.call_args
    verdict_args, _ = writer.save_verdict.call_args
    assert plan_args[0] == 42
    assert verdict_args[0] == 42
    # save_verdict(step_idx, passed, reason)
    assert verdict_args[1] is False
    assert "Parse failure" in verdict_args[2]
