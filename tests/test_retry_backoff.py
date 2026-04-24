"""Unit tests for the Gemini retry-with-backoff helper.

These tests mock ``time.sleep`` so the backoff is instant, and only assert
on call counts / delay-cap arithmetic rather than real timing.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from google.genai import errors as genai_errors

from agent.vlm import _RETRYABLE_STATUS, _call_with_retry


def _server_error(code: int) -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"error": {"message": "x"}})


def _client_error(code: int) -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"error": {"message": "x"}})


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _server_error(503)
        return "OK"

    with patch("agent.vlm.time.sleep") as sleep_mock:
        result = _call_with_retry(
            fn,
            label="plan",
            max_attempts=5,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
        )

    assert result == "OK"
    assert calls["n"] == 3
    # Two transient failures -> two sleeps.
    assert sleep_mock.call_count == 2


def test_retry_raises_after_exhausting_attempts():
    def fn():
        raise _client_error(429)

    with patch("agent.vlm.time.sleep"), pytest.raises(genai_errors.ClientError) as excinfo:
        _call_with_retry(
            fn,
            label="verify",
            max_attempts=3,
            base_delay_seconds=0.5,
            max_delay_seconds=10.0,
        )
    assert excinfo.value.code == 429


def test_non_retryable_error_raised_immediately():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _client_error(400)  # bad request — not retryable

    with patch("agent.vlm.time.sleep") as sleep_mock, pytest.raises(genai_errors.ClientError):
        _call_with_retry(
            fn,
            label="refine",
            max_attempts=5,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
        )

    assert calls["n"] == 1
    assert sleep_mock.call_count == 0


def test_backoff_is_capped_by_max_delay():
    captured: list[float] = []

    def record_sleep(seconds: float) -> None:
        captured.append(seconds)

    def fn():
        raise _server_error(503)

    # base=10, max_delay=30, attempts=5 -> caps = 10, 20, 30, 30
    with (
        patch("agent.vlm.time.sleep", side_effect=record_sleep),
        patch("agent.vlm.random.uniform", side_effect=lambda lo, hi: hi),
        pytest.raises(genai_errors.ServerError),
    ):
        _call_with_retry(
            fn,
            label="plan",
            max_attempts=5,
            base_delay_seconds=10.0,
            max_delay_seconds=30.0,
        )

    assert captured == [10.0, 20.0, 30.0, 30.0]


@pytest.mark.parametrize("code", sorted(_RETRYABLE_STATUS))
def test_all_retryable_codes_trigger_retry(code: int):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            if code >= 500:
                raise _server_error(code)
            raise _client_error(code)
        return "done"

    with patch("agent.vlm.time.sleep"):
        assert (
            _call_with_retry(
                fn,
                label="t",
                max_attempts=4,
                base_delay_seconds=0.1,
                max_delay_seconds=1.0,
            )
            == "done"
        )
    assert calls["n"] == 2
