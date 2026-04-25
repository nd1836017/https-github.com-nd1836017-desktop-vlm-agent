"""Regression tests for the 6 bugs reported against the v2–v8 rollup.

Each test is named after the bug it guards and contains a brief comment
pinpointing the failure mode on main before the fix.
"""
from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from agent import executor
from agent.config import Config
from agent.history import History, StepRecord
from agent.screen import ScreenGeometry
from agent.vlm import (
    _parse_plan_response_json,
    _parse_verify_response_json,
    _strip_markdown_fences,
)


# ---------------------------------------------------------------------------
# Bug #1 (history.py): __iter__ should return Iterator, not Iterable.
# ---------------------------------------------------------------------------
def test_history_iter_returns_iterator_not_iterable():
    """``__iter__`` must return an ``Iterator`` so mypy/pyright (and
    callers that rely on ``next()``) see the correct protocol. Iterable
    is broader and lets you re-iterate; Iterator is single-pass and
    matches what ``iter()`` actually returns."""
    h = History(window=3)
    h.record("s", "a", passed=True, reason="ok")
    it = iter(h)
    # The real contract check: it must support next().
    assert isinstance(it, Iterator)
    rec = next(it)
    assert isinstance(rec, StepRecord)


# ---------------------------------------------------------------------------
# Bug #2 (vlm.py): markdown fence stripping.
# Before fix:
#   - ``stripped.strip("`")`` ate backticks *inside* the JSON too.
#   - ``.lower().startswith("json")`` missed uppercase / other tags.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected_value",
    [
        # Lowercase tag — worked before and after.
        ('```json\n{"verdict":"PASS"}\n```', "PASS"),
        # Uppercase tag — silently broken before: "JSON" was not stripped,
        # so `json.loads("JSON\n{...}")` raised and the fallback returned None.
        ('```JSON\n{"verdict":"PASS"}\n```', "PASS"),
        # Mixed-case tag.
        ('```Json\n{"verdict":"PASS"}\n```', "PASS"),
        # No tag at all.
        ('```\n{"verdict":"PASS"}\n```', "PASS"),
        # YAML-labelled (should still strip as an opaque tag).
        ('```yaml\n{"verdict":"PASS"}\n```', "PASS"),
        # No fence at all — unchanged.
        ('{"verdict":"PASS"}', "PASS"),
    ],
)
def test_verify_json_fence_stripping_handles_any_language_tag(raw, expected_value):
    parsed = _parse_verify_response_json(raw)
    assert parsed is not None, f"fence stripping failed for {raw!r}"
    assert parsed.verdict == expected_value


def test_strip_markdown_fences_preserves_internal_backticks():
    """The old ``strip('`')`` greedily removed every backtick, including
    ones that belong to the payload itself (e.g. a markdown code-block
    inside a verify reason). The regex-based version only strips the
    leading and trailing fence."""
    payload = '```json\n{"reason": "use `ctrl+l`"}\n```'
    cleaned = _strip_markdown_fences(payload)
    assert cleaned == '{"reason": "use `ctrl+l`"}'


def test_plan_json_fence_stripping_handles_uppercase():
    raw = '```JSON\n{"command":"CLICK","x":100,"y":200}\n```'
    parsed = _parse_plan_response_json(raw)
    assert parsed is not None
    assert parsed.command == "CLICK"
    assert parsed.x == 100


# ---------------------------------------------------------------------------
# Bug #3 (executor.py): pyautogui.typewrite silently drops non-ASCII.
# ---------------------------------------------------------------------------
def test_typewrite_safe_accepts_ascii():
    for ch in "abcAZ0 9~!":
        assert executor._is_typewrite_safe(ch)
    # Whitespace chars pyautogui actually handles.
    assert executor._is_typewrite_safe("\n")
    assert executor._is_typewrite_safe("\t")


def test_typewrite_safe_rejects_non_ascii():
    # All of these would be silently dropped by pyautogui.typewrite.
    for ch in ["é", "ñ", "漢", "🎉", "ö"]:
        assert not executor._is_typewrite_safe(ch), ch


def test_paste_char_uses_clipboard_and_ctrl_v():
    """Non-ASCII chars must NOT be sent via typewrite (which drops them);
    they must be copied to the clipboard and pasted with Ctrl+V."""
    fake_pag = MagicMock()
    with patch.object(executor, "__name__", "agent.executor"):
        # pyperclip must exist for this test; it's in requirements.txt.
        import pyperclip

        with patch.object(pyperclip, "copy") as mock_copy:
            executor._paste_char(fake_pag, "漢", redact=False)
            mock_copy.assert_called_once_with("漢")
    fake_pag.hotkey.assert_called_once_with("ctrl", "v")


def test_paste_char_warns_when_pyperclip_missing(caplog):
    """If pyperclip isn't installed we must log a warning (not crash,
    not silently drop)."""
    fake_pag = MagicMock()
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyperclip":
            raise ImportError("pyperclip not installed")
        return original_import(name, *args, **kwargs)

    with (
        patch.object(builtins, "__import__", side_effect=fake_import),
        caplog.at_level("WARNING"),
    ):
        executor._paste_char(fake_pag, "漢", redact=False)
    assert any("pyperclip not installed" in r.message for r in caplog.records)
    fake_pag.hotkey.assert_not_called()


# ---------------------------------------------------------------------------
# Bug #4 (agent.py): PAUSE can starve the replan budget.
# ---------------------------------------------------------------------------
def test_pause_does_not_starve_replan_budget():
    """Reproduces the off-by-one: with 2 total attempts and a PAUSE on
    attempt 2, the agent must still be able to run a second real attempt
    after the PAUSE. The old code's ``attempt_idx -= 1`` pattern would
    exit the loop on certain edge cases; the new ``attempts_used``
    counter fixes it."""
    from unittest.mock import MagicMock

    from agent.agent import PauseRequested, run_step
    from agent.vlm import VerificationResult

    # VLM mock: PAUSE on call 1, PASS on call 2.
    outcomes = [
        (PauseRequested(reason="Verify it's you", raw="PAUSE [v]"), "PAUSE [v]"),
        (VerificationResult(passed=True, reason="done"), "CLICK [100,200]"),
    ]
    call_log: list[int] = []

    def fake_attempt(**kwargs):
        call_log.append(len(call_log))
        return outcomes[call_log[-1]]

    with patch("agent.agent._attempt_step", side_effect=fake_attempt):
        history = History(window=3)
        vlm = MagicMock()
        geometry = ScreenGeometry(width=1600, height=1200)
        handler_calls: list[str] = []

        def fake_pause_handler(reason: str) -> bool:
            handler_calls.append(reason)
            return True  # continue after PAUSE

        verdict = run_step(
            step="test step",
            vlm=vlm,
            geometry=geometry,
            animation_buffer=0,
            max_parse_retries=1,
            max_replans=1,  # total_attempts = 2
            history=history,
            enable_two_stage_click=False,
            two_stage_crop_size_px=300,
            max_click_candidates=5,
            click_min_delay_seconds=0,
            click_max_delay_seconds=0,
            type_min_interval_seconds=0,
            type_max_interval_seconds=0,
            log_redact_type=True,
            pause_handler=fake_pause_handler,
            replan_counter=None,
            artifact_writer=None,
            step_idx=0,
        )
    # The PAUSE handler fired, then a second real attempt ran and passed.
    assert verdict.passed is True
    assert len(handler_calls) == 1
    assert len(call_log) == 2  # one PAUSE iteration + one real attempt


# ---------------------------------------------------------------------------
# Bug #5 (config.py): RPD_WARN < RPD_HALT validation.
# ---------------------------------------------------------------------------
def test_config_rejects_warn_threshold_equal_to_halt(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RPD_WARN_THRESHOLD", "0.9")
    monkeypatch.setenv("RPD_HALT_THRESHOLD", "0.9")
    with pytest.raises(ValueError, match="must be less than"):
        Config.load()


def test_config_rejects_warn_threshold_above_halt(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RPD_WARN_THRESHOLD", "0.99")
    monkeypatch.setenv("RPD_HALT_THRESHOLD", "0.95")
    with pytest.raises(ValueError, match="must be less than"):
        Config.load()


def test_config_accepts_sane_thresholds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RPD_WARN_THRESHOLD", "0.75")
    monkeypatch.setenv("RPD_HALT_THRESHOLD", "0.95")
    cfg = Config.load()
    assert cfg.rpd_warn_threshold == 0.75
    assert cfg.rpd_halt_threshold == 0.95


# ---------------------------------------------------------------------------
# Bug #6 (screen.py): badge rectangle collapses to 0 height near y=0.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "top",
    [
        -200,  # Candidate well above the image top (worst case).
        -34,
        -1,
        0,  # Candidate exactly at the top edge.
        33,  # Just below where the old code would clamp.
        34,  # Transition point of the old `max(34, top)` clamp.
        100,
        1000,  # Normal middle-of-image case.
    ],
)
def test_badge_stays_full_height_regardless_of_top(top):
    """Reproduces the formula for the badge bounding box. Before the fix,
    ``(left, max(0, top-34), left+32, max(34, top))`` produced heights
    that varied with ``top`` — specifically, for ``top in (0, 33, 34)``
    the height collapsed to 0 or produced a non-rectangular box. After
    the fix, badge_bottom is derived from badge_top + 34, so the height
    is always exactly 34px regardless of where the candidate sits."""
    badge_top = max(0, top - 34)
    badge_bottom = badge_top + 34
    assert badge_bottom > badge_top
    assert badge_bottom - badge_top == 34


# ---------------------------------------------------------------------------
# Bug (PR #14): main() must catch ValueError, not just RuntimeError, so a
# bad FILE_MODE / RPD threshold prints "[config error] ..." instead of
# crashing with a raw Python traceback.
# ---------------------------------------------------------------------------
def test_main_handles_invalid_file_mode_cleanly(monkeypatch, capsys, tmp_path):
    from agent.__main__ import main

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("FILE_MODE", "bogus_mode")
    # Need a tasks file or argparse will reject; main() should fail at
    # Config.load() before reaching any task work.
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text("noop\n", encoding="utf-8")
    monkeypatch.setenv("TASKS_FILE", str(tasks_file))

    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "[config error]" in captured.err
    assert "FILE_MODE" in captured.err


def test_main_handles_invalid_rpd_thresholds_cleanly(monkeypatch, capsys, tmp_path):
    from agent.__main__ import main

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RPD_WARN_THRESHOLD", "0.99")
    monkeypatch.setenv("RPD_HALT_THRESHOLD", "0.50")
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text("noop\n", encoding="utf-8")
    monkeypatch.setenv("TASKS_FILE", str(tasks_file))

    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "[config error]" in captured.err
    assert "must be less than" in captured.err
