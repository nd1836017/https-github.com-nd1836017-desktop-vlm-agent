"""Unit tests for agent.history."""
from __future__ import annotations

import pytest

from agent.history import History, render_command
from agent.parser import (
    ClickCommand,
    PressCommand,
    RecallCommand,
    RememberCommand,
    TypeCommand,
)


def test_render_command_forms():
    assert render_command(ClickCommand(x=10, y=20)) == "CLICK [10,20]"
    assert render_command(PressCommand(key="ctrl+l")) == "PRESS [ctrl+l]"
    assert render_command(TypeCommand(text="hello world")) == "TYPE [hello world]"


def test_history_empty_summary():
    h = History(window=5)
    assert h.summary() == ""
    assert len(h) == 0


def test_history_records_and_renders():
    h = History(window=5)
    h.record("step A", "CLICK [1,2]", passed=True, reason="VERDICT: PASS — ok")
    h.record("step B", "TYPE [hi]", passed=False, reason="VERDICT: FAIL — wrong app")
    summary = h.summary()
    assert "step A" in summary
    assert "CLICK [1,2]" in summary
    assert "PASS" in summary
    assert "step B" in summary
    assert "FAIL" in summary


def test_history_window_evicts_oldest():
    h = History(window=2)
    h.record("a", "CLICK [1,1]", True, "PASS")
    h.record("b", "CLICK [2,2]", True, "PASS")
    h.record("c", "CLICK [3,3]", True, "PASS")
    assert len(h) == 2
    summary = h.summary()
    assert "a" not in summary.split("\n")[0].split("=")[1] or "step='a'" not in summary
    # More precise: 'a' should have been evicted
    assert "step='a'" not in summary
    assert "step='b'" in summary
    assert "step='c'" in summary


def test_history_window_zero_disables():
    h = History(window=0)
    h.record("a", "CLICK [1,1]", True, "PASS")
    assert len(h) == 0
    assert h.summary() == ""


def test_history_window_negative_rejected():
    with pytest.raises(ValueError):
        History(window=-1)


def test_history_truncates_long_reasons():
    h = History(window=1)
    long_reason = "x" * 500
    h.record("a", "CLICK [0,0]", False, long_reason)
    summary = h.summary()
    # Reason is truncated to 120 chars in the rendered summary.
    assert "x" * 120 in summary
    assert "x" * 200 not in summary


# --- REMEMBER / RECALL rendering (regression: dataclass repr leak) ---


def test_render_command_remember_from_screen():
    cmd = RememberCommand(name="order_id", from_screen=True)
    assert render_command(cmd) == "REMEMBER [order_id]"


def test_render_command_remember_literal():
    cmd = RememberCommand(name="user", literal_value="alice")
    assert render_command(cmd) == "REMEMBER [user = alice]"


def test_render_command_remember_literal_redacted():
    cmd = RememberCommand(name="password", literal_value="hunter2")
    rendered = render_command(cmd, redact_type=True)
    # Don't leak the literal; do report the length so postmortems show
    # the field was set.
    assert "hunter2" not in rendered
    assert rendered == "REMEMBER [password = <REDACTED, 7 chars>]"


def test_render_command_recall():
    cmd = RecallCommand(name="order_id")
    assert render_command(cmd) == "RECALL [order_id]"


def test_render_command_remember_does_not_fall_through_to_repr():
    """Regression: history.render_command used to lack RememberCommand /
    RecallCommand handlers, so the fallback ``str(cmd)`` rendered the
    raw dataclass repr (``RememberCommand(kind='REMEMBER', name='x', ...)``)
    into the planner prompt. That's verbose, leaks internal field names,
    and breaks the canonical-form contract every other command obeys.
    """
    cmd = RememberCommand(name="x", from_screen=True)
    rendered = render_command(cmd)
    assert "RememberCommand(" not in rendered
    assert "kind=" not in rendered
    assert rendered.startswith("REMEMBER ")
