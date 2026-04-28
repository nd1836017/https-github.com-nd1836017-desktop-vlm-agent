"""Tests for IF/ELSE/END_IF/WAIT_UNTIL parsing + agent control flow."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from agent.tasks_loader import TasksLoadError, load_steps


def _w(tmp_path: Path, text: str) -> Path:
    """Helper: write `text` to a temp tasks.txt and return the path."""
    p = tmp_path / "tasks.txt"
    p.write_text(text)
    return p


# -----------------------------------------------------------------------------
# IF / ELSE / END_IF parsing


def test_if_else_endif_emits_control_steps_and_branches(tmp_path: Path):
    steps = load_steps(
        _w(
            tmp_path,
            "before\n"
            "IF [signed in] THEN\n"
            "    do A\n"
            "ELSE\n"
            "    do B\n"
            "END_IF\n"
            "after\n",
        )
    )
    kinds = [s.control_kind for s in steps]
    assert kinds == [None, "if_begin", None, "if_else", None, "if_end", None]
    assert [s.branch for s in steps] == [
        None, "then", "then", "else", "else", "else", None
    ]
    # block_id is set on directives; active_block_id is set on every
    # step inside the block including the directives.
    assert steps[1].block_id == 0
    assert steps[3].block_id == 0
    assert steps[5].block_id == 0
    assert all(s.active_block_id == 0 for s in steps[1:6])
    assert steps[0].active_block_id is None and steps[6].active_block_id is None
    assert steps[1].condition_text == "signed in"


def test_if_without_else_is_valid(tmp_path: Path):
    steps = load_steps(
        _w(tmp_path, "IF [signed in] THEN\n  do A\nEND_IF\nafter\n")
    )
    assert [s.control_kind for s in steps] == [
        "if_begin", None, "if_end", None
    ]
    # Without an ELSE, the only branch is "then".
    assert [s.branch for s in steps] == ["then", "then", "then", None]


def test_unmatched_else_raises(tmp_path: Path):
    with pytest.raises(TasksLoadError, match="ELSE"):
        load_steps(_w(tmp_path, "ELSE\n"))


def test_unmatched_end_if_raises(tmp_path: Path):
    with pytest.raises(TasksLoadError, match="END_IF"):
        load_steps(_w(tmp_path, "END_IF\n"))


def test_unclosed_if_raises(tmp_path: Path):
    with pytest.raises(TasksLoadError, match="never closed"):
        load_steps(_w(tmp_path, "IF [x] THEN\nstep\n"))


def test_two_else_raises(tmp_path: Path):
    with pytest.raises(TasksLoadError, match="ELSE"):
        load_steps(
            _w(tmp_path, "IF [x] THEN\nA\nELSE\nB\nELSE\nC\nEND_IF\n")
        )


def test_nested_if_rejected(tmp_path: Path):
    with pytest.raises(TasksLoadError, match="nested IF"):
        load_steps(
            _w(
                tmp_path,
                "IF [a] THEN\n"
                "    IF [b] THEN\n"
                "        x\n"
                "    END_IF\n"
                "END_IF\n",
            )
        )


def test_empty_if_condition_rejected(tmp_path: Path):
    with pytest.raises(TasksLoadError, match="non-empty"):
        load_steps(_w(tmp_path, "IF [] THEN\nstep\nEND_IF\n"))


def test_if_inside_for_each_row_rejected(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n2\n")
    with pytest.raises(TasksLoadError, match="not supported inside"):
        load_steps(
            _w(
                tmp_path,
                "FOR_EACH_ROW [data.csv]\n"
                "IF [x] THEN\n"
                "click {{row.a}}\n"
                "END_IF\n"
                "END_FOR_EACH_ROW\n",
            )
        )


def test_for_each_row_inside_if_inherits_branch(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("name\nfoo\nbar\n")
    steps = load_steps(
        _w(
            tmp_path,
            "IF [signed in] THEN\n"
            "FOR_EACH_ROW [data.csv]\n"
            "click {{row.name}}\n"
            "END_FOR_EACH_ROW\n"
            "END_IF\n",
        )
    )
    # if_begin + 2 expanded loop body steps + if_end.
    assert [s.control_kind for s in steps] == [
        "if_begin", None, None, "if_end"
    ]
    assert [s.branch for s in steps] == ["then", "then", "then", "then"]
    assert [s.text for s in steps[1:3]] == ["click foo", "click bar"]


# -----------------------------------------------------------------------------
# WAIT_UNTIL parsing


def test_wait_until_parses(tmp_path: Path):
    steps = load_steps(_w(tmp_path, "WAIT_UNTIL [Welcome back]\n"))
    assert len(steps) == 1
    assert steps[0].control_kind == "wait_until"
    assert steps[0].condition_text == "Welcome back"


def test_empty_wait_until_rejected(tmp_path: Path):
    with pytest.raises(TasksLoadError, match="non-empty"):
        load_steps(_w(tmp_path, "WAIT_UNTIL []\n"))


def test_wait_until_inside_for_each_rejected(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n")
    with pytest.raises(TasksLoadError, match="not supported inside"):
        load_steps(
            _w(
                tmp_path,
                "FOR_EACH_ROW [data.csv]\n"
                "WAIT_UNTIL [done]\n"
                "END_FOR_EACH_ROW\n",
            )
        )


def test_wait_until_inside_if_branch_inherits(tmp_path: Path):
    steps = load_steps(
        _w(
            tmp_path,
            "IF [x] THEN\n"
            "    WAIT_UNTIL [y]\n"
            "ELSE\n"
            "    do other\n"
            "END_IF\n",
        )
    )
    wait_step = next(s for s in steps if s.control_kind == "wait_until")
    assert wait_step.branch == "then"
    assert wait_step.active_block_id == 0


# -----------------------------------------------------------------------------
# Agent loop control-flow handling


def test_maybe_handle_control_if_true_takes_then(monkeypatch):
    """IF condition True records True and skips ELSE-branch steps."""
    from agent.agent import _maybe_handle_control
    from agent.tasks_loader import TaskStep

    fake_vlm = mock.Mock()
    fake_vlm.check_condition.return_value = True
    fake_geom = mock.Mock()
    monkeypatch.setattr(
        "agent.agent.capture_screenshot",
        lambda: object(),  # any sentinel; vlm is mocked
    )

    decisions: dict[int, bool] = {}
    if_step = TaskStep(
        text="IF [x] THEN",
        control_kind="if_begin",
        block_id=0,
        active_block_id=0,
        branch="then",
        condition_text="x",
    )
    out = _maybe_handle_control(
        task_step=if_step,
        idx=1,
        total_steps=4,
        vlm=fake_vlm,
        geometry=fake_geom,
        branch_decisions=decisions,
        wait_until_timeout_seconds=30,
        wait_until_poll_seconds=2,
    )
    assert out is not None and out.passed
    assert decisions == {0: True}

    # Step in ELSE branch should be skipped (synthetic PASS).
    else_step = TaskStep(
        text="do other",
        active_block_id=0,
        branch="else",
    )
    out = _maybe_handle_control(
        task_step=else_step,
        idx=4,
        total_steps=4,
        vlm=fake_vlm,
        geometry=fake_geom,
        branch_decisions=decisions,
        wait_until_timeout_seconds=30,
        wait_until_poll_seconds=2,
    )
    assert out is not None and out.passed
    assert "skipped" in out.reason

    # Step in THEN branch should NOT be a control match (returns None
    # so the run loop keeps going to the planner path).
    then_step = TaskStep(
        text="do A",
        active_block_id=0,
        branch="then",
    )
    out = _maybe_handle_control(
        task_step=then_step,
        idx=2,
        total_steps=4,
        vlm=fake_vlm,
        geometry=fake_geom,
        branch_decisions=decisions,
        wait_until_timeout_seconds=30,
        wait_until_poll_seconds=2,
    )
    assert out is None


def test_maybe_handle_control_wait_until_passes_when_condition_seen(monkeypatch):
    """WAIT_UNTIL returns PASS as soon as check_condition is True."""
    from agent.agent import _maybe_handle_control
    from agent.tasks_loader import TaskStep

    fake_vlm = mock.Mock()
    # Fail twice, then pass.
    fake_vlm.check_condition.side_effect = [False, False, True]
    monkeypatch.setattr("agent.agent.capture_screenshot", lambda: object())
    monkeypatch.setattr("time.sleep", lambda _: None)  # skip real waits

    step = TaskStep(
        text="WAIT_UNTIL [Welcome]",
        control_kind="wait_until",
        condition_text="Welcome",
    )
    out = _maybe_handle_control(
        task_step=step,
        idx=1,
        total_steps=1,
        vlm=fake_vlm,
        geometry=mock.Mock(),
        branch_decisions={},
        wait_until_timeout_seconds=30,
        wait_until_poll_seconds=0,
    )
    assert out is not None and out.passed
    assert "Welcome" in out.reason


def test_wait_until_inside_non_taken_branch_is_skipped(monkeypatch):
    """Regression: WAIT_UNTIL in a non-taken branch must be skipped.

    Before the fix, the kind=="wait_until" path entered the polling
    loop unconditionally, which (a) burned VLM quota for a directive
    the user never intended to reach, and (b) could halt the run on
    timeout for a branch that wasn't even taken.
    """
    from agent.agent import _maybe_handle_control
    from agent.tasks_loader import TaskStep

    fake_vlm = mock.Mock()
    # If the bug regresses, this gets called and the test fails.
    fake_vlm.check_condition.side_effect = AssertionError(
        "WAIT_UNTIL inside non-taken branch must NOT poll the screen"
    )
    monkeypatch.setattr("agent.agent.capture_screenshot", lambda: object())

    decisions: dict[int, bool] = {0: True}  # THEN branch was taken

    wait_step = TaskStep(
        text="WAIT_UNTIL [Sign in form]",
        control_kind="wait_until",
        condition_text="Sign in form",
        active_block_id=0,
        branch="else",  # WAIT_UNTIL is in the ELSE branch
    )
    out = _maybe_handle_control(
        task_step=wait_step,
        idx=4,
        total_steps=6,
        vlm=fake_vlm,
        geometry=mock.Mock(),
        branch_decisions=decisions,
        wait_until_timeout_seconds=30,
        wait_until_poll_seconds=0,
    )
    assert out is not None and out.passed
    assert "skipped" in out.reason
    assert "else" in out.reason
    fake_vlm.check_condition.assert_not_called()


def test_maybe_handle_control_wait_until_times_out(monkeypatch):
    """WAIT_UNTIL returns FAIL when condition stays False past the budget."""
    from agent.agent import _maybe_handle_control
    from agent.tasks_loader import TaskStep

    fake_vlm = mock.Mock()
    fake_vlm.check_condition.return_value = False  # never satisfies
    monkeypatch.setattr("agent.agent.capture_screenshot", lambda: object())
    monkeypatch.setattr("time.sleep", lambda _: None)

    # Use a tiny timeout (in monotonic time) — pin time.monotonic so it
    # crosses the deadline after exactly one attempt.
    fake_clock = iter([0.0, 0.0, 5.0])

    def _mono():
        try:
            return next(fake_clock)
        except StopIteration:
            return 99.0

    monkeypatch.setattr("time.monotonic", _mono)

    step = TaskStep(
        text="WAIT_UNTIL [Welcome]",
        control_kind="wait_until",
        condition_text="Welcome",
    )
    out = _maybe_handle_control(
        task_step=step,
        idx=1,
        total_steps=1,
        vlm=fake_vlm,
        geometry=mock.Mock(),
        branch_decisions={},
        wait_until_timeout_seconds=1.0,  # < 5.0
        wait_until_poll_seconds=0,
    )
    assert out is not None and not out.passed
    assert "timed out" in out.reason
