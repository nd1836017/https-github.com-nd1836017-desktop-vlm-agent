"""Unit tests for agent.state (checkpoint persistence)."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.state import AgentState, load_state, reset_state, save_state


def test_initial_state_has_zero_completed():
    state = AgentState.initial(Path("tasks.txt"), total_steps=10)
    assert state.last_completed_step == 0
    assert state.total_steps == 10
    assert state.tasks_file == "tasks.txt"
    assert state.version >= 1


def test_advance_increments_counter():
    state = AgentState.initial(Path("tasks.txt"), total_steps=3)
    state = state.advance()
    assert state.last_completed_step == 1
    state = state.advance().advance()
    assert state.last_completed_step == 3


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / ".agent_state.json"
    original = AgentState(
        version=1, tasks_file="t.txt", total_steps=7, last_completed_step=3
    )
    save_state(path, original)
    assert path.exists()
    loaded = load_state(path)
    assert loaded == original


def test_load_nonexistent_returns_none(tmp_path):
    assert load_state(tmp_path / "does-not-exist.json") is None


def test_load_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not valid json {", encoding="utf-8")
    assert load_state(path) is None


def test_load_missing_keys_returns_none(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    assert load_state(path) is None


def test_reset_state_removes_file(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    reset_state(path)
    assert not path.exists()


def test_reset_state_on_missing_file_is_idempotent(tmp_path):
    path = tmp_path / "missing.json"
    reset_state(path)  # must not raise
    assert not path.exists()


def test_save_state_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "state.json"
    state = AgentState.initial(Path("x"), 1)
    save_state(nested, state)
    assert nested.exists()


def test_save_state_is_atomic_no_temp_leak(tmp_path):
    path = tmp_path / "state.json"
    state = AgentState.initial(Path("x"), 1).advance()
    save_state(path, state)
    # Only the final file should exist (no leftover temp files).
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"state.json"}


@pytest.mark.parametrize("bad_value", ['"not a number"', "null"])
def test_load_state_rejects_wrong_types(tmp_path, bad_value):
    path = tmp_path / "x.json"
    path.write_text(
        f'{{"version": {bad_value}, "tasks_file": "t", '
        f'"total_steps": 1, "last_completed_step": 0}}',
        encoding="utf-8",
    )
    assert load_state(path) is None
