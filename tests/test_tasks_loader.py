"""Unit tests for agent.tasks_loader (FOR_EACH_ROW + CSV templating)."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.tasks_loader import TasksLoadError, load_tasks


def _write(tmp: Path, name: str, body: str) -> Path:
    path = tmp / name
    path.write_text(body, encoding="utf-8")
    return path


# --- Plain (non-loop) parsing keeps the legacy behavior. ----------------------


def test_plain_tasks_file_passes_through_unchanged(tmp_path: Path) -> None:
    tasks = _write(
        tmp_path,
        "tasks.txt",
        "# header comment\n"
        "Open Chrome\n"
        "\n"
        "Click the address bar\n"
        "  Type \"hello\"  \n",
    )
    assert load_tasks(tasks) == [
        "Open Chrome",
        "Click the address bar",
        'Type "hello"',
    ]


def test_blank_and_comment_only_file_returns_empty_list(tmp_path: Path) -> None:
    tasks = _write(tmp_path, "t.txt", "# only comments\n\n   \n")
    assert load_tasks(tasks) == []


def test_load_tasks_raises_filenotfound_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "missing.txt")


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    tasks = tmp_path / "bom.txt"
    tasks.write_bytes(b"\xef\xbb\xbfFirst step\nSecond step\n")
    assert load_tasks(tasks) == ["First step", "Second step"]


# --- FOR_EACH_ROW basic expansion. --------------------------------------------


def _make_csv_pair(tmp_path: Path, csv_body: str, tasks_body: str) -> Path:
    _write(tmp_path, "data.csv", csv_body)
    return _write(tmp_path, "tasks.txt", tasks_body)


def test_for_each_row_expands_per_row_with_substitution(tmp_path: Path) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "name,email\nAlice,a@example.com\nBob,b@example.com\n",
        "Open form\n"
        "FOR_EACH_ROW [data.csv]\n"
        "    Type \"{{row.name}}\"\n"
        "    Type \"{{row.email}}\"\n"
        "END_FOR_EACH_ROW\n"
        "Click submit\n",
    )
    assert load_tasks(tasks) == [
        "Open form",
        'Type "Alice"',
        'Type "a@example.com"',
        'Type "Bob"',
        'Type "b@example.com"',
        "Click submit",
    ]


def test_for_each_row_resolves_csv_relative_to_tasks_file(tmp_path: Path) -> None:
    nested = tmp_path / "deep"
    nested.mkdir()
    _write(nested, "data.csv", "x\n1\n2\n")
    tasks = _write(
        nested,
        "tasks.txt",
        "FOR_EACH_ROW [data.csv]\n"
        "  Use {{row.x}}\n"
        "END_FOR_EACH_ROW\n",
    )
    # Run from a totally unrelated working directory; relative resolution
    # must be against tasks.txt's directory.
    assert load_tasks(tasks) == ["Use 1", "Use 2"]


def test_for_each_row_csv_override(tmp_path: Path) -> None:
    _write(tmp_path, "demo.csv", "x\nA\n")  # would normally be picked
    other = _write(tmp_path, "real.csv", "x\nB\nC\n")
    tasks = _write(
        tmp_path,
        "tasks.txt",
        "FOR_EACH_ROW [demo.csv]\n"
        "  Use {{row.x}}\n"
        "END_FOR_EACH_ROW\n",
    )
    assert load_tasks(tasks, csv_override=other) == ["Use B", "Use C"]


def test_csv_override_relative_resolves_against_cwd_not_tasks_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative ``--csv`` path is resolved against the user's CWD.

    This is what users intuitively expect for one-off CLI overrides like
    ``python -m agent --csv my_data.csv`` — they don't want it resolved
    against the tasks file's parent directory (which may live in a
    completely different tree).
    """
    tasks_dir = tmp_path / "workflows"
    tasks_dir.mkdir()
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()

    # CSV in the tasks dir would shadow the right one if resolution were
    # tasks-dir-relative — populate it with WRONG data so an incorrect fix
    # would surface in the assertion.
    _write(tasks_dir, "real.csv", "x\nWRONG\n")
    # The CSV the user actually meant lives in their CWD.
    _write(cwd_dir, "real.csv", "x\nRIGHT\n")
    # Tasks file references some unrelated demo csv that we'll override.
    _write(tasks_dir, "demo.csv", "x\nIGNORED\n")
    tasks = _write(
        tasks_dir,
        "tasks.txt",
        "FOR_EACH_ROW [demo.csv]\n  Use {{row.x}}\nEND_FOR_EACH_ROW\n",
    )

    monkeypatch.chdir(cwd_dir)
    assert load_tasks(tasks, csv_override=Path("real.csv")) == ["Use RIGHT"]


def test_csv_override_absolute_path_is_respected(tmp_path: Path) -> None:
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    other = _write(other_dir, "real.csv", "x\nABS\n")
    tasks = _make_csv_pair(
        tmp_path,
        "x\nIGNORED\n",
        "FOR_EACH_ROW [data.csv]\n  Use {{row.x}}\nEND_FOR_EACH_ROW\n",
    )
    assert load_tasks(tasks, csv_override=other) == ["Use ABS"]


def test_default_value_is_used_when_cell_is_empty(tmp_path: Path) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "name,note\nAlice,\nBob,actual\n",
        "FOR_EACH_ROW [data.csv]\n"
        "  Type \"{{row.note|Hello!}}\"\n"
        "END_FOR_EACH_ROW\n",
    )
    assert load_tasks(tasks) == ['Type "Hello!"', 'Type "actual"']


def test_default_with_internal_whitespace_is_stripped(tmp_path: Path) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "x,y\n,present\n",  # x is empty, y is present
        "FOR_EACH_ROW [data.csv]\n"
        "  Use {{ row.x | fallback value }} {{row.y}}\n"
        "END_FOR_EACH_ROW\n",
    )
    assert load_tasks(tasks) == ["Use fallback value present"]


def test_quoted_csv_cells_are_supported(tmp_path: Path) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        'name,note\nAlice,"hello, world"\n',
        "FOR_EACH_ROW [data.csv]\n"
        "  Type \"{{row.note}}\"\n"
        "END_FOR_EACH_ROW\n",
    )
    assert load_tasks(tasks) == ['Type "hello, world"']


def test_blank_lines_and_comments_inside_block_are_skipped(tmp_path: Path) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "x\nA\n",
        "FOR_EACH_ROW [data.csv]\n"
        "\n"
        "  # ignored\n"
        "  Use {{row.x}}\n"
        "END_FOR_EACH_ROW\n",
    )
    assert load_tasks(tasks) == ["Use A"]


def test_empty_csv_is_warned_and_loop_expanded_to_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "x\n",  # header only, no rows
        "Before\n"
        "FOR_EACH_ROW [data.csv]\n"
        "  Use {{row.x}}\n"
        "END_FOR_EACH_ROW\n"
        "After\n",
    )
    with caplog.at_level("WARNING", logger="agent.tasks_loader"):
        assert load_tasks(tasks) == ["Before", "After"]
    assert any("zero rows" in rec.message for rec in caplog.records)


def test_empty_block_body_is_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "x\nA\n",
        "FOR_EACH_ROW [data.csv]\n"
        "  # only a comment\n"
        "END_FOR_EACH_ROW\n",
    )
    with caplog.at_level("WARNING", logger="agent.tasks_loader"):
        assert load_tasks(tasks) == []
    assert any("empty body" in rec.message for rec in caplog.records)


# --- Error handling. ----------------------------------------------------------


def test_unknown_field_raises_with_available_fields_listed(tmp_path: Path) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "name,email\nAlice,a@example.com\n",
        "FOR_EACH_ROW [data.csv]\n"
        "  Type \"{{row.first_name}}\"\n"
        "END_FOR_EACH_ROW\n",
    )
    with pytest.raises(TasksLoadError) as exc:
        load_tasks(tasks)
    msg = str(exc.value)
    assert "first_name" in msg
    assert "name" in msg and "email" in msg


def test_for_each_without_end_raises(tmp_path: Path) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "x\nA\n",
        "FOR_EACH_ROW [data.csv]\n"
        "  Use {{row.x}}\n",
    )
    with pytest.raises(TasksLoadError, match="never closed"):
        load_tasks(tasks)


def test_end_for_each_without_for_each_raises(tmp_path: Path) -> None:
    tasks = _write(tmp_path, "tasks.txt", "Foo\nEND_FOR_EACH_ROW\n")
    with pytest.raises(TasksLoadError, match="without a matching FOR_EACH_ROW"):
        load_tasks(tasks)


def test_nested_for_each_raises(tmp_path: Path) -> None:
    _write(tmp_path, "outer.csv", "x\nA\n")
    _write(tmp_path, "inner.csv", "y\nB\n")
    tasks = _write(
        tmp_path,
        "tasks.txt",
        "FOR_EACH_ROW [outer.csv]\n"
        "  FOR_EACH_ROW [inner.csv]\n"
        "    Use {{row.y}}\n"
        "  END_FOR_EACH_ROW\n"
        "END_FOR_EACH_ROW\n",
    )
    with pytest.raises(TasksLoadError, match="nested FOR_EACH_ROW"):
        load_tasks(tasks)


def test_for_each_with_no_csv_path_raises(tmp_path: Path) -> None:
    tasks = _write(
        tmp_path,
        "tasks.txt",
        "FOR_EACH_ROW [   ]\n"
        "  Use {{row.x}}\n"
        "END_FOR_EACH_ROW\n",
    )
    with pytest.raises(TasksLoadError, match="requires a CSV path"):
        load_tasks(tasks)


def test_missing_csv_file_raises_clearly(tmp_path: Path) -> None:
    tasks = _write(
        tmp_path,
        "tasks.txt",
        "FOR_EACH_ROW [does_not_exist.csv]\n"
        "  Use {{row.x}}\n"
        "END_FOR_EACH_ROW\n",
    )
    with pytest.raises(TasksLoadError, match="does not exist"):
        load_tasks(tasks)


def test_placeholder_outside_block_raises(tmp_path: Path) -> None:
    tasks = _write(
        tmp_path,
        "tasks.txt",
        "Type \"{{row.email}}\"\n",
    )
    with pytest.raises(TasksLoadError, match="outside a FOR_EACH_ROW block"):
        load_tasks(tasks)


def test_csv_without_header_raises(tmp_path: Path) -> None:
    _write(tmp_path, "data.csv", "")  # totally empty
    tasks = _write(
        tmp_path,
        "tasks.txt",
        "FOR_EACH_ROW [data.csv]\n"
        "  Use {{row.x}}\n"
        "END_FOR_EACH_ROW\n",
    )
    with pytest.raises(TasksLoadError, match="no header row"):
        load_tasks(tasks)


def test_error_message_includes_lineno_and_csv_name(tmp_path: Path) -> None:
    _write(tmp_path, "data.csv", "name\nAlice\n")
    tasks = _write(
        tmp_path,
        "tasks.txt",
        "FOR_EACH_ROW [data.csv]\n"
        "  Type \"{{row.email}}\"\n"
        "END_FOR_EACH_ROW\n",
    )
    with pytest.raises(TasksLoadError) as exc:
        load_tasks(tasks)
    msg = str(exc.value)
    assert "Line 2" in msg
    assert "data.csv" in msg


# --- Case-insensitivity of FOR_EACH_ROW / END_FOR_EACH_ROW directives. --------


def test_directives_are_case_insensitive(tmp_path: Path) -> None:
    tasks = _make_csv_pair(
        tmp_path,
        "x\nA\n",
        "for_each_row [data.csv]\n"
        "  Use {{row.x}}\n"
        "End_For_Each_Row\n",
    )
    assert load_tasks(tasks) == ["Use A"]


# --- Bundled example file is sane (smoke test). -------------------------------


def test_bundled_example_loads_without_errors(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_tasks = repo_root / "examples" / "tasks_csv_demo.txt"
    src_csv = repo_root / "examples" / "data_demo.csv"
    assert src_tasks.exists() and src_csv.exists()
    # Copy to tmp so the relative CSV path is preserved.
    tasks = _write(tmp_path, "tasks_csv_demo.txt", src_tasks.read_text())
    _write(tmp_path, "data_demo.csv", src_csv.read_text())
    expanded = load_tasks(tasks)
    # 3 demo rows × 10 lines per block + 3 lines before + 2 lines after = 35.
    assert len(expanded) == 3 * 10 + 3 + 2
    # Bob has empty message → default substitution should kick in.
    assert any('Type "Hello!"' in step for step in expanded)
