"""Tests for ``agent.files``: workspace, modes, naming, feature inspection."""
from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.files import (
    FileMode,
    FileWorkspace,
    RunFeatures,
    execute_attach_file,
    execute_capture_for_ai,
    execute_download,
    format_features_summary,
    inspect_features,
    name_with_row_suffix,
    resolve_mode,
    sanitize_filename,
)
from agent.parser import (
    AttachFileCommand,
    CaptureForAiCommand,
    DownloadCommand,
)
from agent.tasks_loader import TaskStep

# --------------------------------------------------------------- sanitize_filename


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("invoice.pdf", "invoice.pdf"),
        ("a/b/c.txt", "a_b_c.txt"),
        ("..\\evil\\..\\foo.bin", "_evil_.._foo.bin"),
        ('weird:"<>|*?.txt', "weird_.txt"),
        ("", "download.bin"),
        ("   ", "download.bin"),
        ("...", "download.bin"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_strips_path_separators() -> None:
    # No remaining slashes / backslashes — no directory traversal possible.
    out = sanitize_filename("../../etc/passwd")
    assert "/" not in out
    assert "\\" not in out


def test_sanitize_filename_caps_length() -> None:
    long_name = "a" * 300 + ".pdf"
    out = sanitize_filename(long_name)
    assert out.endswith(".pdf")
    assert len(out) <= 200


# ------------------------------------------------------------ name_with_row_suffix


def test_name_with_row_suffix_inserts_row_marker() -> None:
    # User's exact example from the spec.
    assert (
        name_with_row_suffix("ND 00002348.pdf", row_index=50)
        == "ND 00002348(row50).pdf"
    )


def test_name_with_row_suffix_no_row_passthrough() -> None:
    assert name_with_row_suffix("invoice.pdf", row_index=None) == "invoice.pdf"


def test_name_with_row_suffix_no_extension() -> None:
    assert name_with_row_suffix("README", row_index=3) == "README(row3)"


# ------------------------------------------------------------------ FileWorkspace


def test_temp_workspace_creates_dir_and_wipes_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ws = FileWorkspace.create(mode=FileMode.TEMP)
    assert ws.root is not None
    assert ws.root.is_dir()

    saved = ws.save(content=b"hello", suggested_name="greet.txt")
    assert saved is not None
    assert saved.read_bytes() == b"hello"
    root = ws.root

    ws.finalize(success=True)
    assert not root.exists()


def test_temp_workspace_preserves_dir_on_failure(tmp_path: Path) -> None:
    ws = FileWorkspace.create(mode=FileMode.TEMP)
    saved = ws.save(content=b"hi", suggested_name="x.bin")
    assert saved is not None
    root = ws.root
    ws.finalize(success=False)
    assert root.exists(), "temp dir should be kept on failure for debugging"
    # cleanup so we don't leak on the test runner
    import shutil

    shutil.rmtree(root, ignore_errors=True)


def test_save_workspace_persists_to_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "out"
    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=workdir)
    ws.begin_step(row_index=7, csv_name="data.csv")
    saved = ws.save(content=b"row7", suggested_name="invoice ND 00002348.pdf")
    assert saved is not None
    assert saved.name == "invoice ND 00002348(row7).pdf"
    assert saved.read_bytes() == b"row7"
    ws.finalize(success=True)
    # Save mode never deletes.
    assert saved.exists()


def test_feed_workspace_never_writes_disk() -> None:
    ws = FileWorkspace.create(mode=FileMode.FEED)
    saved = ws.save(content=b"abc", suggested_name="x.bin")
    assert saved is None  # FEED never returns a path
    assert ws.consume_feed() == [b"abc"]
    # Buffer is drained after consume.
    assert ws.consume_feed() == []
    ws.finalize(success=True)


def test_save_into_temp_collision_appends_suffix(tmp_path: Path) -> None:
    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=tmp_path / "out")
    a = ws.save(content=b"v1", suggested_name="report.pdf")
    b = ws.save(content=b"v2", suggested_name="report.pdf")
    assert a is not None and b is not None
    assert a.name == "report.pdf"
    assert b.name != a.name
    assert a.read_bytes() == b"v1"
    assert b.read_bytes() == b"v2"


# ------------------------------------------------------------------- resolve_mode


def test_resolve_mode_cli_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, _ = resolve_mode(
        cli_mode=FileMode.SAVE,
        cli_workdir=None,
        env_mode=FileMode.FEED,
        env_workdir=None,
        interactive=True,
    )
    assert mode is FileMode.SAVE


def test_resolve_mode_env_when_no_cli() -> None:
    mode, _ = resolve_mode(
        cli_mode=None,
        cli_workdir=None,
        env_mode=FileMode.FEED,
        env_workdir=None,
        interactive=True,
    )
    assert mode is FileMode.FEED


def test_resolve_mode_falls_back_to_temp_without_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pytest's stdin is not a tty so the interactive prompt is skipped.
    mode, workdir = resolve_mode(
        cli_mode=None,
        cli_workdir=None,
        env_mode=None,
        env_workdir=None,
        interactive=True,
    )
    assert mode is FileMode.TEMP
    assert workdir is None


# ----------------------------------------------------------- file-command runners


def test_execute_download_writes_via_workspace(tmp_path: Path) -> None:
    fake_response = MagicMock()
    fake_response.read.return_value = b"PDFBYTES"
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False

    def opener(req, timeout):
        return fake_response

    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=tmp_path)
    ok, action = execute_download(
        DownloadCommand(url="https://example.com/foo.pdf", filename="foo.pdf"),
        ws,
        opener=opener,
    )
    assert ok is True
    assert "foo.pdf" in action
    out = list(tmp_path.glob("*.pdf"))
    assert len(out) == 1
    assert out[0].read_bytes() == b"PDFBYTES"


def test_execute_download_rejects_non_http_urls(tmp_path: Path) -> None:
    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=tmp_path)
    ok, action = execute_download(
        DownloadCommand(url="ftp://example.com/x", filename="x.bin"),
        ws,
    )
    assert ok is False
    assert "scheme" in action.lower()


def test_execute_download_reports_url_errors(tmp_path: Path) -> None:
    def opener(req, timeout):
        raise urllib.error.URLError("boom")

    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=tmp_path)
    ok, action = execute_download(
        DownloadCommand(url="https://example.com/x", filename="x.bin"),
        ws,
        opener=opener,
    )
    assert ok is False
    assert "DOWNLOAD failed" in action


def test_execute_attach_file_uses_workspace_files(tmp_path: Path) -> None:
    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=tmp_path)
    target = tmp_path / "spec.pdf"
    target.write_bytes(b"x")

    fake_pyautogui = MagicMock()
    ok, action = execute_attach_file(
        AttachFileCommand(filename="spec.pdf"),
        ws,
        pyautogui_module=fake_pyautogui,
        sleep=lambda _s: None,
    )
    assert ok is True
    assert str(target) in action
    fake_pyautogui.hotkey.assert_called_with("ctrl", "l")
    fake_pyautogui.typewrite.assert_called_once()
    fake_pyautogui.press.assert_called_with("enter")


def test_execute_attach_file_falls_back_to_disk_path(tmp_path: Path) -> None:
    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=tmp_path / "ws")
    on_disk = tmp_path / "elsewhere.txt"
    on_disk.write_text("x")
    fake_pyautogui = MagicMock()
    ok, action = execute_attach_file(
        AttachFileCommand(filename=str(on_disk)),
        ws,
        pyautogui_module=fake_pyautogui,
        sleep=lambda _s: None,
    )
    assert ok is True
    assert str(on_disk) in action


def test_execute_attach_file_missing_file_fails(tmp_path: Path) -> None:
    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=tmp_path)
    ok, action = execute_attach_file(
        AttachFileCommand(filename="does_not_exist.pdf"),
        ws,
        pyautogui_module=MagicMock(),
        sleep=lambda _s: None,
    )
    assert ok is False
    assert "not found" in action


def test_capture_for_ai_uses_current_screenshot() -> None:
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    ws = FileWorkspace.create(mode=FileMode.FEED)
    ok, _ = execute_capture_for_ai(
        CaptureForAiCommand(filename=""),
        ws,
        screenshot=img,
    )
    assert ok is True
    assert len(ws.consume_feed()) == 1


def test_capture_for_ai_reads_named_file(tmp_path: Path) -> None:
    ws = FileWorkspace.create(mode=FileMode.SAVE, workdir=tmp_path)
    target = tmp_path / "snap.png"
    target.write_bytes(b"\x89PNG...")
    ok, action = execute_capture_for_ai(
        CaptureForAiCommand(filename="snap.png"),
        ws,
    )
    assert ok is True
    assert "snap.png" in action
    assert ws.consume_feed() == [b"\x89PNG..."]


# ----------------------------------------------------------- feature inspection


def test_inspect_features_detects_csv_loop() -> None:
    steps = [
        TaskStep(text="Open the form"),
        TaskStep(text="Fill name", row_index=1, csv_name="data.csv"),
        TaskStep(text="Fill email", row_index=1, csv_name="data.csv"),
        TaskStep(text="Fill name", row_index=2, csv_name="data.csv"),
        TaskStep(text="Fill email", row_index=2, csv_name="data.csv"),
    ]
    f = inspect_features(steps)
    assert f.uses_csv_loop is True
    assert f.csv_row_count == 2
    assert f.csv_files == ("data.csv",)
    assert f.uses_files is False


def test_inspect_features_detects_file_primitives() -> None:
    steps = [
        TaskStep(text="DOWNLOAD [https://x/foo.pdf, foo.pdf]"),
        TaskStep(text="DOWNLOAD [https://y/bar.pdf]"),
        TaskStep(text="ATTACH_FILE [foo.pdf]"),
        TaskStep(text="CAPTURE_FOR_AI"),
    ]
    f = inspect_features(steps)
    assert f.uses_downloads is True
    assert f.download_count == 2
    assert f.uses_attach_file is True
    assert f.attach_file_count == 1
    assert f.uses_capture_for_ai is True
    assert f.capture_for_ai_count == 1
    assert f.uses_files is True


def test_inspect_features_simple_task_has_nothing() -> None:
    steps = [
        TaskStep(text="Press Win"),
        TaskStep(text="Type Notepad"),
        TaskStep(text="Press Enter"),
    ]
    f = inspect_features(steps)
    assert not f.uses_csv_loop
    assert not f.uses_files
    summary = format_features_summary(f, total_steps=3)
    assert "no special features" in summary


def test_format_features_summary_lists_csv_and_files() -> None:
    f = RunFeatures(
        uses_csv_loop=True,
        csv_row_count=50,
        csv_files=("data.csv",),
        uses_downloads=True,
        download_count=50,
        uses_attach_file=True,
        attach_file_count=50,
    )
    summary = format_features_summary(f, total_steps=152)
    assert "152" in summary
    assert "FOR_EACH_ROW" in summary
    assert "DOWNLOAD" in summary
    assert "ATTACH_FILE" in summary
