"""File-handling workspace and primitive executors.

Three modes determine where files captured during a run live:

- ``TEMP`` — files go to an OS-temp run directory. Cleaned up automatically
  when the run succeeds; **kept** when the run fails or is aborted, so the
  user can inspect them. Default mode for unattended runs.
- ``SAVE`` — files persist in a user-supplied directory. Never cleaned up.
- ``FEED`` — files are not written to disk at all. Bytes stay in memory and
  are attached to the next VLM ``plan_action`` call so the model can "see"
  them. The buffer is consumed by exactly one plan call and then cleared.

When a step is part of a ``FOR_EACH_ROW`` loop, downloaded files get a
``(row<N>)`` suffix so the same file name across rows doesn't collide:

    invoice_ND12345.pdf -> invoice_ND12345(row7).pdf

The ``FileWorkspace`` is the single coordination point. The agent calls
``begin_step(step)`` before each step (so the workspace knows which row
we're on) and ``finalize(success=...)`` exactly once at the end of the run.
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import guarded for type hints only.
    from PIL.Image import Image

    from .parser import (
        AttachFileCommand,
        CaptureForAiCommand,
        DownloadCommand,
    )

log = logging.getLogger(__name__)


class FileMode(str, Enum):
    """How files captured during a run should be persisted."""

    TEMP = "temp"
    SAVE = "save"
    FEED = "feed"


# Maximum length we'll allow on a sanitized download filename. Some OSes /
# filesystems have surprising limits — keep things conservative.
_MAX_FILENAME_LEN = 200

# Filenames may not contain any of these characters on Windows. NUL is also
# forbidden universally. We also strip leading/trailing dots and spaces.
_FORBIDDEN_FILENAME_CHARS = re.compile(r'[\x00<>:"/\\|?*]+')


def sanitize_filename(name: str, *, fallback: str = "download.bin") -> str:
    """Return a safe filename derived from ``name``.

    Drops directory separators, control chars, and reserved characters.
    Empty / dot-only / unreasonable inputs fall back to ``fallback``.
    """
    candidate = name.strip().replace("\r", "").replace("\n", "")
    candidate = _FORBIDDEN_FILENAME_CHARS.sub("_", candidate)
    candidate = candidate.strip(" .")
    if not candidate or candidate in {".", ".."}:
        return fallback
    if len(candidate) > _MAX_FILENAME_LEN:
        # Keep extension if any.
        stem, dot, ext = candidate.rpartition(".")
        if dot and len(ext) <= 16:
            keep = _MAX_FILENAME_LEN - len(ext) - 1
            candidate = stem[:keep] + dot + ext
        else:
            candidate = candidate[:_MAX_FILENAME_LEN]
    return candidate


def _row_suffix(row_index: int | None) -> str:
    """Return ``(rowN)`` when inside a FOR_EACH_ROW loop, otherwise empty."""
    if row_index is None:
        return ""
    return f"(row{row_index})"


def name_with_row_suffix(name: str, row_index: int | None) -> str:
    """Insert the ``(rowN)`` suffix before the extension.

    >>> name_with_row_suffix("invoice_ND12345.pdf", 7)
    'invoice_ND12345(row7).pdf'
    >>> name_with_row_suffix("plain.txt", None)
    'plain.txt'
    >>> name_with_row_suffix("noext", 3)
    'noext(row3)'
    """
    suffix = _row_suffix(row_index)
    if not suffix:
        return name
    stem, dot, ext = name.rpartition(".")
    if dot and ext and len(ext) <= 16:  # plausible extension
        return f"{stem}{suffix}.{ext}"
    return f"{name}{suffix}"


@dataclass
class FileWorkspace:
    """Coordinates where DOWNLOAD/CAPTURE files go for the current run.

    Fields are all keyword-only via ``create()``; do not construct directly.
    """

    mode: FileMode
    root: Path | None  # None when mode == FEED (no on-disk root)
    run_id: str
    _row_index: int | None = field(default=None)
    _csv_name: str | None = field(default=None)
    _feed_buffer: list[bytes] = field(default_factory=list)
    _saved_paths: list[Path] = field(default_factory=list)
    _is_temp_root: bool = field(default=False)

    # ------------------------------------------------------------------ create
    @classmethod
    def create(
        cls,
        *,
        mode: FileMode,
        workdir: Path | None = None,
        run_id: str | None = None,
    ) -> FileWorkspace:
        """Build a workspace for a run.

        - ``TEMP``: ignores ``workdir``; a fresh OS-temp subdir is created.
        - ``SAVE``: ``workdir`` must be provided; created if missing.
        - ``FEED``: no on-disk root.
        """
        rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if mode is FileMode.FEED:
            return cls(mode=mode, root=None, run_id=rid)

        if mode is FileMode.TEMP:
            base = Path(tempfile.mkdtemp(prefix=f"agent_run_{rid}_"))
            return cls(mode=mode, root=base, run_id=rid, _is_temp_root=True)

        # SAVE mode.
        if workdir is None:
            raise ValueError("FileMode.SAVE requires a workdir.")
        target = workdir.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        return cls(mode=mode, root=target, run_id=rid)

    # ------------------------------------------------------------- step hooks
    def begin_step(
        self,
        *,
        row_index: int | None,
        csv_name: str | None,
    ) -> None:
        """Update the workspace's current-row context before each step."""
        self._row_index = row_index
        self._csv_name = csv_name

    @property
    def current_row(self) -> int | None:
        return self._row_index

    # -------------------------------------------------------------------- save
    def save(self, *, content: bytes, suggested_name: str) -> Path | None:
        """Persist ``content`` and return its path (None for FEED mode).

        Filename gets a ``(rowN)`` suffix if we're inside a FOR_EACH_ROW.
        Conflicts are resolved by appending ``-N`` before the extension.
        """
        clean = sanitize_filename(suggested_name)
        clean = name_with_row_suffix(clean, self._row_index)

        if self.mode is FileMode.FEED:
            self._feed_buffer.append(content)
            log.info(
                "FEED: buffered %d bytes (would have been %s)", len(content), clean
            )
            return None

        assert self.root is not None  # narrowed by the mode check
        target = self._unique_path(self.root / clean)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self._saved_paths.append(target)
        log.info("Saved %d bytes to %s", len(content), target)
        return target

    def push_feed(self, content: bytes) -> None:
        """Push raw bytes into the feed buffer regardless of current mode.

        Used by ``CAPTURE_FOR_AI``, which always feeds the VLM directly
        (no disk write) — even when the global mode is TEMP or SAVE.
        """
        self._feed_buffer.append(content)
        log.info("CAPTURE_FOR_AI: buffered %d bytes for next plan call", len(content))

    def consume_feed(self) -> list[bytes]:
        """Return and clear any pending feed-buffer contents."""
        if not self._feed_buffer:
            return []
        out = list(self._feed_buffer)
        self._feed_buffer.clear()
        return out

    def _unique_path(self, target: Path) -> Path:
        if not target.exists():
            return target
        stem, ext = target.stem, target.suffix
        for n in range(1, 1000):
            candidate = target.with_name(f"{stem}-{n}{ext}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(
            f"Could not find a free filename next to {target} after 1000 tries."
        )

    # ---------------------------------------------------------------- finalize
    def finalize(self, *, success: bool) -> None:
        """Run end-of-run cleanup logic.

        - TEMP + success: wipe the temp dir.
        - TEMP + failure: keep it and log the path so the user can inspect.
        - SAVE / FEED: no-op (SAVE always persists; FEED has no disk).
        """
        if self.mode is FileMode.FEED:
            return
        if self.mode is FileMode.SAVE:
            log.info(
                "Run finished (success=%s); %d file(s) saved under %s",
                success,
                len(self._saved_paths),
                self.root,
            )
            return
        # TEMP mode.
        assert self.root is not None
        if not self._is_temp_root:
            return
        if success:
            try:
                shutil.rmtree(self.root, ignore_errors=False)
                log.info("Cleaned up TEMP workspace %s", self.root)
            except OSError as exc:
                log.warning("Failed to clean TEMP workspace %s: %s", self.root, exc)
        else:
            log.warning(
                "Run failed; KEEPING TEMP workspace at %s (%d file(s)) so you "
                "can inspect what was downloaded.",
                self.root,
                len(self._saved_paths),
            )


# ---------------------------------------------------------------- prompting --


def prompt_mode_interactively(
    *,
    default_workdir: Path | None = None,
    out=None,
    in_=None,
) -> tuple[FileMode, Path | None]:
    """Ask the user how files for this run should be handled.

    Returns ``(mode, workdir)``. ``workdir`` is None unless ``mode`` is
    ``SAVE``. ``out`` / ``in_`` are injectable for tests.
    """
    out = out if out is not None else sys.stderr
    in_ = in_ if in_ is not None else sys.stdin

    print(
        "\n"
        "File handling for this run:\n"
        "  [t] temp  — auto-cleanup on success, kept on failure for debugging "
        "(default)\n"
        "  [s] save  — persist all downloads to a directory\n"
        "  [f] feed  — never write to disk; show files to the VLM only",
        file=out,
    )
    while True:
        try:
            print("Choose [t/s/f] (default t): ", end="", file=out, flush=True)
            raw = in_.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("(no input — defaulting to TEMP)", file=out)
            return FileMode.TEMP, None
        if raw in {"", "t", "temp"}:
            return FileMode.TEMP, None
        if raw in {"f", "feed"}:
            return FileMode.FEED, None
        if raw in {"s", "save"}:
            break
        print(f"(unrecognized: {raw!r}; please type t, s, or f)", file=out)

    default = (default_workdir or Path("agent_files")).expanduser()
    while True:
        print(
            f"Save downloads to which directory? "
            f"(blank → {default}): ",
            end="",
            file=out,
            flush=True,
        )
        try:
            raw = in_.readline().strip()
        except (EOFError, KeyboardInterrupt):
            print(f"(no input — using {default})", file=out)
            raw = ""
        chosen = Path(raw).expanduser() if raw else default
        try:
            chosen = chosen.resolve()
            chosen.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"(could not create {chosen}: {exc} — try another path)", file=out)
            continue
        return FileMode.SAVE, chosen


def resolve_mode(
    *,
    cli_mode: FileMode | None,
    cli_workdir: Path | None,
    env_mode: FileMode | None,
    env_workdir: Path | None,
    interactive: bool,
    out=None,
    in_=None,
) -> tuple[FileMode, Path | None]:
    """Combine CLI / env / interactive sources to pick a final (mode, workdir).

    Precedence: CLI > env > interactive prompt > TEMP fallback. ``interactive``
    being ``False`` means "no TTY available — never prompt" (e.g. inside CI
    or a daemon).
    """
    mode = cli_mode if cli_mode is not None else env_mode
    workdir = cli_workdir if cli_workdir is not None else env_workdir

    if mode is None and interactive and sys.stdin is not None and sys.stdin.isatty():
        return prompt_mode_interactively(
            default_workdir=workdir,
            out=out,
            in_=in_,
        )

    if mode is None:
        mode = FileMode.TEMP

    if mode is FileMode.SAVE and workdir is None:
        workdir = Path(os.getcwd()) / "agent_files"
        log.warning(
            "FileMode.SAVE chosen but no workdir supplied; defaulting to %s.",
            workdir,
        )

    return mode, workdir


# ----------------------------------------------------- file-command executors


# Cap on how many bytes we'll fetch from a single DOWNLOAD URL. Defends
# against the planner pasting a 10GB ISO link by accident. Override via
# ``DOWNLOAD_MAX_BYTES`` if you really need to.
DOWNLOAD_MAX_BYTES = int(os.getenv("DOWNLOAD_MAX_BYTES", str(200 * 1024 * 1024)))
DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "60"))

_USER_AGENT = (
    "DesktopVLMAgent/1.0 (+https://github.com/nd1836017/"
    "https-github.com-nd1836017-desktop-vlm-agent)"
)


def _filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    base = os.path.basename(parsed.path) or "download.bin"
    if "." not in base:
        # No extension; punt with a generic suffix so the dialog doesn't
        # treat it as a directory.
        base = base + ".bin"
    return base


def execute_download(
    cmd: DownloadCommand,
    workspace: FileWorkspace,
    *,
    opener=None,
) -> tuple[bool, str]:
    """Fetch ``cmd.url`` over HTTP and persist it via ``workspace``.

    ``opener`` is for tests — defaults to ``urllib.request.urlopen``.

    Returns ``(success, action_text)``. ``action_text`` is the rendered
    string the agent records in summary/history.
    """
    url = cmd.url.strip()
    if not url:
        return False, "DOWNLOAD failed: empty URL"

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"DOWNLOAD failed: scheme {parsed.scheme!r} is not http(s)"

    suggested = (cmd.filename or _filename_from_url(url)).strip()

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    open_fn = opener if opener is not None else urllib.request.urlopen
    started = time.monotonic()
    try:
        with open_fn(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
            content = resp.read(DOWNLOAD_MAX_BYTES + 1)
    except urllib.error.URLError as exc:
        return False, f"DOWNLOAD failed: {exc}"
    except (OSError, ValueError) as exc:
        return False, f"DOWNLOAD failed: {exc}"

    if len(content) > DOWNLOAD_MAX_BYTES:
        return (
            False,
            f"DOWNLOAD aborted: response exceeded {DOWNLOAD_MAX_BYTES} bytes",
        )

    elapsed = time.monotonic() - started
    saved = workspace.save(content=content, suggested_name=suggested)
    where = "FEED buffer" if saved is None else str(saved)
    return (
        True,
        f"DOWNLOAD {url} -> {where} ({len(content)}B, {elapsed:.1f}s)",
    )


def execute_attach_file(
    cmd: AttachFileCommand,
    workspace: FileWorkspace,
    *,
    pyautogui_module=None,
    sleep=None,
) -> tuple[bool, str]:
    """Fill the most recently focused OS file-picker dialog with a path.

    Strategy: assume an OS file dialog is already open. Press ``Ctrl+L``
    (Windows / GNOME / KDE all support this for "focus the path bar"),
    type the absolute path, and press Enter. This is brittle by design —
    the planner is expected to click the dialog's "Browse" button first
    and only then emit ``ATTACH_FILE``.

    The path is resolved in priority order:
      1. exact filename inside the run workspace,
      2. ``(rowN)``-suffixed filename inside the workspace,
      3. absolute / CWD-relative path on disk.
    """
    candidate = cmd.filename.strip()
    if not candidate:
        return False, "ATTACH_FILE failed: empty filename"

    resolved = _resolve_attach_path(candidate, workspace)
    if resolved is None:
        return False, f"ATTACH_FILE failed: file not found: {candidate}"

    pyautogui = pyautogui_module if pyautogui_module is not None else _import_pyautogui()
    pyautogui.hotkey("ctrl", "l")
    (sleep or time.sleep)(0.4)
    # typewrite is fine here: file paths are ASCII on Windows + Linux dialogs.
    pyautogui.typewrite(str(resolved), interval=0.01)
    (sleep or time.sleep)(0.2)
    pyautogui.press("enter")
    (sleep or time.sleep)(0.6)
    return True, f"ATTACH_FILE {resolved}"


def _resolve_attach_path(candidate: str, workspace: FileWorkspace) -> Path | None:
    if workspace.root is not None:
        direct = workspace.root / candidate
        if direct.is_file():
            return direct.resolve()
        suffixed = workspace.root / name_with_row_suffix(
            candidate, workspace.current_row
        )
        if suffixed.is_file():
            return suffixed.resolve()

    on_disk = Path(candidate).expanduser()
    if on_disk.is_file():
        return on_disk.resolve()
    return None


def _import_pyautogui():  # pragma: no cover - exercised via mocks in tests.
    import pyautogui

    pyautogui.FAILSAFE = True
    return pyautogui


def execute_capture_for_ai(
    cmd: CaptureForAiCommand,
    workspace: FileWorkspace,
    *,
    screenshot: Image | None = None,
) -> tuple[bool, str]:
    """Buffer image bytes for the next plan call.

    - With ``filename``: read bytes from ``filename`` (resolved against
      the workspace, then disk).
    - Without: serialize ``screenshot`` (the current frame) as PNG.
    """
    name = cmd.filename.strip()
    if name:
        path = _resolve_attach_path(name, workspace)
        if path is None:
            return (
                False,
                f"CAPTURE_FOR_AI failed: file not found: {name}",
            )
        try:
            content = path.read_bytes()
        except OSError as exc:
            return False, f"CAPTURE_FOR_AI failed: {exc}"
        workspace.push_feed(content)
        return True, f"CAPTURE_FOR_AI {path}"

    if screenshot is None:
        return False, "CAPTURE_FOR_AI failed: no screenshot available"

    buf = io.BytesIO()
    screenshot.save(buf, format="PNG")
    workspace.push_feed(buf.getvalue())
    return True, "CAPTURE_FOR_AI <current screenshot>"


# ----------------------------------------------------- task-feature inspection


@dataclass(frozen=True)
class RunFeatures:
    """Which tasks-file features a run actually uses.

    Populated by ``inspect_features``; the agent uses these flags to gate
    interactive prompts (only ask about file modes when the run actually
    has file primitives) and to print a clear "what's in this run" summary.
    """

    uses_csv_loop: bool = False
    csv_row_count: int = 0
    csv_files: tuple[str, ...] = ()
    uses_downloads: bool = False
    download_count: int = 0
    uses_attach_file: bool = False
    attach_file_count: int = 0
    uses_capture_for_ai: bool = False
    capture_for_ai_count: int = 0

    @property
    def uses_files(self) -> bool:
        """Any feature that needs an on-disk or in-memory file workspace."""
        return (
            self.uses_downloads
            or self.uses_attach_file
            or self.uses_capture_for_ai
        )


# Pattern hits any of the file-primitive command verbs at the start of a
# step's natural-language text. Tasks files written in plain English
# generally won't trigger these unless the user really did write them.
_DOWNLOAD_HINT_RE = re.compile(r"\bDOWNLOAD\s*\[", re.IGNORECASE)
_ATTACH_FILE_HINT_RE = re.compile(r"\bATTACH[_\s]?FILE\s*\[", re.IGNORECASE)
_CAPTURE_FOR_AI_HINT_RE = re.compile(r"\bCAPTURE[_\s]?FOR[_\s]?AI\b", re.IGNORECASE)


def inspect_features(steps) -> RunFeatures:
    """Scan expanded TaskSteps to determine which features the run uses.

    ``steps`` should be the list of ``TaskStep`` objects from
    ``load_steps``. Both literal command syntax (``DOWNLOAD [url]``) and
    natural-language hints get counted — the planner is free to translate
    "download the invoice PDF from {url}" into a DOWNLOAD command, but
    the user's intent is already visible in the source text.
    """
    csv_rows = 0
    csv_files: set[str] = set()
    download = 0
    attach = 0
    capture = 0

    for step in steps:
        # Heuristic flags from natural-language text. We match the literal
        # primitive verb so casual mentions ("download" lowercase in a
        # sentence) don't count as opt-in.
        if _DOWNLOAD_HINT_RE.search(step.text):
            download += 1
        if _ATTACH_FILE_HINT_RE.search(step.text):
            attach += 1
        if _CAPTURE_FOR_AI_HINT_RE.search(step.text):
            capture += 1
        # Row metadata is set by the loader for every step expanded out
        # of a FOR_EACH_ROW block.
        if step.row_index is not None:
            csv_rows = max(csv_rows, step.row_index)
            if step.csv_name:
                csv_files.add(step.csv_name)

    return RunFeatures(
        uses_csv_loop=csv_rows > 0,
        csv_row_count=csv_rows,
        csv_files=tuple(sorted(csv_files)),
        uses_downloads=download > 0,
        download_count=download,
        uses_attach_file=attach > 0,
        attach_file_count=attach,
        uses_capture_for_ai=capture > 0,
        capture_for_ai_count=capture,
    )


def format_features_summary(features: RunFeatures, *, total_steps: int) -> str:
    """Return a one-paragraph human-readable rundown of detected features."""
    lines = [f"Tasks loaded: {total_steps} step(s)"]
    if features.uses_csv_loop:
        suffix = (
            f"({features.csv_row_count} rows × inner steps "
            f"from {', '.join(features.csv_files)})"
            if features.csv_files
            else f"({features.csv_row_count} rows)"
        )
        lines.append(f"  - FOR_EACH_ROW {suffix}")
    if features.uses_downloads:
        lines.append(f"  - DOWNLOAD × {features.download_count}")
    if features.uses_attach_file:
        lines.append(f"  - ATTACH_FILE × {features.attach_file_count}")
    if features.uses_capture_for_ai:
        lines.append(f"  - CAPTURE_FOR_AI × {features.capture_for_ai_count}")
    if not (
        features.uses_csv_loop or features.uses_files
    ):
        lines.append("  - (no special features detected)")
    return "\n".join(lines)
