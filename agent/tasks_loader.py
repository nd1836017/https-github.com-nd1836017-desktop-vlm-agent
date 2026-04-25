"""Tasks-file loader with FOR_EACH_ROW + CSV templating support.

The legacy ``tasks.txt`` format is one natural-language instruction per
line, with blank lines and ``#`` comments ignored. This loader extends
that format with two new directives so a single tasks file can drive a
loop of repetitive work (e.g. filling a form once per row of a CSV).

Syntax
------

    FOR_EACH_ROW [path/to/data.csv]
        Click the First name field
        Type "{{row.first_name}}"
        Click the Email field
        Type "{{row.email}}"
        Click Submit
    END_FOR_EACH_ROW

* Lines between ``FOR_EACH_ROW`` and ``END_FOR_EACH_ROW`` are repeated
  once per CSV row.
* ``{{row.<field_name>}}`` is replaced with the cell value for that
  field. Field names match the CSV header exactly (case-sensitive).
* ``{{row.<field>|<default>}}`` substitutes ``<default>`` when the cell
  is empty. Whitespace inside the placeholder is tolerated:
  ``{{ row.email | none@example.com }}`` works the same.
* The CSV path can be absolute or relative to the tasks file's directory.
* The CSV path can be overridden at runtime by passing
  ``csv_override`` (e.g. via ``python -m agent --csv real_data.csv``).
* Nesting ``FOR_EACH_ROW`` blocks is not supported and is rejected at
  load time. The same goes for any other malformed block.
* Lines outside a ``FOR_EACH_ROW`` block must not contain
  ``{{row.<field>}}`` placeholders — that would be silently meaningless,
  so we reject it loudly.
"""
from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable, Sequence
from io import StringIO
from pathlib import Path

log = logging.getLogger(__name__)


_FOR_EACH_RE = re.compile(
    r"^\s*FOR_EACH_ROW\s*\[\s*(.+?)\s*\]\s*$",
    re.IGNORECASE,
)
_END_FOR_EACH_RE = re.compile(r"^\s*END_FOR_EACH_ROW\s*$", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"\{\{\s*row\.([A-Za-z0-9_][A-Za-z0-9_\- ]*?)\s*(?:\|([^}]*))?\s*\}\}"
)


class TasksLoadError(ValueError):
    """Raised when a tasks file cannot be parsed (clear, user-facing message)."""


def load_tasks(
    path: Path,
    csv_override: Path | None = None,
) -> list[str]:
    """Parse and expand a tasks file into a flat list of step strings.

    The returned list has all ``FOR_EACH_ROW`` blocks fully expanded with
    ``{{row.<field>}}`` placeholders substituted. The returned strings
    are exactly what the planner will see — so a sample row's expansion
    can be inspected directly.

    Raises ``FileNotFoundError`` if ``path`` does not exist, and
    ``TasksLoadError`` for any structural or templating issue.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    raw_text = path.read_text(encoding="utf-8-sig")
    lines = raw_text.splitlines()
    return _expand(lines, base_dir=path.parent, csv_override=csv_override)


def _expand(
    lines: Sequence[str],
    *,
    base_dir: Path,
    csv_override: Path | None,
) -> list[str]:
    out: list[str] = []
    in_block = False
    block_csv_path: Path | None = None
    block_lineno: int = 0
    block_body: list[tuple[int, str]] = []

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        end_match = _END_FOR_EACH_RE.match(stripped)
        if end_match is not None:
            if not in_block:
                raise TasksLoadError(
                    f"Line {lineno}: END_FOR_EACH_ROW without a matching "
                    f"FOR_EACH_ROW."
                )
            assert block_csv_path is not None  # for type checker
            out.extend(
                _expand_block(
                    block_body,
                    csv_path=block_csv_path,
                    block_lineno=block_lineno,
                )
            )
            in_block = False
            block_csv_path = None
            block_body = []
            continue

        for_match = _FOR_EACH_RE.match(stripped)
        if for_match is not None:
            if in_block:
                raise TasksLoadError(
                    f"Line {lineno}: nested FOR_EACH_ROW is not supported "
                    f"(outer block opened on line {block_lineno})."
                )
            csv_arg = for_match.group(1).strip()
            if not csv_arg:
                raise TasksLoadError(
                    f"Line {lineno}: FOR_EACH_ROW requires a CSV path "
                    f"(e.g. FOR_EACH_ROW [data.csv])."
                )
            chosen = csv_override if csv_override is not None else Path(csv_arg)
            block_csv_path = chosen if chosen.is_absolute() else (base_dir / chosen)
            block_lineno = lineno
            in_block = True
            block_body = []
            continue

        if in_block:
            block_body.append((lineno, stripped))
        else:
            # Outside a block: reject placeholders so users don't silently
            # ship unsubstituted text to the VLM.
            if _PLACEHOLDER_RE.search(stripped):
                raise TasksLoadError(
                    f"Line {lineno}: '{{{{row.<field>}}}}' placeholder used "
                    f"outside a FOR_EACH_ROW block (no CSV row to substitute)."
                )
            out.append(stripped)

    if in_block:
        raise TasksLoadError(
            f"FOR_EACH_ROW opened on line {block_lineno} was never closed "
            f"with END_FOR_EACH_ROW."
        )

    return out


def _expand_block(
    body: Sequence[tuple[int, str]],
    *,
    csv_path: Path,
    block_lineno: int,
) -> list[str]:
    if not body:
        log.warning(
            "FOR_EACH_ROW on line %d has an empty body; skipping the loop.",
            block_lineno,
        )
        return []

    rows = _read_csv_rows(csv_path, block_lineno=block_lineno)
    if not rows:
        log.warning(
            "FOR_EACH_ROW on line %d: CSV %s has zero rows; loop expanded "
            "to nothing.",
            block_lineno,
            csv_path,
        )
        return []

    fields = list(rows[0].keys())
    expanded: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        for body_lineno, body_line in body:
            try:
                expanded.append(_substitute(body_line, row=row, fields=fields))
            except TasksLoadError as exc:
                raise TasksLoadError(
                    f"Line {body_lineno} (row {row_idx} of {csv_path.name}): {exc}"
                ) from None
    return expanded


def _read_csv_rows(csv_path: Path, *, block_lineno: int) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise TasksLoadError(
            f"CSV file referenced by FOR_EACH_ROW on line {block_lineno} "
            f"does not exist: {csv_path}"
        )

    text = csv_path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise TasksLoadError(
            f"CSV {csv_path} has no header row; FOR_EACH_ROW needs named "
            f"columns to substitute via {{{{row.<field>}}}}."
        )
    return [_clean_row(r) for r in reader]


def _clean_row(row: dict[str, str | None]) -> dict[str, str]:
    """Normalize cell values: drop None (extra columns) and strip cells."""
    cleaned: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue  # extra columns past the header
        if value is None:
            cleaned[key] = ""
        else:
            cleaned[key] = value.strip()
    return cleaned


def _substitute(
    line: str,
    *,
    row: dict[str, str],
    fields: Iterable[str],
) -> str:
    """Replace ``{{row.<field>}}`` placeholders with row values.

    Raises ``TasksLoadError`` if a placeholder references an unknown field
    so the user gets a clear error rather than silent empty text.
    """

    def repl(match: re.Match[str]) -> str:
        field = match.group(1).strip()
        default = match.group(2)
        if field not in row:
            available = ", ".join(fields) or "(none)"
            raise TasksLoadError(
                f"unknown CSV field {field!r} (available: {available})"
            )
        value = row[field]
        if value == "" and default is not None:
            return default.strip()
        return value

    return _PLACEHOLDER_RE.sub(repl, line)
