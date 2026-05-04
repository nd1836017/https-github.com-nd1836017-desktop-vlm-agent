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
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path

from .skills import (
    SkillError,
    expand_use_directives,
    find_skill_for_step,
    load_skill_triggers,
)
from .task_router import RoutingHint, parse_inline_annotation

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskStep:
    """One expanded instruction with optional FOR_EACH_ROW row metadata.

    ``text`` is the natural-language instruction post-substitution — exactly
    what the planner will see. ``row_index`` is 1-indexed and ``None``
    outside a FOR_EACH_ROW block; ``csv_name`` is the basename of the CSV
    that drove this row.

    ``routing_hint`` is the smart-router classification (``browser-fast``
    / ``browser-vlm`` / ``desktop-vlm``) plus an optional
    ``suggested_command``. Set by ``apply_router_hints`` AFTER load.
    Inline ``[tag]`` annotations from tasks.txt are parsed at load time
    and surface here as ``RoutingHint(source='manual', ...)`` — we do
    that here (not in ``apply_router``) so the user-visible step text is
    cleaned of the bracket prefix before any other code sees it.

    ``control_kind`` flags this step as a control-flow directive:
    ``if_begin`` / ``if_else`` / ``if_end`` / ``wait_until``. None for
    plain action steps. Control directives never call the planner; the
    agent loop interprets them directly.

    ``block_id`` groups the if_begin / if_else / if_end of the same
    IF/ELSE/END_IF triple. ``active_block_id`` is set on every step
    that lives INSIDE such a block (including the directives
    themselves), and ``branch`` is ``"then"`` or ``"else"`` depending
    on which side of the block the step is in. These let the agent
    loop skip the non-taken branch with one comparison per step.

    ``condition_text`` carries the bracketed text of an IF or
    WAIT_UNTIL directive. ``wait_timeout_seconds`` / ``wait_poll_seconds``
    are set only on WAIT_UNTIL steps; both default to None and are
    filled in from config at run time.
    """

    text: str
    row_index: int | None = None
    csv_name: str | None = None
    routing_hint: RoutingHint | None = None
    control_kind: str | None = None
    block_id: int | None = None
    active_block_id: int | None = None
    branch: str | None = None
    condition_text: str | None = None
    wait_timeout_seconds: float | None = None
    wait_poll_seconds: float | None = None
    # Smart step-skip manual annotation. When True, the agent runs an
    # "is the goal already on screen?" check BEFORE the planner ever
    # fires for this step — if it is, the step is recorded as a
    # synthetic PASS and the run advances. Set by a leading
    # ``[skippable]`` prefix on the tasks.txt line. Always False for
    # control-flow directives (IF/ELSE/END_IF/WAIT_UNTIL).
    skippable: bool = False


_SKIPPABLE_RE = re.compile(
    r"^\s*\[\s*skippable\s*\]\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)


def _parse_step_annotations(line: str) -> tuple[RoutingHint | None, bool, str]:
    """Strip leading ``[routing]`` and ``[skippable]`` annotations.

    Both annotations are independent; either or both may appear in
    any order at the start of the line:

        ``[browser] click submit``                  → routing only
        ``[skippable] click submit``                → skippable only
        ``[skippable] [browser] click submit``      → both
        ``[browser] [skippable] click submit``      → both (any order)

    Returns ``(routing_hint, skippable, remaining_text)``. We loop
    until neither pattern matches the head, so a third unrecognised
    bracket prefix (e.g. ``[admin]`` as a literal placeholder) is
    left untouched on the line — same conservative behaviour as the
    underlying ``parse_inline_annotation``.
    """
    skippable = False
    hint: RoutingHint | None = None
    remaining = line
    # Cap at 4 iterations so a pathological "[skippable] [browser]
    # [skippable] …" line can't loop forever. In practice we expect
    # at most 2 prefixes.
    for _ in range(4):
        next_hint, next_remaining = parse_inline_annotation(remaining)
        if next_hint is not None and next_hint.source == "manual":
            # Last-write-wins on routing — but a duplicate hint is a
            # red flag in the user's tasks file. We don't raise (the
            # router/run loop tolerates it) but we do log so it's
            # visible.
            if hint is not None:
                log.warning(
                    "tasks file: multiple routing annotations on a single "
                    "line; using the last one (%r).",
                    next_hint.complexity,
                )
            hint = next_hint
            remaining = next_remaining
            continue
        skip_match = _SKIPPABLE_RE.match(remaining)
        if skip_match is not None:
            skippable = True
            remaining = skip_match.group("text")
            continue
        break
    return hint, skippable, remaining


_FOR_EACH_RE = re.compile(
    r"^\s*FOR_EACH_ROW\s*\[\s*(.+?)\s*\]\s*$",
    re.IGNORECASE,
)
_END_FOR_EACH_RE = re.compile(r"^\s*END_FOR_EACH_ROW\s*$", re.IGNORECASE)
_IF_RE = re.compile(
    # ``(.*?)`` (zero or more) so an empty ``IF [] THEN`` still matches
    # the directive and produces our "non-empty condition required"
    # error instead of falling through to the action-step path.
    r"^\s*IF\s*\[\s*(.*?)\s*\]\s*THEN\s*$",
    re.IGNORECASE,
)
_ELSE_RE = re.compile(r"^\s*ELSE\s*$", re.IGNORECASE)
_END_IF_RE = re.compile(r"^\s*END_IF\s*$", re.IGNORECASE)
_WAIT_UNTIL_RE = re.compile(
    r"^\s*WAIT_UNTIL\s*\[\s*(.*?)\s*\]\s*$",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"\{\{\s*row\.([A-Za-z0-9_][A-Za-z0-9_\- ]*?)\s*(?:\|([^}]*))?\s*\}\}"
)


class TasksLoadError(ValueError):
    """Raised when a tasks file cannot be parsed (clear, user-facing message)."""


def load_tasks(
    path: Path,
    csv_override: Path | None = None,
    skills_dir: Path | None = None,
) -> list[str]:
    """Parse and expand a tasks file into a flat list of step strings.

    The returned list has all ``FOR_EACH_ROW`` blocks fully expanded with
    ``{{row.<field>}}`` placeholders substituted, and every ``USE
    skill_name`` line replaced with the contents of that skill file.
    Returned strings are exactly what the planner will see, so a
    sample row's expansion can be inspected directly. Control-flow
    directives (IF/ELSE/END_IF/WAIT_UNTIL) appear as their own entries
    so an inspection still shows the structure.

    Raises ``FileNotFoundError`` if ``path`` does not exist, and
    ``TasksLoadError`` for any structural or templating issue.
    """
    return [
        s.text for s in load_steps(path, csv_override=csv_override, skills_dir=skills_dir)
    ]


def load_steps(
    path: Path,
    csv_override: Path | None = None,
    skills_dir: Path | None = None,
) -> list[TaskStep]:
    """Like ``load_tasks`` but returns ``TaskStep`` objects with row metadata.

    Use this from the agent run loop so primitives like ``DOWNLOAD`` can
    suffix filenames with ``(rowN)`` when a step is part of a FOR_EACH_ROW
    block. ``load_tasks`` is preserved as a thin wrapper for callers that
    only need the flat string list (notably the test suite).

    ``skills_dir`` enables the ``USE skill_name`` directive — when None,
    a tasks file containing ``USE`` lines raises ``TasksLoadError``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    if csv_override is not None and not csv_override.is_absolute():
        csv_override = csv_override.resolve()

    raw_text = path.read_text(encoding="utf-8-sig")
    lines = raw_text.splitlines()
    # Expand USE skill_name directives BEFORE FOR_EACH_ROW / IF / WAIT_UNTIL
    # parsing so a skill file can use any of those directives itself.
    try:
        lines = expand_use_directives(lines, skills_dir=skills_dir)
    except Exception as exc:
        # Wrap SkillError in TasksLoadError so callers handle one
        # exception type at the load boundary.
        raise TasksLoadError(str(exc)) from exc
    return _expand(lines, base_dir=path.parent, csv_override=csv_override)


def attach_routing_hints(
    steps: Sequence[TaskStep],
    hints: Sequence[RoutingHint | None],
) -> list[TaskStep]:
    """Pair ``hints`` with ``steps`` to produce a new annotated list.

    Used by the run loop after ``apply_router`` returns. Manual hints
    (already attached at load time) BEAT auto hints — the user's
    explicit ``[tag]`` annotation is always more authoritative than
    the router's classification. ``hints`` longer or shorter than
    ``steps`` is a programmer error and raises ``ValueError``.
    """
    if len(steps) != len(hints):
        raise ValueError(
            f"attach_routing_hints: {len(hints)} hints for {len(steps)} steps"
        )
    out: list[TaskStep] = []
    for step, auto_hint in zip(steps, hints, strict=True):
        if step.routing_hint is not None and step.routing_hint.source == "manual":
            # Keep the manual annotation; auto-router doesn't override.
            out.append(step)
            continue
        if auto_hint is None:
            out.append(step)
            continue
        out.append(replace(step, routing_hint=auto_hint))
    return out


def apply_skill_auto_use(
    steps: Sequence[TaskStep],
    *,
    skills_dir: Path | None,
    enabled: bool,
) -> list[TaskStep]:
    """Auto-expand skill triggers in a TaskStep list.

    For each step whose text matches a skill's ``# TRIGGERS:`` keyword
    list, REPLACE the step with the parsed contents of that skill.
    Skills are read from ``skills_dir``; control directives inside
    the skill (IF/ELSE/WAIT_UNTIL/etc.) are re-parsed via the same
    loader path used by ``load_steps``, so a skill that opens an IF
    block also gets a matching control_kind / block_id on its
    substeps.

    Single-pass design: substeps produced by an auto-expansion are
    NOT themselves re-checked for matches. This guarantees no
    infinite expansion even if a skill's content text happens to
    contain another skill's trigger keyword.

    Steps that ARE skipped from matching:
      * Control-flow directives (``IF``/``ELSE``/``END_IF``/``WAIT_UNTIL``).
        These are run-time control structures, not user actions; we
        never expand a skill in their place.
      * Steps whose ``routing_hint.source == "manual"``. The user
        has already declared an explicit routing intent for these,
        so we treat them as "do not touch".

    Returns ``list(steps)`` unchanged when:
      * ``enabled`` is ``False``;
      * ``skills_dir`` is None or empty;
      * no skill declares a ``# TRIGGERS:`` header.

    Errors expanding an individual skill are logged and the original
    step kept (skill auto-use is advisory; never crashes the run).
    """
    if not enabled:
        return list(steps)
    triggers = load_skill_triggers(skills_dir)
    if not triggers or skills_dir is None:
        return list(steps)

    out: list[TaskStep] = []
    matched_total = 0
    for step in steps:
        if step.control_kind is not None:
            out.append(step)
            continue
        if step.routing_hint is not None and step.routing_hint.source == "manual":
            out.append(step)
            continue
        skill_name = find_skill_for_step(step.text, triggers)
        if skill_name is None:
            out.append(step)
            continue
        try:
            sub_steps = _expand_skill_into_steps(skill_name, skills_dir)
        except (SkillError, TasksLoadError) as exc:
            log.warning(
                "skill auto-use: failed to expand %r for step %r: %s; "
                "leaving step unchanged.",
                skill_name,
                step.text,
                exc,
            )
            out.append(step)
            continue
        log.info(
            "skill auto-use: matched %r → step %r (expanded into %d sub-step(s))",
            skill_name,
            step.text,
            len(sub_steps),
        )
        out.extend(sub_steps)
        matched_total += 1
    if matched_total:
        log.info(
            "skill auto-use: expanded %d step(s) using %d skill trigger set(s)",
            matched_total,
            len(triggers),
        )
    return out


def _expand_skill_into_steps(
    skill_name: str, skills_dir: Path
) -> list[TaskStep]:
    """Produce a TaskStep list from the contents of one skill.

    Re-uses the same USE-expansion + control-flow parser as
    ``load_steps``, so the skill's IF/ELSE/WAIT_UNTIL blocks are
    fully understood. ``base_dir`` for FOR_EACH_ROW CSVs defaults to
    ``skills_dir`` — skills almost never use FOR_EACH_ROW (a CSV
    inside a skill folder is unusual) but when they do, relative
    CSV paths resolve against the skills directory.
    """
    raw_lines = expand_use_directives(
        [f"USE {skill_name}"], skills_dir=skills_dir
    )
    return _expand(
        raw_lines,
        base_dir=skills_dir,
        csv_override=None,
    )


def _expand(
    lines: Sequence[str],
    *,
    base_dir: Path,
    csv_override: Path | None,
) -> list[TaskStep]:
    out: list[TaskStep] = []
    # FOR_EACH_ROW state.
    in_for_block = False
    block_csv_path: Path | None = None
    block_lineno: int = 0
    block_body: list[tuple[int, str]] = []
    # IF/ELSE/END_IF state. We only allow a single (un-nested) IF block
    # at the top level for v1 — nested IF is rejected at parse time so
    # the agent loop's branch tracking can stay a single field.
    if_state: str | None = None  # None | "then" | "else"
    if_block_id: int = 0
    next_block_id: int = 0
    if_lineno: int = 0

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        end_match = _END_FOR_EACH_RE.match(stripped)
        if end_match is not None:
            if not in_for_block:
                raise TasksLoadError(
                    f"Line {lineno}: END_FOR_EACH_ROW without a matching "
                    f"FOR_EACH_ROW."
                )
            assert block_csv_path is not None  # for type checker
            block_steps = _expand_block(
                block_body,
                csv_path=block_csv_path,
                block_lineno=block_lineno,
            )
            # If this FOR_EACH_ROW lives inside an active IF branch,
            # tag every emitted step with that branch's metadata.
            if if_state is not None:
                block_steps = [
                    replace(
                        s,
                        active_block_id=if_block_id,
                        branch=if_state,
                    )
                    for s in block_steps
                ]
            out.extend(block_steps)
            in_for_block = False
            block_csv_path = None
            block_body = []
            continue

        for_match = _FOR_EACH_RE.match(stripped)
        if for_match is not None:
            if in_for_block:
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
            in_for_block = True
            block_body = []
            continue

        if in_for_block:
            # Inside a FOR_EACH_ROW: collect raw lines for later
            # per-row expansion. Reject IF / WAIT_UNTIL inside a
            # loop block — control flow is allowed at the top level
            # only for v1, otherwise the parser/agent semantics
            # multiply out (e.g. an IF that should be evaluated once
            # per row vs once before the loop).
            if (
                _IF_RE.match(stripped)
                or _ELSE_RE.match(stripped)
                or _END_IF_RE.match(stripped)
                or _WAIT_UNTIL_RE.match(stripped)
            ):
                raise TasksLoadError(
                    f"Line {lineno}: IF / ELSE / END_IF / WAIT_UNTIL "
                    f"are not supported inside a FOR_EACH_ROW block "
                    f"(opened on line {block_lineno})."
                )
            block_body.append((lineno, stripped))
            continue

        # Top-level (outside FOR_EACH_ROW) — handle IF/ELSE/END_IF/WAIT_UNTIL.
        if_match = _IF_RE.match(stripped)
        if if_match is not None:
            if if_state is not None:
                raise TasksLoadError(
                    f"Line {lineno}: nested IF is not supported "
                    f"(outer IF opened on line {if_lineno})."
                )
            condition_text = if_match.group(1).strip()
            if not condition_text:
                raise TasksLoadError(
                    f"Line {lineno}: IF requires non-empty condition text "
                    f"(e.g. IF [Sign in to your account] THEN)."
                )
            if_block_id = next_block_id
            next_block_id += 1
            if_state = "then"
            if_lineno = lineno
            out.append(
                TaskStep(
                    text=stripped,
                    control_kind="if_begin",
                    block_id=if_block_id,
                    active_block_id=if_block_id,
                    branch="then",
                    condition_text=condition_text,
                )
            )
            continue
        if _ELSE_RE.match(stripped) is not None:
            if if_state != "then":
                raise TasksLoadError(
                    f"Line {lineno}: ELSE without a matching IF (or used "
                    f"twice in the same block)."
                )
            if_state = "else"
            out.append(
                TaskStep(
                    text=stripped,
                    control_kind="if_else",
                    block_id=if_block_id,
                    active_block_id=if_block_id,
                    branch="else",
                )
            )
            continue
        if _END_IF_RE.match(stripped) is not None:
            if if_state is None:
                raise TasksLoadError(
                    f"Line {lineno}: END_IF without a matching IF."
                )
            out.append(
                TaskStep(
                    text=stripped,
                    control_kind="if_end",
                    block_id=if_block_id,
                    active_block_id=if_block_id,
                    branch=if_state,
                )
            )
            if_state = None
            continue

        wait_match = _WAIT_UNTIL_RE.match(stripped)
        if wait_match is not None:
            condition_text = wait_match.group(1).strip()
            if not condition_text:
                raise TasksLoadError(
                    f"Line {lineno}: WAIT_UNTIL requires non-empty condition "
                    f"text (e.g. WAIT_UNTIL [Welcome back])."
                )
            out.append(
                TaskStep(
                    text=stripped,
                    control_kind="wait_until",
                    condition_text=condition_text,
                    active_block_id=if_block_id if if_state is not None else None,
                    branch=if_state,
                )
            )
            continue

        # Plain action step.
        if _PLACEHOLDER_RE.search(stripped):
            raise TasksLoadError(
                f"Line {lineno}: '{{{{row.<field>}}}}' placeholder used "
                f"outside a FOR_EACH_ROW block (no CSV row to substitute)."
            )
        hint, skippable, cleaned = _parse_step_annotations(stripped)
        out.append(
            TaskStep(
                text=cleaned,
                routing_hint=hint,
                active_block_id=if_block_id if if_state is not None else None,
                branch=if_state,
                skippable=skippable,
            )
        )

    if in_for_block:
        raise TasksLoadError(
            f"FOR_EACH_ROW opened on line {block_lineno} was never closed "
            f"with END_FOR_EACH_ROW."
        )
    if if_state is not None:
        raise TasksLoadError(
            f"IF opened on line {if_lineno} was never closed with END_IF."
        )

    return out


def _expand_block(
    body: Sequence[tuple[int, str]],
    *,
    csv_path: Path,
    block_lineno: int,
) -> list[TaskStep]:
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
    csv_name = csv_path.name
    expanded: list[TaskStep] = []
    for row_idx, row in enumerate(rows, start=1):
        for body_lineno, body_line in body:
            try:
                text = _substitute(body_line, row=row, fields=fields)
            except TasksLoadError as exc:
                raise TasksLoadError(
                    f"Line {body_lineno} (row {row_idx} of {csv_path.name}): {exc}"
                ) from None
            # Strip inline routing annotations on FOR_EACH_ROW body lines too.
            # The same ``[browser-fast] click submit`` syntax should work
            # whether the line is inside or outside a loop.
            hint, skippable, cleaned = _parse_step_annotations(text)
            expanded.append(
                TaskStep(
                    text=cleaned,
                    row_index=row_idx,
                    csv_name=csv_name,
                    routing_hint=hint,
                    skippable=skippable,
                )
            )
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
