"""Skill library — reusable named tasks that can be ``USE``d from any tasks file.

A *skill* is just a tasks file living in the configured skills directory
(``SKILLS_DIR``, default ``skills/``). Each ``.txt`` file is one skill; the
filename (minus extension) is the skill name.

In a tasks file:

    USE login_to_gmail
    click compose
    type "hello"

…is equivalent to inlining the contents of ``skills/login_to_gmail.txt``
at the ``USE`` line. The expansion happens at load time, *before*
FOR_EACH_ROW expansion, IF/ELSE control flow, the decomposer, or the
router. Any feature the agent supports inside a normal tasks file works
inside a skill file too.

Why a separate file rather than copy/pasting steps:

* Reuse — once you've solved "log into Gmail", you don't re-author it.
* Versioning — skills are plain text, easy to ``git diff``.
* Sharing — a folder of ``.txt`` files is the simplest possible
  exchange format.

Skills can ``USE`` other skills. Cycles ("a uses b uses a") are
detected and rejected with a clear error pointing at the cycle path.
A configurable ``max_depth`` keeps pathological dependency chains
from blowing the call stack.

Authoring helpers:

* ``list_skills(skills_dir)`` — return ``SkillInfo`` for every skill
  found, used by the ``--list-skills`` CLI flag.
* ``scaffold_skill(skills_dir, name)`` — write a starter ``.txt`` for
  a new skill, used by the ``--new-skill <name>`` CLI flag.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ``USE skill_name`` — skill name is filename-style: letters, digits,
# underscore, hyphen. Reject anything weirder so users can't accidentally
# turn ``USE ../../etc/passwd`` into a path-traversal vector.
_USE_RE = re.compile(r"^\s*USE\s+([A-Za-z0-9_\-]+)\s*$", re.IGNORECASE)
_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# Skill files may declare auto-use trigger keywords via a header
# comment line of the form ``# TRIGGERS: kw1, kw2, kw3``. The
# detector walks each step's text and matches against these keywords
# (case-insensitive, word-boundary). Skills without a TRIGGERS line
# are simply not auto-matched (they remain available via manual USE).
_TRIGGERS_RE = re.compile(
    r"^\s*#\s*triggers\s*:\s*(?P<list>.+)$",
    re.IGNORECASE,
)


# Default directory name relative to the user's CWD when they don't set
# SKILLS_DIR explicitly. Agreed-upon convention so users can grep for it.
DEFAULT_SKILLS_DIR_NAME = "skills"

# Hard cap on USE-chain depth. 8 is plenty for human-authored skill
# trees and keeps a bug from turning into a stack overflow.
DEFAULT_MAX_DEPTH = 8

# Length cap on the one-line preview shown by ``list_skills``. Long
# enough to convey purpose, short enough to keep the listing readable.
PREVIEW_MAX_CHARS = 80


class SkillError(ValueError):
    """Raised on any skill-loading failure (missing skill, cycle, bad name)."""


@dataclass(frozen=True)
class SkillInfo:
    """Metadata for one skill, suitable for ``--list-skills`` output."""

    name: str
    path: Path
    line_count: int
    preview: str  # one-line description (first non-blank, non-comment line)


def expand_use_directives(
    lines: Sequence[str],
    *,
    skills_dir: Path | None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[str]:
    """Expand any ``USE skill_name`` lines into the referenced skill content.

    ``lines`` is the raw tasks-file content (one element per line, no
    trailing newlines required). Returned list is the fully-inlined
    sequence — the rest of the loader pipeline can treat the result
    exactly like a tasks file with no skills.

    Skill files are read from ``skills_dir`` (e.g. ``skills/<name>.txt``).
    If ``skills_dir`` is ``None`` or doesn't exist, ``USE`` lines raise
    ``SkillError`` — we don't silently ignore them, because a missing
    skill almost always indicates user error (typo, wrong env var).

    Recursive expansion is supported: a skill file can itself contain
    ``USE`` lines. ``max_depth`` bounds the chain to catch infinite
    self-reference; cycles are detected explicitly via the skill chain.
    """
    return _expand(
        list(lines),
        skills_dir=skills_dir,
        chain=(),
        depth=0,
        max_depth=max_depth,
    )


def _expand(
    lines: list[str],
    *,
    skills_dir: Path | None,
    chain: tuple[str, ...],
    depth: int,
    max_depth: int,
) -> list[str]:
    out: list[str] = []
    for raw in lines:
        match = _USE_RE.match(raw)
        if match is None:
            out.append(raw)
            continue

        skill_name = match.group(1).strip()
        if depth >= max_depth:
            raise SkillError(
                f"USE {skill_name!r}: skill chain exceeded max depth "
                f"({max_depth}). Likely an unintended recursive chain "
                f"(via {' -> '.join(chain)})."
            )
        if skill_name in chain:
            cycle = " -> ".join((*chain, skill_name))
            raise SkillError(
                f"USE {skill_name!r}: cyclic skill reference "
                f"({cycle}). Skills must form a directed acyclic graph."
            )
        skill_path = _resolve_skill(skill_name, skills_dir=skills_dir)
        skill_lines = skill_path.read_text(encoding="utf-8-sig").splitlines()
        # Recurse so USE inside skills also expands.
        expanded = _expand(
            skill_lines,
            skills_dir=skills_dir,
            chain=(*chain, skill_name),
            depth=depth + 1,
            max_depth=max_depth,
        )
        out.extend(expanded)
    return out


def _resolve_skill(name: str, *, skills_dir: Path | None) -> Path:
    """Validate a skill name, locate its file, and return the resolved path.

    Validates the name against ``_VALID_NAME_RE`` (defense-in-depth even
    though the regex in ``_USE_RE`` already enforces it — a programmer
    error elsewhere shouldn't open a path-traversal hole).
    """
    if not _VALID_NAME_RE.match(name):
        raise SkillError(
            f"Invalid skill name {name!r}: must be letters/digits/underscore/hyphen."
        )
    if skills_dir is None:
        raise SkillError(
            f"USE {name!r}: skills directory not configured. Set SKILLS_DIR "
            f"to the folder containing your .txt skill files."
        )
    if not skills_dir.exists():
        raise SkillError(
            f"USE {name!r}: skills directory {skills_dir} does not exist. "
            f"Create it with `mkdir {skills_dir}` and add a {name}.txt file, "
            f"or set SKILLS_DIR to point at an existing directory."
        )
    skill_path = skills_dir / f"{name}.txt"
    # Defense-in-depth: ensure the resolved path stays inside skills_dir.
    # ``_VALID_NAME_RE`` already prevents ``..`` / slash / backslash in
    # the name, but a hostile environment could symlink one of the
    # files; this catches that.
    try:
        skill_path.resolve().relative_to(skills_dir.resolve())
    except ValueError as exc:
        raise SkillError(
            f"USE {name!r}: resolved skill path {skill_path} is outside "
            f"the skills directory {skills_dir}."
        ) from exc
    if not skill_path.exists():
        # List a few near-matches to help the user spot a typo.
        suggestions = _suggest_skill_names(name, skills_dir)
        suggestion_text = (
            f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        )
        raise SkillError(
            f"USE {name!r}: skill file {skill_path} not found.{suggestion_text}"
        )
    return skill_path


def _suggest_skill_names(target: str, skills_dir: Path) -> list[str]:
    """Find skill names with a small edit distance to ``target``.

    Cheap O(n) similarity: lowercase prefix match. Pure UX nicety so the
    error message is useful.
    """
    out: list[str] = []
    target_lower = target.lower()
    if not skills_dir.is_dir():
        return out
    for path in sorted(skills_dir.glob("*.txt")):
        name = path.stem
        if name.lower().startswith(target_lower[:3]):
            out.append(name)
    return out[:5]


def list_skills(skills_dir: Path | None) -> list[SkillInfo]:
    """Return ``SkillInfo`` for every ``.txt`` skill in ``skills_dir``.

    Returns an empty list when the directory doesn't exist or is empty.
    Sort order is alphabetical for stable output; downstream callers can
    re-sort if they want a different presentation.
    """
    if skills_dir is None or not skills_dir.is_dir():
        return []
    out: list[SkillInfo] = []
    for path in sorted(skills_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            log.warning("could not read skill file %s: %s", path, exc)
            continue
        lines = text.splitlines()
        preview = _first_non_blank(lines)
        if len(preview) > PREVIEW_MAX_CHARS:
            preview = preview[: PREVIEW_MAX_CHARS - 1] + "…"
        out.append(
            SkillInfo(
                name=path.stem,
                path=path,
                line_count=len(lines),
                preview=preview,
            )
        )
    return out


def _first_non_blank(lines: Sequence[str]) -> str:
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # Comments make great descriptions for skills.
            return stripped.lstrip("#").strip()
        return stripped
    return ""


_SKILL_TEMPLATE = """\
# {name} — short description of what this skill does
#
# Edit this file to define the steps. Each line is one natural-language
# instruction; blank lines and lines starting with # are ignored.
#
# This skill becomes available as `USE {name}` in any tasks file.

# Example: a tiny browser sequence. Replace these with your own steps.
[browser-fast] BROWSER_GO [https://example.com]
wait for the page to load
click the first link
"""


def scaffold_skill(
    skills_dir: Path,
    name: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a starter skill file at ``<skills_dir>/<name>.txt`` and return its path.

    Creates ``skills_dir`` if it doesn't exist. Refuses to clobber an
    existing skill unless ``overwrite=True``. The template includes a
    header comment, an example ``BROWSER_GO``, and a couple of
    natural-language steps so first-time authors have something
    runnable to edit.
    """
    if not _VALID_NAME_RE.match(name):
        raise SkillError(
            f"Invalid skill name {name!r}: must be letters/digits/underscore/hyphen."
        )
    skills_dir.mkdir(parents=True, exist_ok=True)
    target = skills_dir / f"{name}.txt"
    if target.exists() and not overwrite:
        raise SkillError(
            f"Skill {name!r} already exists at {target}. Pass --overwrite-skill "
            f"to replace it, or pick a different name."
        )
    target.write_text(_SKILL_TEMPLATE.format(name=name), encoding="utf-8")
    log.info("scaffolded new skill at %s", target)
    return target


def parse_skill_triggers(skill_text: str) -> list[str]:
    """Extract auto-use trigger keywords from a skill file's header.

    Scans the first 30 lines for a ``# TRIGGERS: kw1, kw2, ...`` comment.
    The list is comma-separated; whitespace is trimmed; case is
    preserved as-written but matching is case-insensitive at lookup
    time. Returns an empty list when no TRIGGERS header is present.

    Multi-word phrases work: ``# TRIGGERS: cookie banner, gdpr``
    will match either of those phrases as substrings of a step's text.
    """
    out: list[str] = []
    for line in skill_text.splitlines()[:30]:
        match = _TRIGGERS_RE.match(line)
        if match is None:
            continue
        raw = match.group("list").strip()
        for kw in raw.split(","):
            kw_clean = kw.strip()
            if kw_clean:
                out.append(kw_clean)
        # First TRIGGERS line wins; later ones are ignored to keep
        # the contract simple. We log if we see additional ones for
        # the author's benefit.
        break
    return out


def load_skill_triggers(skills_dir: Path | None) -> dict[str, list[str]]:
    """Return ``{skill_name: [triggers]}`` for every skill that declares triggers.

    Skills without a ``# TRIGGERS:`` header are excluded from the
    returned dict — the auto-use detector iterates this mapping, so
    any skill not present in it is simply not auto-matched.
    Returns an empty dict when ``skills_dir`` is unset or empty.
    """
    out: dict[str, list[str]] = {}
    if skills_dir is None or not skills_dir.is_dir():
        return out
    for path in sorted(skills_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            log.warning("skill auto-use: could not read %s: %s", path, exc)
            continue
        triggers = parse_skill_triggers(text)
        if triggers:
            out[path.stem] = triggers
    return out


def find_skill_for_step(
    step_text: str, triggers: dict[str, list[str]]
) -> str | None:
    """Return the skill name whose triggers best match ``step_text``.

    Matching is case-insensitive substring with simple word-boundary
    awareness: a trigger ``yt`` matches ``open yt`` and ``yt music``
    but NOT ``crypto`` (the trigger is bordered by start-of-string,
    end-of-string, or non-alphanumeric on both sides). When multiple
    skills match, prefers the one whose matching trigger is *longest*
    (more specific), then falls back to alphabetical skill name for
    determinism.
    """
    if not triggers:
        return None
    text_lower = step_text.lower()
    best: tuple[int, str] | None = None  # (-trigger_len, skill_name) for sort
    for skill_name in sorted(triggers):
        for kw in triggers[skill_name]:
            kw_lower = kw.lower()
            if not kw_lower:
                continue
            if not _word_bounded_in(kw_lower, text_lower):
                continue
            score = (-len(kw_lower), skill_name)
            if best is None or score < best:
                best = score
    return best[1] if best is not None else None


def _word_bounded_in(needle: str, haystack: str) -> bool:
    """Substring match with simple word-boundary checks at both ends.

    ``needle`` is assumed already lowercased. Both endpoints of the
    match must be either at the start/end of haystack OR adjacent to
    a non-alphanumeric character. This prevents ``yt`` from matching
    inside ``crypto``.
    """
    pos = 0
    while True:
        idx = haystack.find(needle, pos)
        if idx == -1:
            return False
        before_ok = idx == 0 or not haystack[idx - 1].isalnum()
        end = idx + len(needle)
        after_ok = end == len(haystack) or not haystack[end].isalnum()
        if before_ok and after_ok:
            return True
        pos = idx + 1


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_SKILLS_DIR_NAME",
    "PREVIEW_MAX_CHARS",
    "SkillError",
    "SkillInfo",
    "expand_use_directives",
    "find_skill_for_step",
    "list_skills",
    "load_skill_triggers",
    "parse_skill_triggers",
    "scaffold_skill",
]
