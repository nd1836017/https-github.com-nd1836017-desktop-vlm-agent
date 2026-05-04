"""Auto-use of skills based on ``# TRIGGERS:`` keyword matching.

Covers:
  - parse_skill_triggers — reads the ``# TRIGGERS:`` header.
  - load_skill_triggers — aggregates per-skill triggers from a directory.
  - find_skill_for_step — keyword matching with word boundaries +
    longest-match preference.
  - apply_skill_auto_use — end-to-end TaskStep replacement.
"""
from __future__ import annotations

from pathlib import Path

from agent.skills import (
    find_skill_for_step,
    load_skill_triggers,
    parse_skill_triggers,
)
from agent.task_router import RoutingComplexity, RoutingHint
from agent.tasks_loader import TaskStep, apply_skill_auto_use


def _write(tmp_path: Path, name: str, content: str) -> Path:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(exist_ok=True)
    (skills_dir / f"{name}.txt").write_text(content, encoding="utf-8")
    return skills_dir


# -----------------------------------------------------------------------------
# parse_skill_triggers
# -----------------------------------------------------------------------------

def test_parse_skill_triggers_basic():
    text = "# my_skill\n# TRIGGERS: foo, bar, baz\nclick foo"
    assert parse_skill_triggers(text) == ["foo", "bar", "baz"]


def test_parse_skill_triggers_empty_when_absent():
    text = "# my_skill\nclick foo"
    assert parse_skill_triggers(text) == []


def test_parse_skill_triggers_case_insensitive_header():
    text = "# my_skill\n# triggers: foo\nclick foo"
    assert parse_skill_triggers(text) == ["foo"]


def test_parse_skill_triggers_strips_whitespace():
    text = "# TRIGGERS:    foo  ,   bar baz   \nclick foo"
    assert parse_skill_triggers(text) == ["foo", "bar baz"]


def test_parse_skill_triggers_first_block_wins():
    text = (
        "# TRIGGERS: a, b\n"
        "# TRIGGERS: c, d\n"
        "click foo"
    )
    assert parse_skill_triggers(text) == ["a", "b"]


def test_parse_skill_triggers_only_scans_first_30_lines():
    """A TRIGGERS line buried below the header isn't read."""
    text = "\n".join(["# leading"] * 50) + "\n# TRIGGERS: late\n"
    assert parse_skill_triggers(text) == []


# -----------------------------------------------------------------------------
# load_skill_triggers
# -----------------------------------------------------------------------------

def test_load_skill_triggers_returns_only_trigger_skills(tmp_path):
    _write(tmp_path, "alpha", "# TRIGGERS: foo, bar\nclick foo")
    _write(tmp_path, "beta", "click bar")  # no TRIGGERS header
    skills_dir = tmp_path / "skills"
    triggers = load_skill_triggers(skills_dir)
    assert triggers == {"alpha": ["foo", "bar"]}


def test_load_skill_triggers_empty_when_dir_missing(tmp_path):
    assert load_skill_triggers(tmp_path / "does_not_exist") == {}


def test_load_skill_triggers_empty_when_dir_none():
    assert load_skill_triggers(None) == {}


# -----------------------------------------------------------------------------
# find_skill_for_step — keyword matching
# -----------------------------------------------------------------------------

def test_find_skill_for_step_simple_match():
    triggers = {"open_youtube": ["youtube", "yt"]}
    assert find_skill_for_step("open youtube", triggers) == "open_youtube"


def test_find_skill_for_step_short_alias_word_bounded():
    """``yt`` matches ``open yt`` but NOT inside ``crypto``."""
    triggers = {"open_youtube": ["yt"]}
    assert find_skill_for_step("open yt", triggers) == "open_youtube"
    assert find_skill_for_step("crypto", triggers) is None
    assert find_skill_for_step("decrypt", triggers) is None


def test_find_skill_for_step_no_match_returns_none():
    triggers = {"open_gmail": ["gmail", "inbox"]}
    assert find_skill_for_step("open spotify", triggers) is None


def test_find_skill_for_step_longer_trigger_wins():
    """When two skills could match, the one with the LONGER trigger wins."""
    triggers = {
        "general": ["yt"],
        "specific": ["yt music"],
    }
    # "open yt music" matches both; "yt music" is more specific.
    assert find_skill_for_step("open yt music", triggers) == "specific"


def test_find_skill_for_step_alphabetical_tiebreaker_for_equal_length():
    triggers = {
        "alpha_skill": ["abc"],
        "beta_skill": ["xyz"],
    }
    assert find_skill_for_step("abc xyz", triggers) == "alpha_skill"


def test_find_skill_for_step_case_insensitive():
    triggers = {"open_youtube": ["YouTube"]}
    assert find_skill_for_step("open YOUTUBE", triggers) == "open_youtube"


def test_find_skill_for_step_empty_triggers_returns_none():
    assert find_skill_for_step("anything", {}) is None


# -----------------------------------------------------------------------------
# apply_skill_auto_use
# -----------------------------------------------------------------------------

def test_apply_skill_auto_use_replaces_matched_step(tmp_path):
    skills_dir = _write(
        tmp_path,
        "open_youtube",
        "# TRIGGERS: youtube, yt\nBROWSER_GO [https://www.youtube.com]\n",
    )
    steps = [
        TaskStep(text="open youtube"),
        TaskStep(text="search for justin bieber"),
    ]
    result = apply_skill_auto_use(
        steps, skills_dir=skills_dir, enabled=True
    )
    # The matched step is REPLACED by the skill's content (1 substep
    # here — BROWSER_GO). The unmatched step is kept as-is.
    texts = [s.text for s in result]
    assert "BROWSER_GO [https://www.youtube.com]" in texts
    assert "search for justin bieber" in texts
    # The original "open youtube" step is gone (replaced).
    assert "open youtube" not in texts


def test_apply_skill_auto_use_disabled_returns_unchanged(tmp_path):
    skills_dir = _write(
        tmp_path,
        "open_youtube",
        "# TRIGGERS: youtube\nBROWSER_GO [https://www.youtube.com]\n",
    )
    steps = [TaskStep(text="open youtube")]
    result = apply_skill_auto_use(
        steps, skills_dir=skills_dir, enabled=False
    )
    assert [s.text for s in result] == ["open youtube"]


def test_apply_skill_auto_use_no_triggers_returns_unchanged(tmp_path):
    """A skill without a TRIGGERS header is never auto-matched."""
    skills_dir = _write(
        tmp_path,
        "open_youtube",
        "BROWSER_GO [https://www.youtube.com]\n",  # no TRIGGERS line
    )
    steps = [TaskStep(text="open youtube")]
    result = apply_skill_auto_use(
        steps, skills_dir=skills_dir, enabled=True
    )
    assert [s.text for s in result] == ["open youtube"]


def test_apply_skill_auto_use_skips_control_directives(tmp_path):
    skills_dir = _write(
        tmp_path,
        "alpha",
        "# TRIGGERS: dummy\nclick something\n",
    )
    # Manually-built control-flow steps. apply_skill_auto_use must
    # NEVER expand a skill in their place — the run loop interprets
    # them directly and an expansion would break the IF/ELSE pairing.
    steps = [
        TaskStep(text="dummy condition", control_kind="if_begin", block_id=1),
        TaskStep(text="dummy", control_kind="if_end", block_id=1),
    ]
    result = apply_skill_auto_use(
        steps, skills_dir=skills_dir, enabled=True
    )
    assert len(result) == 2
    assert all(s.control_kind is not None for s in result)


def test_apply_skill_auto_use_skips_steps_with_manual_routing(tmp_path):
    """Manual ``[browser]`` etc. routing means the user has already
    declared an intent; auto-skill must not override it."""
    skills_dir = _write(
        tmp_path,
        "open_youtube",
        "# TRIGGERS: youtube\nBROWSER_GO [https://www.youtube.com]\n",
    )
    manual_hint = RoutingHint(
        complexity=RoutingComplexity.BROWSER_FAST,
        source="manual",
        reasoning="user-tagged",
    )
    steps = [TaskStep(text="open youtube", routing_hint=manual_hint)]
    result = apply_skill_auto_use(
        steps, skills_dir=skills_dir, enabled=True
    )
    # Step kept as-is, manual routing intact.
    assert len(result) == 1
    assert result[0].text == "open youtube"
    assert result[0].routing_hint is manual_hint


def test_apply_skill_auto_use_single_pass_no_recursion(tmp_path):
    """A skill's own content is NOT re-checked against triggers.

    If skill A's content contains text that matches skill A's trigger,
    we'd loop forever without single-pass discipline.
    """
    skills_dir = _write(
        tmp_path,
        "alpha",
        "# TRIGGERS: foo\nclick foo button\n",
    )
    steps = [TaskStep(text="press the foo")]
    result = apply_skill_auto_use(
        steps, skills_dir=skills_dir, enabled=True
    )
    # The expanded substep "click foo button" contains "foo" but is
    # NOT re-expanded.
    texts = [s.text for s in result]
    assert texts == ["click foo button"]


def test_apply_skill_auto_use_passes_through_when_skills_dir_missing():
    steps = [TaskStep(text="open youtube")]
    result = apply_skill_auto_use(steps, skills_dir=None, enabled=True)
    assert [s.text for s in result] == ["open youtube"]


def test_apply_skill_auto_use_user_example(tmp_path):
    """The exact scenario the user described:

    "play justin bieber on yt" decomposed into atomic substeps;
    the "open yt" / "open youtube" step matches the open_youtube skill.
    """
    skills_dir = _write(
        tmp_path,
        "open_youtube",
        "# TRIGGERS: youtube, yt\nBROWSER_GO [https://www.youtube.com]\n",
    )
    # Simulate the post-decomposer state.
    steps = [
        TaskStep(text="open yt"),
        TaskStep(text="search for justin bieber"),
        TaskStep(text="click the first video result"),
    ]
    result = apply_skill_auto_use(
        steps, skills_dir=skills_dir, enabled=True
    )
    texts = [s.text for s in result]
    # Skill expanded: BROWSER_GO appears.
    assert "BROWSER_GO [https://www.youtube.com]" in texts
    # Subsequent steps untouched.
    assert "search for justin bieber" in texts
    assert "click the first video result" in texts
