"""Tests for the skill library: USE expansion, listing, scaffolding."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.skills import (
    SkillError,
    expand_use_directives,
    list_skills,
    scaffold_skill,
)
from agent.tasks_loader import TasksLoadError, load_steps

# -----------------------------------------------------------------------------
# expand_use_directives — happy paths


def test_use_directive_inlines_skill_content(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "open_chrome.txt").write_text("open chrome\nfocus the address bar\n")

    out = expand_use_directives(
        ["before", "USE open_chrome", "after"], skills_dir=skills
    )
    assert out == ["before", "open chrome", "focus the address bar", "after"]


def test_use_is_case_insensitive(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "x.txt").write_text("hello\n")
    out = expand_use_directives(["use x", "Use x", "USE x"], skills_dir=skills)
    assert out == ["hello", "hello", "hello"]


def test_use_in_skill_recurses(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "a.txt").write_text("USE b\nstep from a\n")
    (skills / "b.txt").write_text("step from b\n")

    out = expand_use_directives(["USE a"], skills_dir=skills)
    assert out == ["step from b", "step from a"]


def test_blank_lines_and_comments_preserved_through_expansion(tmp_path: Path):
    """Comments / blanks still go to the loader for normal handling."""
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "s.txt").write_text("# a comment\n\nrun this\n")
    out = expand_use_directives(["USE s"], skills_dir=skills)
    assert out == ["# a comment", "", "run this"]


# -----------------------------------------------------------------------------
# expand_use_directives — error paths


def test_use_with_no_skills_dir_raises(tmp_path: Path):
    with pytest.raises(SkillError, match="not configured"):
        expand_use_directives(["USE x"], skills_dir=None)


def test_use_with_missing_skills_dir_raises(tmp_path: Path):
    with pytest.raises(SkillError, match="does not exist"):
        expand_use_directives(["USE x"], skills_dir=tmp_path / "no-such-dir")


def test_use_with_missing_skill_file_raises(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "login.txt").write_text("step\n")
    with pytest.raises(SkillError, match="not found"):
        expand_use_directives(["USE logn"], skills_dir=skills)


def test_use_typo_suggests_near_match(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "login_to_gmail.txt").write_text("step\n")
    with pytest.raises(SkillError, match="login_to_gmail"):
        expand_use_directives(["USE login"], skills_dir=skills)


def test_cycle_detected(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "a.txt").write_text("USE b\n")
    (skills / "b.txt").write_text("USE a\n")
    with pytest.raises(SkillError, match="cyclic"):
        expand_use_directives(["USE a"], skills_dir=skills)


def test_self_reference_detected(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "a.txt").write_text("USE a\n")
    with pytest.raises(SkillError, match="cyclic"):
        expand_use_directives(["USE a"], skills_dir=skills)


def test_max_depth_exceeded(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    # Linear chain a -> b -> c -> ...; depth=2 allows only 2 hops.
    (skills / "a.txt").write_text("USE b\n")
    (skills / "b.txt").write_text("USE c\n")
    (skills / "c.txt").write_text("USE d\n")
    (skills / "d.txt").write_text("step\n")
    with pytest.raises(SkillError, match="max depth"):
        expand_use_directives(["USE a"], skills_dir=skills, max_depth=2)


# -----------------------------------------------------------------------------
# Integration with load_steps


def test_load_steps_inlines_skill(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "greet.txt").write_text("say hi\nwave\n")
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("USE greet\nthen finish\n")

    steps = load_steps(tasks, skills_dir=skills)
    assert [s.text for s in steps] == ["say hi", "wave", "then finish"]


def test_load_steps_use_without_skills_dir_raises(tmp_path: Path):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("USE greet\n")
    with pytest.raises(TasksLoadError, match="not configured"):
        load_steps(tasks, skills_dir=None)


# -----------------------------------------------------------------------------
# list_skills + scaffold_skill


def test_list_skills_returns_sorted_infos(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "zeta.txt").write_text("# zeta description\nrun\n")
    (skills / "alpha.txt").write_text("plain first line\nstuff\n")

    infos = list_skills(skills)
    assert [i.name for i in infos] == ["alpha", "zeta"]
    assert infos[0].line_count == 2
    assert infos[0].preview == "plain first line"
    assert infos[1].preview == "zeta description"


def test_list_skills_handles_empty_dir(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    assert list_skills(skills) == []


def test_list_skills_handles_missing_dir(tmp_path: Path):
    assert list_skills(tmp_path / "nope") == []


def test_scaffold_creates_starter_file(tmp_path: Path):
    skills = tmp_path / "skills"
    out = scaffold_skill(skills, "demo")
    assert out == skills / "demo.txt"
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "demo" in body
    assert "BROWSER_GO" in body  # template uses a real primitive


def test_scaffold_refuses_to_overwrite(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    target = skills / "x.txt"
    target.write_text("existing")
    with pytest.raises(SkillError, match="already exists"):
        scaffold_skill(skills, "x")
    # And without --overwrite-skill the original is preserved.
    assert target.read_text() == "existing"


def test_scaffold_overwrites_when_requested(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "x.txt").write_text("existing")
    scaffold_skill(skills, "x", overwrite=True)
    assert "BROWSER_GO" in (skills / "x.txt").read_text()


def test_scaffold_rejects_invalid_name(tmp_path: Path):
    with pytest.raises(SkillError, match="Invalid skill name"):
        scaffold_skill(tmp_path, "../etc/passwd")
