"""Regression tests for the planner system prompt (UX upgrades #2 and #3).

The planner's only steering wheel is its system prompt. Two pieces of
guidance were added to handle compound steps and "blind TYPE" failure
modes; if either is silently dropped from the prompt in a future edit
the planner will regress to the old behaviour (e.g. typing into the URL
bar, or repeating the same TYPE three times in a row).

These tests pin the contract by asserting that high-signal phrases
remain present in ``ACTION_SYSTEM_PROMPT``. They are deliberately loose
on exact wording — they only fail when a section or its core rule has
been removed entirely.
"""
from __future__ import annotations

from agent.vlm import ACTION_SYSTEM_PROMPT


def test_prompt_documents_one_command_at_a_time_for_compound_steps() -> None:
    """The planner must be told to pick ONE foundational action per call."""
    prompt = ACTION_SYSTEM_PROMPT
    # Section header must exist so the rule is grouped and findable.
    assert "ONE COMMAND AT A TIME" in prompt
    # Core directive: each call emits one command.
    assert "ONE command per call" in prompt
    # Replan-on-failure expectation must be explicit so the planner
    # doesn't interpret a FAIL as "redo the same thing harder".
    assert "verifier" in prompt.lower()
    assert "next attempt" in prompt.lower() or "next action" in prompt.lower()


def test_prompt_documents_focus_before_type_rule() -> None:
    """The planner must know that TYPE without focus is a no-op."""
    prompt = ACTION_SYSTEM_PROMPT
    # Section header.
    assert "FOCUS-BEFORE-TYPE" in prompt
    # Mechanism explanation: TYPE only sends to the focused element.
    assert "focused" in prompt.lower()
    # Recovery rule: after an "empty field" failure, CLICK first — never TYPE again.
    assert "field is still empty" in prompt or "search bar empty" in prompt
    assert "CLICK" in prompt


def test_prompt_includes_concrete_compound_step_example() -> None:
    """A worked example anchors the rule and reduces hallucinated behaviour."""
    prompt = ACTION_SYSTEM_PROMPT
    # The "type X and press Enter" example is what triggered the original
    # stuck-step report; keep a worked example for it.
    assert "Enter" in prompt
    assert "1st attempt" in prompt
    assert "2nd attempt" in prompt
