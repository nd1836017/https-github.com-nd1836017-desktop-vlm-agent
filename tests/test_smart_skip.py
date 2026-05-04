"""Smart step-skip diagnosis tests.

Covers:
  - Tier-2 diagnose_already_done (yes / no / VLM exception).
  - Tier-3 classify_and_match (jump to future step / no match / past
    step regression refused).
  - Top-level diagnose_step_failure orchestration (skip / jump / halt
    / continue).
  - Manual ``[skippable]`` annotation parsing in tasks_loader.
"""
from __future__ import annotations

from PIL import Image

from agent.skip_diagnosis import (
    SkipDecision,
    classify_and_match,
    diagnose_already_done,
    diagnose_step_failure,
    diagnose_unrelated_screen,
)
from agent.tasks_loader import _parse_step_annotations

# -----------------------------------------------------------------------------
# Test doubles
# -----------------------------------------------------------------------------

class FakeVLM:
    """Records every check_condition / describe_screen call.

    ``conditions`` is a list of (substring → return_value). The first
    entry whose substring is contained in the prompt fires. Default
    is False so an unmocked check returns "no".
    """

    def __init__(
        self,
        conditions: list[tuple[str, bool]] | None = None,
        description: str = "Stubbed description.",
        raises: bool = False,
    ):
        self._conditions = list(conditions or [])
        self._description = description
        self._raises = raises
        self.condition_calls: list[str] = []
        self.describe_calls: int = 0

    def check_condition(self, condition_text: str, screenshot):
        self.condition_calls.append(condition_text)
        if self._raises:
            raise RuntimeError("transport error")
        for needle, value in self._conditions:
            if needle in condition_text:
                return value
        return False

    def describe_screen(self, screenshot):
        self.describe_calls += 1
        if self._raises:
            raise RuntimeError("transport error")
        return self._description


def _img() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(0, 0, 0))


# -----------------------------------------------------------------------------
# Tier 2 — single-question helpers
# -----------------------------------------------------------------------------

def test_diagnose_already_done_returns_true_when_vlm_says_yes():
    vlm = FakeVLM(conditions=[("already", True)])
    assert diagnose_already_done(vlm, "press enter to open chrome", _img()) is True
    assert any("already" in call for call in vlm.condition_calls)


def test_diagnose_already_done_returns_false_when_vlm_says_no():
    vlm = FakeVLM(conditions=[])
    assert diagnose_already_done(vlm, "click login", _img()) is False


def test_diagnose_already_done_treats_vlm_exception_as_false():
    """A failing VLM must NOT crash the run loop."""
    vlm = FakeVLM(raises=True)
    assert diagnose_already_done(vlm, "click login", _img()) is False


def test_diagnose_unrelated_screen_returns_true_when_off_track():
    vlm = FakeVLM(conditions=[("UNRELATED", True)])
    assert diagnose_unrelated_screen(vlm, "click submit", _img()) is True


# -----------------------------------------------------------------------------
# Tier 3 — classify_and_match
# -----------------------------------------------------------------------------

def test_classify_and_match_returns_first_matching_future_step():
    # Steps 1-3; current is step 1. Match step 3 only — the condition
    # prompt embeds the step text, so we needle on a substring of step 3.
    vlm = FakeVLM(conditions=[("click first video", True)])
    description, matched = classify_and_match(
        vlm,
        ["open chrome", "type query", "click first video"],
        current_idx=1,
        screenshot=_img(),
    )
    # We don't pin description content, but we DO check that the
    # matcher walked from current onward and stopped at step 3.
    assert matched == 3


def test_classify_and_match_returns_none_when_no_step_matches():
    vlm = FakeVLM(conditions=[])
    description, matched = classify_and_match(
        vlm,
        ["open chrome", "click login", "type password"],
        current_idx=1,
        screenshot=_img(),
    )
    assert matched is None


def test_classify_and_match_starts_search_from_current_idx_not_step_one():
    """Past-step regressions must not be returned even when the prompt
    text would otherwise match."""
    # Step 1 prompt "open chrome" would match — but current is step 3,
    # so the search window is steps 3 onwards. We expect no match.
    vlm = FakeVLM(conditions=[("open chrome", True)])
    description, matched = classify_and_match(
        vlm,
        ["open chrome", "type query", "click first video"],
        current_idx=3,
        screenshot=_img(),
    )
    assert matched is None


# -----------------------------------------------------------------------------
# Top-level diagnose_step_failure
# -----------------------------------------------------------------------------

def test_diagnose_step_failure_already_done_returns_skip_current():
    vlm = FakeVLM(conditions=[("already", True)])
    decision = diagnose_step_failure(
        vlm,
        ["open chrome", "press enter"],
        current_idx=2,
        screenshot=_img(),
    )
    assert decision.kind == "skip_current"
    assert decision.tier == 2


def test_diagnose_step_failure_jump_to_future_step():
    """Goal not already done, but screen matches a future step."""
    vlm = FakeVLM(
        conditions=[
            # Tier 2 question 1 — "already done?" → False
            ("already", False),
            # Tier 3 step-match — first match at step 3.
            ("click first video", True),
        ]
    )
    decision = diagnose_step_failure(
        vlm,
        ["open chrome", "type query", "click first video"],
        current_idx=2,
        screenshot=_img(),
    )
    assert decision.kind == "jump_to"
    assert decision.target_idx == 3
    assert decision.tier == 3


def test_diagnose_step_failure_jump_to_current_becomes_skip_current():
    """If the matched step is the current step, it's already-done."""
    vlm = FakeVLM(
        conditions=[
            ("already", False),
            # Step 2 (the current one) matches the screen.
            ("type query", True),
        ]
    )
    decision = diagnose_step_failure(
        vlm,
        ["open chrome", "type query", "click first video"],
        current_idx=2,
        screenshot=_img(),
    )
    assert decision.kind == "skip_current"
    assert decision.tier == 3


def test_diagnose_step_failure_unrelated_screen_returns_halt():
    vlm = FakeVLM(
        conditions=[
            ("already", False),
            # No step matches.
            ("UNRELATED", True),
        ]
    )
    decision = diagnose_step_failure(
        vlm,
        ["open chrome", "type query"],
        current_idx=1,
        screenshot=_img(),
    )
    assert decision.kind == "halt"
    assert decision.tier == 3


def test_diagnose_step_failure_no_match_no_unrelated_falls_through():
    vlm = FakeVLM(conditions=[])
    decision = diagnose_step_failure(
        vlm,
        ["open chrome", "type query"],
        current_idx=1,
        screenshot=_img(),
    )
    assert decision.kind == "continue"


def test_diagnose_step_failure_max_tier_one_returns_continue():
    """When max_tier=1, escalation is a no-op (legacy halt path)."""
    vlm = FakeVLM(conditions=[("already", True)])
    decision = diagnose_step_failure(
        vlm,
        ["open chrome", "press enter"],
        current_idx=2,
        screenshot=_img(),
        max_tier=1,
    )
    assert decision.kind == "continue"
    # No VLM calls at all when max_tier<2.
    assert vlm.condition_calls == []


def test_diagnose_step_failure_max_tier_two_skips_tier3():
    """max_tier=2 runs only the already-done check."""
    vlm = FakeVLM(conditions=[])  # nothing matches
    decision = diagnose_step_failure(
        vlm,
        ["open chrome", "press enter"],
        current_idx=2,
        screenshot=_img(),
        max_tier=2,
    )
    assert decision.kind == "continue"
    # Exactly one condition call: the Tier 2 already-done check.
    assert len(vlm.condition_calls) == 1
    assert "already" in vlm.condition_calls[0]


# -----------------------------------------------------------------------------
# Manual [SKIPPABLE] annotation parsing
# -----------------------------------------------------------------------------

def test_skippable_annotation_alone():
    hint, skippable, text = _parse_step_annotations("[skippable] press enter")
    assert hint is None
    assert skippable is True
    assert text == "press enter"


def test_skippable_case_insensitive():
    hint, skippable, text = _parse_step_annotations("[SKIPPABLE] click login")
    assert skippable is True
    assert text == "click login"


def test_skippable_with_routing_in_either_order():
    a = _parse_step_annotations("[skippable] [browser] open youtube")
    b = _parse_step_annotations("[browser] [skippable] open youtube")
    # Both produce skippable=True and a routing hint, with text stripped.
    assert a[1] is True
    assert b[1] is True
    assert a[0] is not None
    assert b[0] is not None
    assert a[2] == "open youtube"
    assert b[2] == "open youtube"


def test_no_annotations_returns_line_unchanged():
    hint, skippable, text = _parse_step_annotations("press enter")
    assert hint is None
    assert skippable is False
    assert text == "press enter"


def test_unknown_bracket_prefix_left_intact():
    """``[admin]`` is not a known tag, so the line is left untouched."""
    hint, skippable, text = _parse_step_annotations("[admin] click submit")
    assert hint is None
    assert skippable is False
    assert text == "[admin] click submit"


# -----------------------------------------------------------------------------
# SkipDecision dataclass shape
# -----------------------------------------------------------------------------

def test_skip_decision_defaults():
    d = SkipDecision(kind="continue", reason="no-op")
    assert d.target_idx is None
    assert d.tier == 0
    assert d.classify_description is None
