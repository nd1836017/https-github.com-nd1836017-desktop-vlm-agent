"""Smart step-skip diagnosis: escalate failed steps before halting.

The classic agent loop halts as soon as ``run_step`` exhausts its
replan budget. That's safe but punishing: in practice the most common
"stuck" situation is a step whose prerequisite is already on screen
from a previous step (e.g. ``press enter`` to open Chrome, but Chrome
is already open, so ``press enter`` does nothing visible and the
verifier rightly says FAIL).

Smart step-skip layers a 2-tier escalation between "replan budget
exhausted" and "halt", driven entirely from VLM observations of the
current screen:

    Tier 2: targeted yes/no diagnosis. Three bracketed yes/no checks
            in one call (or three sequential calls if the model is
            JSON-shy):
              - is_already_done     — has the step's goal already
                                       happened?
              - is_previous_state   — is the previous step's outcome
                                       still on screen, suggesting the
                                       planner picked an action that
                                       doesn't move things forward?
              - is_unrelated_screen — is the current screen entirely
                                       off-track (wrong app)?
            If ``is_already_done``, skip the current step.

    Tier 3: open-ended classify-and-jump. Two free-form VLM calls:
              1. "Describe this screen in one sentence."
              2. Given the task list and that description, "which
                  numbered step does this most plausibly match? Or
                  'none'?"
            If the matched step is the *current* step → treat as
            already-done and skip.
            If the matched step is a *future* step → jump ahead
            (intermediate steps are recorded as "auto-skipped: looks
            like step N is already on screen").
            If 'none' or a *past* step (likely confusion) → halt as
            before.

This module is intentionally small and side-effect free so it can be
unit-tested without spinning up a real run.

Manual ``[SKIPPABLE]`` annotation on a tasks file uses just the
"is_already_done" check from Tier 2, but runs it BEFORE the planner
ever fires — preempting the replan budget burn for steps the user
knows are commonly redundant (e.g. dismissing a cookie banner that's
not always shown).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from PIL.Image import Image

    from .vlm import GeminiClient

log = logging.getLogger(__name__)


SkipDecisionKind = Literal["continue", "skip_current", "jump_to", "halt"]


@dataclass(frozen=True)
class SkipDecision:
    """Outcome of a smart-skip escalation.

    ``kind`` is one of:
        ``continue``     — diagnosis ran but had no opinion; caller
                           should fall through to its existing halt
                           or replan path.
        ``skip_current`` — current step's goal is already on screen;
                           record a synthetic PASS and advance one.
        ``jump_to``      — current screen matches a *future* step;
                           ``target_idx`` is the 1-indexed step the
                           run loop should jump to. Steps between
                           current+1 and target_idx are recorded as
                           "auto-skipped".
        ``halt``         — diagnosis explicitly recommends halting
                           (off-track / unrelated screen with no
                           future-step match). Caller halts as
                           before.

    ``reason`` is a short human-readable explanation suitable for
    artifact summaries / log messages.
    """

    kind: SkipDecisionKind
    reason: str
    target_idx: int | None = None
    tier: int = 0
    classify_description: str | None = None


def _safe_check_condition(
    vlm: GeminiClient, condition: str, screenshot: Image, label: str
) -> bool:
    """Wrap ``vlm.check_condition`` so transient errors don't crash diagnosis.

    Smart-skip is best-effort. If the VLM call itself fails, we want
    the caller to fall through to its existing halt path rather than
    bubbling a new exception up the run loop.
    """
    try:
        return bool(vlm.check_condition(condition, screenshot))
    except Exception as exc:  # noqa: BLE001 — diagnosis must not crash run
        log.warning("skip_diagnosis %s: check_condition raised %s; treating as False.", label, exc)
        return False


def diagnose_already_done(
    vlm: GeminiClient,
    step_text: str,
    screenshot: Image,
) -> bool:
    """Tier-2 question 1: is the step's goal already visible on screen?

    Used by both the auto-escalation path and the manual ``[SKIPPABLE]``
    pre-flight check.
    """
    framed = (
        f"the apparent goal of the step \"{step_text}\" already "
        f"appears to have been achieved on the current screen"
    )
    return _safe_check_condition(vlm, framed, screenshot, label="already_done")


def diagnose_unrelated_screen(
    vlm: GeminiClient,
    step_text: str,
    screenshot: Image,
) -> bool:
    """Tier-2 question 3: is the screen completely off-track?

    True when the current screen looks unrelated to the active step
    (wrong app, wrong page, lost focus). Caller may use this to
    decide between "continue retry on the same step" and "halt early
    so the human can recover".
    """
    framed = (
        f"the current screen is COMPLETELY UNRELATED to the step \"{step_text}\" "
        f"(the agent appears to be on the wrong app or wrong page entirely)"
    )
    return _safe_check_condition(vlm, framed, screenshot, label="unrelated_screen")


def classify_and_match(
    vlm: GeminiClient,
    step_texts: list[str],
    current_idx: int,
    screenshot: Image,
) -> tuple[str, int | None]:
    """Tier-3: ask the VLM what the screen is and which step matches.

    Returns ``(description, matched_step_idx)``. ``matched_step_idx``
    is 1-indexed and may be ``None`` when the VLM can't identify a
    matching step.

    Implementation note: rather than adding a third specialised method
    on ``GeminiClient``, we lean on ``check_condition`` for each
    candidate step. This is two VLM calls per candidate but is robust
    against the model's tendency to invent step numbers — every
    "match" is grounded in a yes/no PASS verdict on a concrete
    statement. We bound the search to the window
    ``[current_idx, len(step_texts)]`` because matching a *past* step
    is almost always a sign of confusion (the agent has already moved
    forward), and we never want to silently regress.
    """
    # Step 1: free-form description (used purely for logs / artifact
    # bundles — not parsed). We don't need the model's exact wording;
    # we just want to capture it for human review when we do skip.
    description = ""
    try:
        description = _classify_screen(vlm, screenshot)
    except Exception as exc:  # noqa: BLE001
        log.warning("classify_screen failed: %s", exc)

    # Step 2: yes/no per future step starting from current. Stop at
    # the first match — the run loop wants the *earliest* matching
    # step, not the last.
    for candidate_idx in range(current_idx, len(step_texts) + 1):
        step_text = step_texts[candidate_idx - 1]
        framed = (
            f"the current screen most plausibly corresponds to the step "
            f"\"{step_text}\" (i.e. that step's expected outcome / "
            f"prerequisite state is what we see now)"
        )
        if _safe_check_condition(
            vlm, framed, screenshot, label=f"match_step_{candidate_idx}"
        ):
            return description, candidate_idx

    return description, None


def _classify_screen(vlm: GeminiClient, screenshot: Image) -> str:
    """Free-form: ask the VLM to describe the current screen briefly.

    Best-effort. Returns an empty string when the VLM is unavailable
    or returns nothing parseable. The description is logged + written
    to artifacts so a human reviewing a skip can see what the agent
    was looking at.
    """
    # Use the existing check_condition path with a "describe…" prompt
    # is awkward (it's yes/no), so we build a minimal free-form call
    # via the same generate_content as verify. We fall back to ""
    # rather than raising — diagnosis is advisory.
    if not hasattr(vlm, "describe_screen"):
        return ""
    try:
        text = vlm.describe_screen(screenshot)
    except Exception as exc:  # noqa: BLE001
        log.warning("describe_screen raised: %s", exc)
        return ""
    return (text or "").strip()


def diagnose_step_failure(
    vlm: GeminiClient,
    step_texts: list[str],
    current_idx: int,
    screenshot: Image,
    *,
    max_tier: int = 3,
) -> SkipDecision:
    """Run the 2-tier escalation. ``current_idx`` is 1-indexed.

    Returns a ``SkipDecision``. ``max_tier`` caps how aggressive the
    escalation gets — useful for the manual ``[SKIPPABLE]``
    pre-flight check (which only wants Tier 2 question 1).
    """
    if max_tier < 2:
        return SkipDecision(kind="continue", reason="smart-skip disabled")

    step_text = step_texts[current_idx - 1] if 1 <= current_idx <= len(step_texts) else ""

    # Tier 2 — targeted yes/no diagnosis.
    if diagnose_already_done(vlm, step_text, screenshot):
        log.info(
            "smart-skip Tier 2: step %d (%r) goal already on screen; skipping.",
            current_idx,
            step_text,
        )
        return SkipDecision(
            kind="skip_current",
            reason=(
                f"step {current_idx} appears already done "
                f"(goal already visible on screen)"
            ),
            tier=2,
        )

    if max_tier < 3:
        # Tier 2 only — fall through to halt without trying classify.
        return SkipDecision(
            kind="continue", reason="Tier 2 found no skip; max_tier=2"
        )

    # Tier 3 — open-ended classify and match.
    description, matched_idx = classify_and_match(
        vlm, step_texts, current_idx, screenshot
    )
    if matched_idx is None:
        # No future step matches. Try the off-track check to decide
        # halt-vs-continue. We default to ``continue`` (fall through
        # to existing halt path) when even this signal is unclear.
        if diagnose_unrelated_screen(vlm, step_text, screenshot):
            return SkipDecision(
                kind="halt",
                reason=(
                    f"smart-skip Tier 3: screen unrelated to step {current_idx} "
                    f"and no future step matches"
                ),
                tier=3,
                classify_description=description,
            )
        return SkipDecision(
            kind="continue",
            reason=f"smart-skip Tier 3: no matching step found (description: {description!r})",
            tier=3,
            classify_description=description,
        )

    if matched_idx == current_idx:
        return SkipDecision(
            kind="skip_current",
            reason=(
                f"smart-skip Tier 3: screen matches current step "
                f"(description: {description!r}); skipping as already-done"
            ),
            tier=3,
            classify_description=description,
        )
    if matched_idx > current_idx:
        return SkipDecision(
            kind="jump_to",
            target_idx=matched_idx,
            reason=(
                f"smart-skip Tier 3: screen matches step {matched_idx}; "
                f"jumping ahead from step {current_idx} "
                f"(description: {description!r})"
            ),
            tier=3,
            classify_description=description,
        )

    # matched_idx < current_idx is suspicious — the agent has already
    # moved past that step. Don't regress; fall through to halt.
    return SkipDecision(
        kind="continue",
        reason=(
            f"smart-skip Tier 3: matched a past step ({matched_idx}); "
            f"refusing to regress, falling through to halt"
        ),
        tier=3,
        classify_description=description,
    )


__all__ = [
    "SkipDecision",
    "SkipDecisionKind",
    "diagnose_already_done",
    "diagnose_step_failure",
    "diagnose_unrelated_screen",
    "classify_and_match",
]
