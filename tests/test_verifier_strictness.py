"""Verifier-prompt strictness tests.

The verifier was previously too lenient — it would return PASS for any
screen "consistent with the goal" even when the action did nothing
visible (the YouTube-autoplay false positive that motivated this fix).

These tests assert the new prompt rules ARE in
``VERIFY_SYSTEM_PROMPT`` so a future cleanup can't accidentally
strip them. They don't actually call Gemini — that's not testable
without a live model.
"""
from __future__ import annotations

from agent.vlm import VERIFY_SYSTEM_PROMPT


class TestVerifyPromptRules:
    def test_rejects_handwavy_phrase_examples(self) -> None:
        # Each of these phrases was specifically called out in the old
        # verifier output. The new prompt MUST list them by name as
        # rejected wording so the model knows not to use them.
        for phrase in [
            "consistent with the goal",
            "appears to show",
            "could be the result of",
        ]:
            assert phrase in VERIFY_SYSTEM_PROMPT, (
                f"VERIFY_SYSTEM_PROMPT should explicitly reject the "
                f"phrase {phrase!r}; otherwise the model will keep "
                f"returning false-positive PASS verdicts."
            )

    def test_requires_positive_evidence(self) -> None:
        assert "POSITIVE EVIDENCE" in VERIFY_SYSTEM_PROMPT

    def test_youtube_autoplay_case_called_out(self) -> None:
        # The classic "YouTube homepage with autoplay running" false
        # positive — the new prompt mentions YouTube + autoplay by
        # name to anchor the model's intuition.
        assert "autoplay" in VERIFY_SYSTEM_PROMPT.lower()

    def test_press_esc_dismissal_pattern_called_out(self) -> None:
        # PRESS [esc] dismissing an unrelated overlay is the specific
        # action that produced the false-positive PASS in the user's
        # bug report. Make sure the prompt names it.
        assert "PRESS [esc]" in VERIFY_SYSTEM_PROMPT

    def test_strictness_section_exists(self) -> None:
        # The PASS/FAIL rules should be enumerated, not embedded in
        # one paragraph. Look for the bulleted PASS/FAIL headings.
        assert "PASS only when ALL of the following hold" in VERIFY_SYSTEM_PROMPT
        assert "FAIL when ANY of the following hold" in VERIFY_SYSTEM_PROMPT
