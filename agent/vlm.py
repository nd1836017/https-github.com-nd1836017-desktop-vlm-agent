"""Gemini Flash client — action planning + verification calls.

Uses the modern `google-genai` SDK (`pip install google-genai`).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from google import genai
from google.genai import types
from PIL import Image

log = logging.getLogger(__name__)


ACTION_SYSTEM_PROMPT = """You are a desktop automation agent. You see a screenshot of the user's screen and a single natural-language step to perform.

RESPOND WITH EXACTLY ONE COMMAND on its own line, chosen from:

    CLICK [X,Y]      — click at normalized coordinates. X and Y are integers in [0, 1000] where (0,0) is the top-left of the screen and (1000,1000) is the bottom-right. Do NOT use pixel coordinates.
    PRESS [KEY]      — press a single key or hotkey. Examples: PRESS [win], PRESS [enter], PRESS [esc], PRESS [ctrl+c], PRESS [alt+tab].
    TYPE [TEXT]      — type literal text. Example: TYPE [hello world].

Rules:
- Output ONLY the command, wrapped in square brackets. No prose, no markdown, no explanation.
- Use the 0-1000 normalized grid for coordinates — never pixels.
- If the step has already been completed, still emit a single no-op friendly command (e.g. PRESS [esc]) — never output nothing.
- You may be shown a summary of previous steps and a previous-attempt failure reason. Use them to avoid repeating mistakes and to pick a DIFFERENT action when replanning after a failure.
"""


VERIFY_SYSTEM_PROMPT = """You are a verification assistant for a desktop automation agent.

You are given:
  1. A screenshot of the screen AFTER an action was performed.
  2. The natural-language goal that the action was meant to achieve.

Respond with EXACTLY ONE LINE in this format:

    VERDICT: PASS — <short reason>
    VERDICT: FAIL — <short reason explaining what actually happened>

Use PASS if the screen state is consistent with the goal having been achieved (or is clearly in progress, e.g. a menu is opening). Use FAIL if something clearly went wrong — e.g. the wrong application opened, an error dialog appeared, or nothing happened when something should have.
Do not include any other text.
"""


REFINE_SYSTEM_PROMPT = """You are a refinement assistant for a desktop automation agent.

You see a CROPPED region of the user's screen and a natural-language step describing what the agent wants to click.

List every plausible click target in this crop that matches the step. For EACH target, output ONE line in the exact format:

    CLICK [X,Y]

where X and Y are integers in [0, 1000] on THIS CROP's own 0-1000 grid (not the full screen). (0,0) is the top-left of this crop. (1000,1000) is the bottom-right.

Rules:
- If NOTHING in this crop plausibly matches the step, respond with the single line: NONE
- Do NOT output anything else — no prose, no numbering, no code fences.
- Prefer precise centers of buttons/links/inputs over their edges.
- If multiple things could match, list them all (at most 5).
"""


DISAMBIG_SYSTEM_PROMPT = """You are a disambiguation assistant for a desktop automation agent.

You see a screenshot annotated with numbered red rectangles labeled 1, 2, 3, ... Each rectangle marks a candidate click target. You also see a natural-language step describing what the user wants to click.

Pick the single number whose rectangle best matches the step. Respond with EXACTLY one line:

    PICK [N]

where N is the 1-based number of the best rectangle. If NONE of the rectangles matches, respond PICK [0]. Do not output anything else.
"""


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._action_config = types.GenerateContentConfig(
            system_instruction=ACTION_SYSTEM_PROMPT,
        )
        self._verify_config = types.GenerateContentConfig(
            system_instruction=VERIFY_SYSTEM_PROMPT,
        )
        self._refine_config = types.GenerateContentConfig(
            system_instruction=REFINE_SYSTEM_PROMPT,
        )
        self._disambig_config = types.GenerateContentConfig(
            system_instruction=DISAMBIG_SYSTEM_PROMPT,
        )
        log.info("Initialized Gemini client with model=%s", model_name)

    def plan_action(
        self,
        step: str,
        screenshot: Image.Image,
        history_summary: str = "",
        previous_failure: str = "",
    ) -> str:
        """Ask the VLM what action to take for the given step.

        `history_summary` is an optional compact text block of recent
        (step, action, verdict) records. `previous_failure` is set when we
        are replanning after a verifier FAIL on this same step — in that
        case the VLM should pick a DIFFERENT action.
        """
        parts: list[str] = []
        if history_summary:
            parts.append(
                "Recent action history (most recent last):\n" + history_summary
            )
        if previous_failure:
            parts.append(
                "Your previous attempt on THIS step failed. Reason:\n"
                f"  {previous_failure}\n"
                "Pick a DIFFERENT action this time."
            )
        parts.append(f"Current step: {step}")
        parts.append("Respond with ONE command only.")
        prompt = "\n\n".join(parts)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[prompt, screenshot],
            config=self._action_config,
        )
        text = (response.text or "").strip()
        log.debug("plan_action response: %r", text)
        return text

    def verify(self, goal: str, screenshot: Image.Image) -> VerificationResult:
        """Ask the VLM to verify that the post-action state matches the goal."""
        prompt = (
            f"Goal of the last action: {goal}\n\n"
            "Did the screen state become consistent with the goal? "
            "Respond with a single VERDICT line."
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[prompt, screenshot],
            config=self._verify_config,
        )
        text = (response.text or "").strip()
        log.debug("verify response: %r", text)
        upper = text.upper()
        if "VERDICT: PASS" in upper:
            return VerificationResult(passed=True, reason=text)
        if "VERDICT: FAIL" in upper:
            return VerificationResult(passed=False, reason=text)
        # Ambiguous — treat as failure so we halt safely.
        return VerificationResult(
            passed=False,
            reason=f"Unparseable verification response: {text!r}",
        )

    def refine_click(
        self,
        step: str,
        crop: Image.Image,
        max_candidates: int = 5,
    ) -> list[tuple[int, int]]:
        """Return candidate click points on the crop's own 0-1000 grid.

        Returns an empty list when the VLM responds `NONE` (nothing in the
        crop matches the step). Clamps the result to at most `max_candidates`.
        """
        prompt = (
            f"Step: {step}\n\n"
            "List plausible click targets on THIS CROP in 0-1000 coords, "
            "one `CLICK [X,Y]` per line, or `NONE`."
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[prompt, crop],
            config=self._refine_config,
        )
        text = (response.text or "").strip()
        log.debug("refine_click response: %r", text)

        if not text or text.upper().startswith("NONE"):
            return []

        candidates: list[tuple[int, int]] = []
        for line in text.splitlines():
            # Primary bracketed form.
            m = re.search(
                r"CLICK\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
                line,
                re.IGNORECASE,
            )
            if not m:
                # Lenient: a bare `[x,y]` or `(x,y)` is also accepted.
                m = re.search(
                    r"[\[(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\])]", line
                )
            if m:
                x = max(0, min(1000, int(m.group(1))))
                y = max(0, min(1000, int(m.group(2))))
                candidates.append((x, y))
            if len(candidates) >= max_candidates:
                break
        return candidates

    def disambiguate_candidates(
        self,
        step: str,
        annotated_screenshot: Image.Image,
        num_candidates: int,
    ) -> int:
        """Return the 1-based index of the best-matching candidate, or 0 if none match."""
        prompt = (
            f"Step: {step}\n\n"
            f"Which of the {num_candidates} numbered red rectangles best "
            "matches the step? Respond with `PICK [N]` only."
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[prompt, annotated_screenshot],
            config=self._disambig_config,
        )
        text = (response.text or "").strip()
        log.debug("disambiguate response: %r", text)

        m = re.search(r"PICK\s*\[\s*(-?\d+)\s*\]", text, re.IGNORECASE)
        if not m:
            # Last-resort: accept any standalone integer.
            m = re.search(r"\b(\d+)\b", text)
        if not m:
            return 0
        pick = int(m.group(1))
        if pick < 0 or pick > num_candidates:
            return 0
        return pick
