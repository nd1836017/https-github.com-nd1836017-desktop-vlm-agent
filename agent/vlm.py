"""Gemini Flash client — action planning + verification calls.

Uses the modern `google-genai` SDK (`pip install google-genai`).
"""
from __future__ import annotations

import logging
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


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._action_config = types.GenerateContentConfig(
            system_instruction=ACTION_SYSTEM_PROMPT,
        )
        self._verify_config = types.GenerateContentConfig(
            system_instruction=VERIFY_SYSTEM_PROMPT,
        )
        log.info("Initialized Gemini client with model=%s", model_name)

    def plan_action(self, step: str, screenshot: Image.Image) -> str:
        """Ask the VLM what action to take for the given step."""
        prompt = f"Current step: {step}\n\nRespond with ONE command only."
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
