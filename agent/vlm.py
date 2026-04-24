"""Gemini Flash client — action planning + verification calls.

Uses the modern `google-genai` SDK (`pip install google-genai`).
"""
from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

log = logging.getLogger(__name__)


# Status codes that are considered transient: the model is up but busy or
# rate-limited. We back off and retry these rather than crashing — the user
# explicitly said: "do not fall back to another model, wait then try again".
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

T = TypeVar("T")


def _call_with_retry(
    fn: Callable[[], T],
    *,
    label: str,
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> T:
    """Invoke ``fn`` with exponential backoff on transient Gemini failures.

    Retries on 429 / 5xx only. Non-retryable errors (bad API key, malformed
    request) are re-raised immediately. Exponential backoff with full jitter:
    sleep ~= random(0, min(max_delay, base * 2**attempt)).
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except (genai_errors.ServerError, genai_errors.ClientError) as exc:
            status = getattr(exc, "status_code", getattr(exc, "code", None))
            if status not in _RETRYABLE_STATUS:
                raise
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            # Exponential backoff with full jitter.
            cap = min(max_delay_seconds, base_delay_seconds * (2 ** attempt))
            delay = random.uniform(0.0, cap)
            log.warning(
                "%s: %s %s — retry %d/%d after %.1fs",
                label,
                status,
                type(exc).__name__,
                attempt + 1,
                max_attempts,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


ACTION_SYSTEM_PROMPT = """You are a desktop automation agent. You see a screenshot of the user's screen and a single natural-language step to perform.

RESPOND WITH EXACTLY ONE COMMAND on its own line, chosen from:

    CLICK [X,Y]              — left-click at normalized coordinates. X and Y are integers in [0, 1000] where (0,0) is the top-left of the screen and (1000,1000) is the bottom-right. Do NOT use pixel coordinates.
    DOUBLE_CLICK [X,Y]       — double-click at normalized coordinates. Use for opening files/folders.
    RIGHT_CLICK [X,Y]        — right-click at normalized coordinates. Use to open context menus.
    MOVE_TO [X,Y]            — move the mouse WITHOUT clicking. Use to trigger hover tooltips or open hover menus.
    PRESS [KEY]              — press a single key or hotkey. Examples: PRESS [win], PRESS [enter], PRESS [esc], PRESS [ctrl+c], PRESS [alt+tab].
    TYPE [TEXT]              — type literal text. Example: TYPE [hello world].
    SCROLL [DIR, AMOUNT]     — scroll the window. DIR is up or down; AMOUNT is a positive integer number of wheel clicks. Example: SCROLL [down, 5].
    DRAG [X1,Y1,X2,Y2]       — press at (X1,Y1), drag to (X2,Y2), release. Use for sliders, drag-to-select, drag-and-drop.
    WAIT [SECONDS]           — sleep SECONDS (float, capped at 60) before the next step. Use when waiting for a page to load or an animation to finish.

Rules:
- Output ONLY the command, wrapped in square brackets. No prose, no markdown, no explanation.
- Use the 0-1000 normalized grid for coordinates — never pixels.
- Prefer CLICK for ordinary buttons/links. Use DOUBLE_CLICK only when the UI requires it (desktop icons, file-manager rows).
- If the step has already been completed, still emit a single no-op friendly command (e.g. WAIT [0.5]) — never output nothing.
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
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.1-flash-lite-preview",
        *,
        retry_max_attempts: int = 6,
        retry_base_delay_seconds: float = 5.0,
        retry_max_delay_seconds: float = 300.0,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._retry_max_attempts = retry_max_attempts
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
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

    def _generate(
        self,
        label: str,
        contents: list,
        config: types.GenerateContentConfig,
    ):
        """Call Gemini with retry-on-503/429 and exponential backoff."""
        return _call_with_retry(
            lambda: self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            ),
            label=label,
            max_attempts=self._retry_max_attempts,
            base_delay_seconds=self._retry_base_delay_seconds,
            max_delay_seconds=self._retry_max_delay_seconds,
        )

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

        response = self._generate("plan_action", [prompt, screenshot], self._action_config)
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
        response = self._generate("verify", [prompt, screenshot], self._verify_config)
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
        response = self._generate("refine_click", [prompt, crop], self._refine_config)
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
        response = self._generate("disambiguate", [prompt, annotated_screenshot], self._disambig_config)
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
