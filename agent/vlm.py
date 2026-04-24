"""Gemini Flash client — action planning + verification calls.

Uses the modern `google-genai` SDK (`pip install google-genai`).
"""
from __future__ import annotations

import json
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
from pydantic import BaseModel, Field, ValidationError

from .cost import RpdGuard
from .parser import (
    ClickCommand,
    ClickTextCommand,
    Command,
    DoubleClickCommand,
    DragCommand,
    MoveToCommand,
    PauseCommand,
    PressCommand,
    RightClickCommand,
    ScrollCommand,
    TypeCommand,
    WaitCommand,
)

log = logging.getLogger(__name__)


class PlanResponseModel(BaseModel):
    """Structured schema for the planner's JSON output.

    One of the sub-command fields is populated for each response. Fields not
    relevant to the chosen command may be omitted or left null; the converter
    below ignores them.
    """

    command: str = Field(
        description=(
            "The action kind. Must be one of: CLICK, DOUBLE_CLICK, "
            "RIGHT_CLICK, MOVE_TO, PRESS, TYPE, SCROLL, DRAG, WAIT, "
            "CLICK_TEXT, PAUSE."
        )
    )
    x: int | None = Field(default=None, description="Normalized X in [0,1000].")
    y: int | None = Field(default=None, description="Normalized Y in [0,1000].")
    key: str | None = Field(default=None, description="Key name for PRESS.")
    text: str | None = Field(default=None, description="Text to TYPE.")
    direction: str | None = Field(default=None, description="up or down, for SCROLL.")
    amount: int | None = Field(default=None, description="Scroll distance (positive int).")
    x1: int | None = Field(default=None, description="DRAG start X.")
    y1: int | None = Field(default=None, description="DRAG start Y.")
    x2: int | None = Field(default=None, description="DRAG end X.")
    y2: int | None = Field(default=None, description="DRAG end Y.")
    seconds: float | None = Field(default=None, description="Seconds for WAIT.")
    label: str | None = Field(default=None, description="Text label for CLICK_TEXT.")
    reason: str | None = Field(
        default=None,
        description="Human-readable reason for PAUSE (e.g. 'Verify it's you prompt').",
    )


def _parse_plan_response_json(text: str) -> PlanResponseModel | None:
    """Best-effort JSON -> PlanResponseModel when the SDK didn't populate `.parsed`.

    This is the safety net: when `ENABLE_JSON_OUTPUT=true` the model is
    configured with `response_mime_type="application/json"`, so the raw
    text should be valid JSON even if `response.parsed` is None (old SDK
    versions, schema-validation quirks, etc.). Without this fallback, the
    caller falls through to `parser.parse_command()` which is a regex
    designed for free-form text and won't extract anything useful from
    a JSON blob.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    # Strip optional markdown fences the model sometimes adds despite the
    # mime_type hint.
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return PlanResponseModel.model_validate(payload)
    except ValidationError:
        return None


def _parse_verify_response_json(text: str) -> VerifyResponseModel | None:
    """Best-effort JSON -> VerifyResponseModel when the SDK didn't populate `.parsed`.

    Mirrors ``_parse_plan_response_json`` for the verify call. Without
    this, a response like ``{"verdict": "PASS", "reason": "..."}`` would
    slip past the isinstance(parsed, VerifyResponseModel) gate and be
    regex-searched for ``VERDICT: PASS`` (which it never contains),
    producing a bogus "Unparseable" FAIL.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return VerifyResponseModel.model_validate(payload)
    except ValidationError:
        return None


def plan_response_to_command(resp: PlanResponseModel) -> Command | None:
    """Convert a parsed JSON schema response into a Command instance."""
    kind = (resp.command or "").upper().strip()
    try:
        if kind == "CLICK" and resp.x is not None and resp.y is not None:
            return ClickCommand(x=int(resp.x), y=int(resp.y))
        if kind == "DOUBLE_CLICK" and resp.x is not None and resp.y is not None:
            return DoubleClickCommand(x=int(resp.x), y=int(resp.y))
        if kind == "RIGHT_CLICK" and resp.x is not None and resp.y is not None:
            return RightClickCommand(x=int(resp.x), y=int(resp.y))
        if kind == "MOVE_TO" and resp.x is not None and resp.y is not None:
            return MoveToCommand(x=int(resp.x), y=int(resp.y))
        if kind == "PRESS" and resp.key:
            return PressCommand(key=resp.key.strip())
        if kind == "TYPE" and resp.text is not None:
            return TypeCommand(text=resp.text)
        if (
            kind == "SCROLL"
            and resp.direction
            and resp.amount is not None
        ):
            direction = resp.direction.lower().strip()
            if direction not in {"up", "down"}:
                return None
            return ScrollCommand(direction=direction, amount=abs(int(resp.amount)))
        if (
            kind == "DRAG"
            and resp.x1 is not None
            and resp.y1 is not None
            and resp.x2 is not None
            and resp.y2 is not None
        ):
            return DragCommand(
                x1=int(resp.x1), y1=int(resp.y1), x2=int(resp.x2), y2=int(resp.y2)
            )
        if kind == "WAIT" and resp.seconds is not None:
            return WaitCommand(seconds=float(resp.seconds))
        if kind == "CLICK_TEXT" and resp.label:
            return ClickTextCommand(label=resp.label.strip())
        if kind == "PAUSE" and resp.reason:
            return PauseCommand(reason=resp.reason.strip())
    except (TypeError, ValueError) as exc:
        log.warning("plan_response_to_command: bad payload: %s", exc)
        return None
    return None


class VerifyResponseModel(BaseModel):
    verdict: str = Field(description="PASS or FAIL.")
    reason: str = Field(default="", description="Short human-readable justification.")


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
    rpd_guard: RpdGuard | None = None,
) -> T:
    """Invoke ``fn`` with exponential backoff on transient Gemini failures.

    Retries on 429 / 5xx only. Non-retryable errors (bad API key, malformed
    request) are re-raised immediately. Exponential backoff with full jitter:
    sleep ~= random(0, min(max_delay, base * 2**attempt)).

    When ``rpd_guard`` is provided, ``record()`` is called once before every
    attempt (including retries) so the RPD counter matches the real number
    of outbound HTTP requests, not just the number of logical calls.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        if rpd_guard is not None:
            rpd_guard.record()
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
    CLICK_TEXT [LABEL]       — click the on-screen text whose label best matches LABEL. Use ONLY for text-labeled UI (buttons, links) where you can read the exact label. Do NOT use for icon-only targets.
    PAUSE [REASON]           — halt and wait for a human to resolve REASON. EMIT THIS WHENEVER you see a screen that requires manual user action that the agent cannot perform: 2FA / device-approval prompts ("Verify it's you", "Check your phone", "Enter the code we sent"), CAPTCHA / "I'm not a robot" challenges, security questions, or any "browser may not be secure" warning. REASON must be a short human-readable string explaining what the user needs to do. Example: PAUSE [Approve the sign-in on your phone, then resume].

Rules:
- Output ONLY the command, wrapped in square brackets. No prose, no markdown, no explanation.
- Use the 0-1000 normalized grid for coordinates — never pixels.
- Prefer CLICK for ordinary buttons/links. Use DOUBLE_CLICK only when the UI requires it (desktop icons, file-manager rows).
- If the step has already been completed, still emit a single no-op friendly command (e.g. WAIT [0.5]) — never output nothing.
- You may be shown a summary of previous steps and a previous-attempt failure reason. Use them to avoid repeating mistakes and to pick a DIFFERENT action when replanning after a failure.
- NEVER attempt to bypass a 2FA or CAPTCHA challenge yourself. ALWAYS emit PAUSE [reason] in that situation — a human will resolve it and resume the agent.
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
        enable_json_output: bool = True,
        rpd_guard: RpdGuard | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._retry_max_attempts = retry_max_attempts
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        self._enable_json_output = enable_json_output
        self._rpd_guard = rpd_guard or RpdGuard()
        self._action_config = types.GenerateContentConfig(
            system_instruction=ACTION_SYSTEM_PROMPT,
        )
        self._action_config_json = types.GenerateContentConfig(
            system_instruction=ACTION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=PlanResponseModel,
        )
        self._verify_config = types.GenerateContentConfig(
            system_instruction=VERIFY_SYSTEM_PROMPT,
        )
        self._verify_config_json = types.GenerateContentConfig(
            system_instruction=VERIFY_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=VerifyResponseModel,
        )
        self._refine_config = types.GenerateContentConfig(
            system_instruction=REFINE_SYSTEM_PROMPT,
        )
        self._disambig_config = types.GenerateContentConfig(
            system_instruction=DISAMBIG_SYSTEM_PROMPT,
        )
        log.info(
            "Initialized Gemini client with model=%s (json_output=%s)",
            model_name,
            enable_json_output,
        )

    @property
    def rpd_guard(self) -> RpdGuard:
        return self._rpd_guard

    def _generate(
        self,
        label: str,
        contents: list,
        config: types.GenerateContentConfig,
    ):
        """Call Gemini with retry-on-503/429 and exponential backoff.

        The RPD guard is recorded inside ``_call_with_retry`` so every real
        HTTP attempt is counted — a single logical request that takes five
        retries to succeed consumes five slots of daily quota.
        """
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
            rpd_guard=self._rpd_guard,
        )

    def plan_action(
        self,
        step: str,
        screenshot: Image.Image,
        history_summary: str = "",
        previous_failure: str = "",
    ) -> tuple[str, Command | None]:
        """Ask the VLM what action to take for the given step.

        Returns `(raw_text, command_or_none)`. When JSON-output mode is
        enabled, `command_or_none` is set to the deterministically-parsed
        Command and the caller can skip regex parsing. On JSON parse failure
        or when JSON is disabled, it is None and the caller should fall back
        to `parse_command(raw_text)`.

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

        if self._enable_json_output:
            parts.append(
                "Respond with a single JSON object following the schema."
            )
            prompt = "\n\n".join(parts)
            response = self._generate(
                "plan_action_json",
                [prompt, screenshot],
                self._action_config_json,
            )
            text = (response.text or "").strip()
            log.debug("plan_action response (json): %r", text)

            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, PlanResponseModel):
                cmd = plan_response_to_command(parsed)
                if cmd is not None:
                    return text, cmd
                log.warning(
                    "plan_action: JSON schema returned but could not be mapped "
                    "to a Command; falling back to manual JSON parse: %r",
                    parsed,
                )
            else:
                log.warning(
                    "plan_action: no structured parse available; "
                    "trying manual JSON parse of raw text."
                )

            # Safety net: the SDK didn't populate `response.parsed`, but the
            # raw text should still be JSON (mime_type was set). Decode it
            # ourselves before handing back to the regex parser — the regex
            # is designed for free-form "CLICK [x,y]" text and won't extract
            # anything from a JSON blob.
            manual = _parse_plan_response_json(text)
            if manual is not None:
                cmd = plan_response_to_command(manual)
                if cmd is not None:
                    return text, cmd
            return text, None

        parts.append("Respond with ONE command only.")
        prompt = "\n\n".join(parts)
        response = self._generate("plan_action", [prompt, screenshot], self._action_config)
        text = (response.text or "").strip()
        log.debug("plan_action response: %r", text)
        return text, None

    def verify(self, goal: str, screenshot: Image.Image) -> VerificationResult:
        """Ask the VLM to verify that the post-action state matches the goal."""
        prompt = (
            f"Goal of the last action: {goal}\n\n"
            "Did the screen state become consistent with the goal? "
            "Respond with a PASS or FAIL verdict."
        )

        if self._enable_json_output:
            response = self._generate(
                "verify_json", [prompt, screenshot], self._verify_config_json
            )
            text = (response.text or "").strip()
            log.debug("verify response (json): %r", text)
            parsed = getattr(response, "parsed", None)
            if not isinstance(parsed, VerifyResponseModel):
                # SDK didn't populate `.parsed` — try to decode the raw JSON
                # ourselves before handing off to the free-form text parser.
                # The legacy `_parse_verify_text` looks for "VERDICT: PASS"
                # literal strings which do NOT appear in a JSON blob
                # containing `"verdict": "PASS"`, so without this step we'd
                # always mislabel valid responses as "Unparseable".
                parsed = _parse_verify_response_json(text)

            if isinstance(parsed, VerifyResponseModel):
                verdict = (parsed.verdict or "").upper().strip()
                reason = (parsed.reason or "").strip() or verdict
                if verdict == "PASS":
                    return VerificationResult(passed=True, reason=reason)
                if verdict == "FAIL":
                    return VerificationResult(passed=False, reason=reason)
                log.warning(
                    "verify: JSON parsed but verdict was %r; treating as FAIL.",
                    parsed.verdict,
                )
                return VerificationResult(
                    passed=False,
                    reason=f"Unparseable verdict: {parsed.verdict!r}",
                )
            # Last-resort fall through — this path now handles only truly
            # non-JSON responses (e.g. the model emitted raw prose).
            return self._parse_verify_text(text)

        response = self._generate("verify", [prompt, screenshot], self._verify_config)
        text = (response.text or "").strip()
        log.debug("verify response: %r", text)
        return self._parse_verify_text(text)

    @staticmethod
    def _parse_verify_text(text: str) -> VerificationResult:
        upper = text.upper()
        if "VERDICT: PASS" in upper or upper.strip().startswith("PASS"):
            return VerificationResult(passed=True, reason=text)
        if "VERDICT: FAIL" in upper or upper.strip().startswith("FAIL"):
            return VerificationResult(passed=False, reason=text)
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
