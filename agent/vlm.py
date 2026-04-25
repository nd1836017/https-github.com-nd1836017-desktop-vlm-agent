"""Gemini Flash client — action planning + verification calls.

Uses the modern `google-genai` SDK (`pip install google-genai`).
"""
from __future__ import annotations

import io
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
    AttachFileCommand,
    CaptureForAiCommand,
    ClickCommand,
    ClickTextCommand,
    Command,
    DoubleClickCommand,
    DownloadCommand,
    DragCommand,
    MoveToCommand,
    PauseCommand,
    PressCommand,
    RecallCommand,
    RememberCommand,
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
            "CLICK_TEXT, PAUSE, DOWNLOAD, ATTACH_FILE, CAPTURE_FOR_AI, "
            "REMEMBER, RECALL."
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
    url: str | None = Field(
        default=None,
        description="URL to fetch for DOWNLOAD. Must be http(s).",
    )
    filename: str | None = Field(
        default=None,
        description=(
            "Filename for DOWNLOAD / ATTACH_FILE / CAPTURE_FOR_AI. For "
            "DOWNLOAD this is optional (derived from the URL when missing). "
            "For ATTACH_FILE it is required. For CAPTURE_FOR_AI it is "
            "optional (the current screen is captured when missing)."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Variable name for REMEMBER / RECALL. Must match "
            "[A-Za-z_][A-Za-z0-9_]*."
        ),
    )
    literal_value: str | None = Field(
        default=None,
        description=(
            "For REMEMBER, an explicit literal value to store. When omitted "
            "and `from_screen=true`, the agent asks the VLM to extract the "
            "value from the current screen."
        ),
    )
    from_screen: bool | None = Field(
        default=None,
        description=(
            "For REMEMBER: true means extract the value from the current "
            "screenshot via the VLM. False/null means use `literal_value`."
        ),
    )


# Fences look like ```json\n...\n``` or ```YAML\n...\n``` or just ```\n...\n```.
# We strip any optional opening ```<language-tag>\n and trailing ``` in one shot.
# Handles any language tag (json, JSON, yaml, ...) or no tag at all. The old
# implementation did `stripped.strip("`")` which removed *all* backticks
# (including ones inside the JSON!) and only stripped a lowercase "json"
# prefix, so `\`\`\`JSON\n...\n\`\`\`` left a bogus "JSON" at the start of the payload.
_MARKDOWN_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z]*\n?")
_MARKDOWN_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")


def _strip_markdown_fences(text: str) -> str:
    r"""Strip an optional ```<lang>\n ... \n``` fence from ``text``.

    Handles any language tag (case-insensitive) or none. Leaves text
    without a fence unchanged.
    """
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    without_open = _MARKDOWN_FENCE_OPEN_RE.sub("", stripped, count=1)
    without_close = _MARKDOWN_FENCE_CLOSE_RE.sub("", without_open, count=1)
    return without_close.strip()


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
    stripped = _strip_markdown_fences(text)
    if not stripped:
        return None
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
    stripped = _strip_markdown_fences(text)
    if not stripped:
        return None
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


def _parse_extract_response_json(text: str):
    """Best-effort JSON -> ExtractResponseModel for extract_value's safety net.

    Returns ``None`` if the text isn't valid JSON in the expected shape.
    Defined here as a module function (mirroring ``_parse_verify_response_json``)
    so it can be unit-tested independently of the live VLM client.
    """
    stripped = _strip_markdown_fences(text)
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return ExtractResponseModel.model_validate(payload)
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
                # Surface the bad VLM response loudly. Returning None
                # silently dropped malformed scrolls and made debugging
                # impossible — the planner just looked like it sent
                # nothing. The text-mode parser already constrains
                # direction via _SCROLL_RE so this path is JSON-only.
                log.warning(
                    "SCROLL: invalid direction %r (expected 'up' or 'down') "
                    "— dropping plan",
                    resp.direction,
                )
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
        if kind == "DOWNLOAD" and resp.url:
            return DownloadCommand(
                url=resp.url.strip(),
                filename=(resp.filename or "").strip(),
            )
        if kind == "ATTACH_FILE" and resp.filename:
            return AttachFileCommand(filename=resp.filename.strip())
        if kind == "CAPTURE_FOR_AI":
            # filename optional — empty means "grab the current screen".
            return CaptureForAiCommand(filename=(resp.filename or "").strip())
        if kind == "REMEMBER" and resp.name:
            literal = resp.literal_value
            from_screen = (
                bool(resp.from_screen) if resp.from_screen is not None
                else (literal is None or literal == "")
            )
            return RememberCommand(
                name=resp.name.strip(),
                literal_value=(literal or "").strip(),
                from_screen=from_screen,
            )
        if kind == "RECALL" and resp.name:
            return RecallCommand(name=resp.name.strip())
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

# Pull a leading "503 ..." style HTTP status code out of an exception's
# message string. Used as the last-ditch fallback when neither ``code``
# nor ``status_code`` is set as an attribute (defensive against future
# google-genai SDK shapes — APIError today exposes ``code``).
_LEADING_HTTP_STATUS_RE = re.compile(r"^\s*(\d{3})\b")


def _status_code_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction from a google-genai APIError.

    The current ``google-genai`` SDK (``APIError`` subclasses) exposes the
    HTTP status as the ``code`` attribute. Older / forked versions, and a
    handful of internal wrapper exceptions, have used ``status_code`` or
    ``http_status`` instead. We try each in order, then fall back to
    parsing the leading status code out of ``str(exc)`` (the SDK formats
    its message as ``"{code} {status}. {details}"``). If nothing extractable
    is found we return ``None`` and the caller re-raises immediately —
    which is the safe default: we'd rather surface an unfamiliar error
    quickly than retry it ``retry_max_attempts`` times for nothing.
    """
    for attr in ("code", "status_code", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    msg = str(exc) if exc is not None else ""
    m = _LEADING_HTTP_STATUS_RE.match(msg)
    if m:
        return int(m.group(1))
    return None


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
            status = _status_code_of(exc)
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
    DOWNLOAD [URL, FILENAME] — fetch URL via HTTPS into the run's file workspace. URL must start with http:// or https://. FILENAME is optional (derived from the URL path when missing). The file is persisted according to the run's file mode (temp / save / feed). Example: DOWNLOAD [https://example.com/inv.pdf, invoice.pdf].
    ATTACH_FILE [FILENAME]   — type a path into the focused OS file-picker dialog (Ctrl+L, type path, press Enter). The dialog must already be open (you should have CLICKed the Browse button on the previous step). FILENAME is resolved against the workspace first, then disk. Example: ATTACH_FILE [invoice.pdf].
    CAPTURE_FOR_AI [FILENAME] — buffer an image to feed into the NEXT plan_action call as additional context. Useful when an answer requires reading text from a PDF / receipt / image you've just opened. With no FILENAME, the current screen is captured. With a FILENAME, that file is read from the workspace or disk. Example: CAPTURE_FOR_AI [] or CAPTURE_FOR_AI [invoice.pdf].
    REMEMBER [NAME]          — extract the value labeled NAME off the current screen and store it as a run variable. NAME is a plain identifier (alphanumeric and underscores). The agent will read the screenshot and decide what value belongs to NAME — pick this command when the next steps need a value (order id, confirmation number, account name, etc.) that is currently visible on screen. Example: REMEMBER [order_id].
    REMEMBER [NAME = LITERAL] — store LITERAL as the value of NAME with no extraction. Useful when you already know the value (e.g. derived from a previous CAPTURE_FOR_AI). Example: REMEMBER [order_id = ND12345].
    RECALL [NAME]            — type the stored value of variable NAME into the focused field. Equivalent to TYPE [{{var.NAME}}]. Use after a REMEMBER. Example: RECALL [order_id].

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


EXTRACT_SYSTEM_PROMPT = """You are a value-extraction assistant for a desktop automation agent.

You are given:
  1. A screenshot of the user's screen.
  2. A NAME (a plain identifier like "order_id" or "confirmation_number" or "total_amount") describing the value the agent wants to read off the screen.
  3. Optionally a HINT — a natural-language sentence elaborating on what to look for.

Find the value of NAME on the screen and respond with EXACTLY ONE line in this format:

    VALUE: <the extracted value>

Rules:
- Output ONLY the literal value (no labels, no surrounding quotes, no prose).
- If the value is clearly NOT visible on screen, respond with the single line: NONE
- Strip leading/trailing whitespace from the value.
- For numeric values, include relevant decimal points / currency symbols / dashes exactly as shown.
- Do NOT include the prefix word (e.g. "Order ID:") in the value — only the value itself.
"""


class ExtractResponseModel(BaseModel):
    """Structured schema for ``extract_value`` JSON output."""

    found: bool = Field(description="Whether the value was found on the screen.")
    value: str = Field(default="", description="The extracted value, stripped.")


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of ``GeminiClient.extract_value``.

    ``found`` is True when the VLM located a value, in which case ``value``
    is the extracted string (already stripped). When ``found`` is False the
    caller should treat the REMEMBER step as a FAIL so the replan loop can
    try again from a different vantage point.
    """

    found: bool
    value: str
    raw: str = ""


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
        self._extract_config = types.GenerateContentConfig(
            system_instruction=EXTRACT_SYSTEM_PROMPT,
        )
        self._extract_config_json = types.GenerateContentConfig(
            system_instruction=EXTRACT_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ExtractResponseModel,
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
        extra_images: list[bytes] | None = None,
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

        `extra_images` are bytes of additional images (PNG/JPEG) to feed
        the planner alongside the current screenshot. CAPTURE_FOR_AI and
        FEED-mode DOWNLOAD push bytes into the workspace's feed buffer;
        the caller drains that buffer and passes it here so the planner
        can see those images on its NEXT call.
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
        if extra_images:
            parts.append(
                f"You also have {len(extra_images)} attached image(s) provided "
                "by the previous step (CAPTURE_FOR_AI / FEED). Use them as "
                "additional context."
            )
        parts.append(f"Current step: {step}")

        # Decode any feed-buffer bytes into PIL Images alongside the live
        # screenshot. Bad bytes are skipped with a warning so a single
        # corrupt capture doesn't poison the whole call.
        attached_images: list[Image.Image] = []
        for blob in extra_images or []:
            try:
                attached_images.append(Image.open(io.BytesIO(blob)))
            except (OSError, ValueError) as exc:
                log.warning(
                    "plan_action: dropping unreadable extra image (%d bytes): %s",
                    len(blob),
                    exc,
                )

        if self._enable_json_output:
            parts.append(
                "Respond with a single JSON object following the schema."
            )
            prompt = "\n\n".join(parts)
            response = self._generate(
                "plan_action_json",
                [prompt, screenshot, *attached_images],
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
        response = self._generate(
            "plan_action",
            [prompt, screenshot, *attached_images],
            self._action_config,
        )
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

    def extract_value(
        self,
        name: str,
        screenshot: Image.Image,
        hint: str = "",
    ) -> ExtractionResult:
        """Ask the VLM to read the value of ``name`` off the screen.

        Used by the REMEMBER primitive when no literal is given. ``name`` is a
        plain identifier like ``order_id``; ``hint`` is an optional natural
        language sentence elaborating on what to look for (e.g. "the
        confirmation number, formatted as 12 digits with dashes").

        Returns an ExtractionResult; on failure the caller treats the step as
        FAIL and the replan loop kicks in.
        """
        parts = [f"NAME: {name}"]
        if hint:
            parts.append(f"HINT: {hint}")
        parts.append(
            "Find the value of NAME on the screen. "
            "Respond with the literal value or NONE."
        )
        prompt = "\n\n".join(parts)

        if self._enable_json_output:
            response = self._generate(
                "extract_value_json",
                [prompt, screenshot],
                self._extract_config_json,
            )
            text = (response.text or "").strip()
            log.debug("extract_value response (json): %r", text)

            parsed = getattr(response, "parsed", None)
            if not isinstance(parsed, ExtractResponseModel):
                # Manual JSON decode as a safety net.
                parsed = _parse_extract_response_json(text)

            if isinstance(parsed, ExtractResponseModel):
                if parsed.found and parsed.value.strip():
                    return ExtractionResult(
                        found=True,
                        value=parsed.value.strip(),
                        raw=text,
                    )
                return ExtractionResult(found=False, value="", raw=text)
            # Last-ditch text parse.
            return self._parse_extract_text(text)

        response = self._generate(
            "extract_value", [prompt, screenshot], self._extract_config
        )
        text = (response.text or "").strip()
        log.debug("extract_value response: %r", text)
        return self._parse_extract_text(text)

    @staticmethod
    def _parse_extract_text(text: str) -> ExtractionResult:
        """Parse a free-form extract response into an ExtractionResult."""
        if not text:
            return ExtractionResult(found=False, value="", raw="")
        upper = text.strip().upper()
        if upper == "NONE" or upper.startswith("NONE\n"):
            return ExtractionResult(found=False, value="", raw=text)
        # Strip the optional ``VALUE: `` prefix the system prompt asks for.
        m = re.match(r"\s*VALUE\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if m:
            value = m.group(1).strip()
        else:
            # Free-form: take the first non-empty line as the value.
            value = next(
                (line.strip() for line in text.splitlines() if line.strip()),
                "",
            )
        if not value or value.upper() == "NONE":
            return ExtractionResult(found=False, value="", raw=text)
        return ExtractionResult(found=True, value=value, raw=text)
