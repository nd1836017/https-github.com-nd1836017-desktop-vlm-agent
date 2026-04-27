"""Smart task router — pre-classify each tasks-file step by complexity.

The router runs ONCE per run, after ``load_steps`` expands ``FOR_EACH_ROW``
blocks but before the agent starts executing. It feeds the whole list of
expanded step strings to Gemini in a single call and gets back, per step:

* a ``complexity`` tag (``browser-fast`` / ``browser-vlm`` / ``desktop-vlm``)
* an optional ``suggested_command`` (literal ``BROWSER_GO [...]`` /
  ``BROWSER_CLICK [...]`` / ``BROWSER_FILL [...]`` syntax) when the step
  is a known browser action
* a brief ``reasoning`` string for debug / artifact bundles

The hint is *advisory*. The planner still makes the final call when it
sees the screen. The hint is spliced into the planner's user prompt
(see ``GeminiClient.plan_action``) so the planner knows "the router
classified this as browser-fast and suggests BROWSER_GO" — but if the
screen disagrees (Chrome lost focus, an unexpected popup, etc.) the
planner is free to ignore the hint.

Three modes via ``TASK_ROUTING``:

* ``auto`` (default) — call Gemini once at run start, attach hints
* ``manual`` — skip the Gemini call; only honor inline annotations on
  task lines (``[browser-fast] open youtube`` / ``[vlm] post on instagram``)
* ``off`` — no router at all; behavior identical to pre-router builds

When the router fails (timeout, schema error, exception), we log a
warning and fall through to ``off`` semantics for the rest of the run.
This is critical: a flaky router must NEVER block a run from starting.

Browser fast-path gating: the router is told whether the BROWSER_*
primitives are actually available. When the bridge isn't connected,
the router downgrades any ``browser-fast`` decisions to ``browser-vlm``
and erases ``suggested_command`` so the planner doesn't see a hint
pointing at a primitive it can't use.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from .vlm import GeminiClient

log = logging.getLogger(__name__)


class RoutingMode(str, Enum):
    """User-facing TASK_ROUTING values."""

    AUTO = "auto"
    MANUAL = "manual"
    OFF = "off"


class RoutingComplexity(str, Enum):
    """How a step should be executed.

    ``browser-fast`` is the cheapest (no vision tokens) and only valid
    when Chrome is reachable via CDP. ``browser-vlm`` and ``desktop-vlm``
    both go through the visual planner; they're separated so the planner
    can be told "you're in a browser, prefer CLICK_TEXT" vs "you're on
    the desktop, prefer raw CLICK".
    """

    BROWSER_FAST = "browser-fast"
    BROWSER_VLM = "browser-vlm"
    DESKTOP_VLM = "desktop-vlm"


@dataclass(frozen=True)
class RoutingHint:
    """Per-step routing decision, attached to ``TaskStep.routing_hint``."""

    complexity: RoutingComplexity
    suggested_command: str = ""
    reasoning: str = ""
    # ``source`` is "auto" when the router classified this step via Gemini,
    # "manual" when it came from an inline ``[tag]`` annotation in
    # tasks.txt, "fallback" when the auto router failed and we kept the
    # step unannotated. Useful for artifact bundles + debug logs.
    source: str = "auto"

    def render_for_prompt(self) -> str:
        """Return the human-readable prompt fragment shown to the planner.

        Format is intentionally short — the planner sees it embedded in
        an already-busy ``plan_action`` prompt and we don't want to
        bury the actual step text. Empty strings here are skipped so an
        ``[off]``-mode hint doesn't waste tokens.
        """
        parts = [f"complexity={self.complexity.value}"]
        if self.suggested_command:
            parts.append(f"suggested={self.suggested_command}")
        return ", ".join(parts)


# JSON schema for Gemini structured output. Mirrors the planner's
# ``PlanResponseModel`` style — one Pydantic class per response shape.

class _RouterStepDecision(BaseModel):
    index: int = Field(
        description=(
            "Zero-based index into the input ``steps`` array. Each input "
            "step must produce exactly one output entry with the matching "
            "index."
        )
    )
    complexity: str = Field(
        description=(
            "One of: 'browser-fast', 'browser-vlm', 'desktop-vlm'. Use "
            "'browser-fast' ONLY when a stable selector or URL is known "
            "up-front AND the bridge is enabled; otherwise prefer "
            "'browser-vlm' inside a browser context, or 'desktop-vlm' "
            "on the OS desktop."
        )
    )
    suggested_command: str = Field(
        default="",
        description=(
            "Optional literal command suggestion. Set ONLY for "
            "'browser-fast' steps where you're confident in the selector. "
            "Examples: 'BROWSER_GO [https://youtube.com]', "
            "'BROWSER_FILL [input[name=q], justin bieber]', "
            "'BROWSER_CLICK [button#search-icon-legacy]'. Leave empty "
            "for 'browser-vlm' / 'desktop-vlm'."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief one-sentence explanation. Useful for debugging.",
    )


class _RouterResponseModel(BaseModel):
    steps: list[_RouterStepDecision] = Field(
        description=(
            "One entry per input step, in input order. Length MUST match "
            "the number of input steps."
        )
    )


# System prompt sent on every router call. Long because the router has
# to learn the literal BROWSER_* command syntax; shorter prompts produced
# malformed selectors in early testing (e.g. CSS selectors with smart
# quotes, missing brackets, etc.).
ROUTER_SYSTEM_PROMPT = """You are a task-routing assistant for a desktop automation agent.

Given a numbered list of natural-language step instructions, classify EACH step's complexity AND optionally suggest the literal command syntax the agent should use.

Three complexity tags (pick the FIRST one that applies):

1. browser-fast — simple browser actions where a stable URL or CSS selector is known up-front. The agent can drive Chrome via the DevTools Protocol with zero vision cost. Examples:
   - "open YouTube"               -> complexity=browser-fast, suggested=BROWSER_GO [https://youtube.com]
   - "go to gmail.com"            -> complexity=browser-fast, suggested=BROWSER_GO [https://gmail.com]
   - "search for justin bieber"   -> complexity=browser-fast, suggested=BROWSER_FILL [input[name=search_query], justin bieber]
   - "click the YouTube search button" -> complexity=browser-fast, suggested=BROWSER_CLICK [button#search-icon-legacy]

2. browser-vlm — browser actions that NEED vision (no reliable selector or URL up-front). The planner will use CLICK_TEXT or refined CLICK at run time. Examples:
   - "click the third video result"
   - "click the play button on the cat video"
   - "scroll until you see the Submit button"
   - "click the first thumbnail"

3. desktop-vlm — non-browser actions (native desktop apps, Excel, Photoshop, system dialogs, OS file pickers, Windows Notepad, etc.). Always vision-based. Examples:
   - "open Notepad"
   - "press Win key, type calc, press Enter"
   - "click the File menu in Photoshop"
   - "save the document with Ctrl+S"

Rules:

A. The output ``steps`` array MUST have EXACTLY one entry per input step, in input order, with matching ``index``. No skipping, no merging, no reordering.

B. ``suggested_command`` MUST follow the agent's literal syntax:
   - BROWSER_GO [<url>]              — URL must start with http:// or https://
   - BROWSER_CLICK [<css-selector>]  — selector inside square brackets, no surrounding quotes
   - BROWSER_FILL [<css-selector>, <value>]  — comma-separated, value un-quoted
   Set ``suggested_command`` ONLY for 'browser-fast' steps where you're confident. Leave it empty (``""``) otherwise. A wrong suggestion costs more than no suggestion — when in doubt, leave empty.

C. Compound steps ("type X and press enter") are tagged by the FIRST/most-foundational action; the verifier will replan to subsequent actions. Don't try to encode multiple commands in suggested_command.

D. If a step ALREADY contains literal command syntax (e.g. "BROWSER_GO [https://youtube.com]" or "REMEMBER [order_id]"), respect it: tag according to the command (BROWSER_* -> browser-fast; everything else -> desktop-vlm) and set suggested_command to the literal step.

E. The agent will tell you whether the BROWSER_* primitives are AVAILABLE this run (Chrome must be launched with --remote-debugging-port and reachable). If they're DISABLED, you MUST NOT use 'browser-fast' or BROWSER_* commands. Downgrade to 'browser-vlm' (browser context) or 'desktop-vlm' (otherwise) and leave suggested_command empty.

F. ``reasoning`` is short — one sentence at most. Skip it when the classification is obvious.
"""


# Manual-mode inline annotation regex. Recognises lines like:
#   [browser-fast] open youtube
#   [browser-vlm] click the third video
#   [vlm] post on instagram        (alias for desktop-vlm)
#   [desktop-vlm] open notepad
# Whitespace is tolerant; the tag is case-insensitive. Anything after
# the closing ``]`` is the actual step text. Lines without a tag are
# left unannotated (the router prompts will see them too in auto mode).
_INLINE_ANNOTATION_RE = re.compile(
    r"^\s*\[(?P<tag>[a-zA-Z][a-zA-Z\-_]*)\]\s*(?P<text>.+?)\s*$"
)

# Aliases the user can write in tasks.txt. ``vlm`` is short for
# ``desktop-vlm`` (most common manual override case is "tell the agent
# to use vision here even though it looks browser-y").
_TAG_ALIASES = {
    "browser-fast": RoutingComplexity.BROWSER_FAST,
    "browser_fast": RoutingComplexity.BROWSER_FAST,
    "browser": RoutingComplexity.BROWSER_FAST,
    "fast": RoutingComplexity.BROWSER_FAST,
    "browser-vlm": RoutingComplexity.BROWSER_VLM,
    "browser_vlm": RoutingComplexity.BROWSER_VLM,
    "desktop-vlm": RoutingComplexity.DESKTOP_VLM,
    "desktop_vlm": RoutingComplexity.DESKTOP_VLM,
    "desktop": RoutingComplexity.DESKTOP_VLM,
    "vlm": RoutingComplexity.DESKTOP_VLM,
}


def parse_inline_annotation(line: str) -> tuple[RoutingHint | None, str]:
    """Strip a leading ``[tag]`` annotation from ``line``.

    Returns ``(hint, remaining_text)``. When the line has no recognised
    annotation, ``hint`` is ``None`` and ``remaining_text`` is the
    original line unchanged. When the line has a ``[tag]`` but the tag
    isn't one we recognise, the ``[tag]`` is left in place — we don't
    want to silently strip user-authored bracket syntax that wasn't
    meant as routing (e.g. "Type [admin]" as a literal placeholder).

    The hint's ``source`` is ``"manual"`` so callers can distinguish it
    from auto-router decisions in artifact bundles.
    """
    match = _INLINE_ANNOTATION_RE.match(line)
    if match is None:
        return None, line
    tag = match.group("tag").strip().lower()
    complexity = _TAG_ALIASES.get(tag)
    if complexity is None:
        # Unknown tag — leave the line untouched so we don't eat
        # legitimate non-routing bracket syntax.
        return None, line
    text = match.group("text").strip()
    return (
        RoutingHint(
            complexity=complexity,
            source="manual",
            reasoning="manual inline annotation",
        ),
        text,
    )


class RouterUnavailable(RuntimeError):
    """Raised by the router when Gemini call fails or returns garbage.

    Callers should catch this and fall through to no-routing behavior.
    """


def route_via_gemini(
    client: GeminiClient,
    step_texts: list[str],
    *,
    enable_browser_fast_path: bool,
) -> list[RoutingHint | None]:
    """One Gemini call to classify all steps. Returns hints aligned with input.

    On any error (transport, schema, length mismatch) raises
    ``RouterUnavailable``. The agent's run loop catches that and proceeds
    with no hints rather than crashing the whole run.

    ``enable_browser_fast_path`` MUST reflect the actual bridge state at
    run start. When ``False``, ``browser-fast`` decisions are downgraded
    to ``browser-vlm`` and any ``suggested_command`` is dropped — the
    planner doesn't have BROWSER_* in its system prompt under that
    config and a hint pointing at it would be useless.
    """
    if not step_texts:
        return []

    user_prompt = _build_user_prompt(
        step_texts,
        enable_browser_fast_path=enable_browser_fast_path,
    )
    try:
        decisions = _call_router(client, user_prompt)
    except RouterUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — we want to swallow all
        # Any unforeseen client error (network, auth, library bug) is
        # converted to RouterUnavailable so the caller's fallback path
        # is the same regardless of failure mode.
        raise RouterUnavailable(f"router call failed: {exc}") from exc

    if len(decisions) != len(step_texts):
        raise RouterUnavailable(
            f"router returned {len(decisions)} decisions for "
            f"{len(step_texts)} steps; expected exact match"
        )

    hints: list[RoutingHint | None] = []
    for idx, decision in enumerate(decisions):
        if decision.index != idx:
            # The schema asks for in-order replies; if the model
            # reorders we discard the whole response — preserving
            # alignment is more important than salvaging partial data.
            raise RouterUnavailable(
                f"router decision #{idx} has index={decision.index}; expected {idx}"
            )
        try:
            complexity = RoutingComplexity(decision.complexity.strip().lower())
        except ValueError:
            log.warning(
                "router: unknown complexity %r on step %d; "
                "treating as desktop-vlm",
                decision.complexity,
                idx,
            )
            complexity = RoutingComplexity.DESKTOP_VLM

        suggested = (decision.suggested_command or "").strip()

        # Bridge-disabled downgrade: the planner won't have BROWSER_*
        # in its system prompt, so any browser-fast hint becomes a
        # liability (the planner would see "use BROWSER_GO" and emit
        # something the executor can't run). Downgrade aggressively.
        if not enable_browser_fast_path and complexity == RoutingComplexity.BROWSER_FAST:
            complexity = RoutingComplexity.BROWSER_VLM
            suggested = ""

        # Defensive: if the model put a BROWSER_* command on a non-
        # browser-fast tag, drop it. The planner would either ignore it
        # (lucky) or emit a doomed BROWSER_* call (unlucky).
        if (
            complexity != RoutingComplexity.BROWSER_FAST
            and suggested.upper().startswith(("BROWSER_GO", "BROWSER_CLICK", "BROWSER_FILL"))
        ):
            suggested = ""

        hints.append(
            RoutingHint(
                complexity=complexity,
                suggested_command=suggested,
                reasoning=(decision.reasoning or "").strip(),
                source="auto",
            )
        )
    return hints


def _build_user_prompt(
    step_texts: list[str],
    *,
    enable_browser_fast_path: bool,
) -> str:
    bridge_note = (
        "BROWSER_* primitives ARE available this run (Chrome's CDP is reachable)."
        if enable_browser_fast_path
        else (
            "BROWSER_* primitives are NOT available this run (Chrome is not "
            "launched with --remote-debugging-port, OR the bridge couldn't "
            "connect). You MUST NOT use 'browser-fast' or any BROWSER_* "
            "command. Downgrade to 'browser-vlm' or 'desktop-vlm'."
        )
    )
    numbered = "\n".join(
        f"  {i}. {text}" for i, text in enumerate(step_texts)
    )
    return (
        f"{bridge_note}\n\n"
        f"Steps to classify (index in brackets):\n{numbered}\n\n"
        "Return ONE entry per step in the same order, with matching "
        "``index``. Empty ``suggested_command`` is fine and often correct."
    )


def _call_router(
    client: GeminiClient,
    user_prompt: str,
) -> list[_RouterStepDecision]:
    """Wrap the GeminiClient call so the public surface stays small.

    Lives here (not as a method on GeminiClient) so the router doesn't
    add to the client's already-busy interface and can be unit-tested
    by stubbing the ``call_router_raw`` method.
    """
    raw = client.call_router_raw(
        system_prompt=ROUTER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=_RouterResponseModel,
    )
    if isinstance(raw, _RouterResponseModel):
        return raw.steps
    # call_router_raw promises to return either a parsed model or raise.
    # If we got here with something else, something's wrong — fail loud.
    raise RouterUnavailable(
        f"router got unexpected response type {type(raw).__name__}"
    )


def apply_router(
    step_texts: list[str],
    *,
    mode: RoutingMode,
    client: GeminiClient | None,
    enable_browser_fast_path: bool,
) -> list[RoutingHint | None]:
    """High-level entry point used by ``agent.py:run``.

    Returns one ``RoutingHint | None`` per input step (same length and
    order as input). ``None`` means "no hint" — the planner sees the
    step bare. Callers attach the result to ``TaskStep.routing_hint``.

    * ``mode=off``  -> return ``[None] * len(step_texts)`` (no router call)
    * ``mode=manual`` -> only honor inline ``[tag]`` prefixes; do NOT
      call Gemini. (Inline annotations are stripped from the step text
      EARLIER, in tasks_loader; by the time this function sees the list
      they're already attached. We just no-op here.)
    * ``mode=auto`` -> call Gemini once. Inline annotations on already-
      tagged steps are kept verbatim (manual override beats router).

    On router error in auto mode, log a warning and return ``[None] * N``
    so the run proceeds without hints rather than crashing.
    """
    n = len(step_texts)
    if n == 0 or mode == RoutingMode.OFF:
        return [None] * n
    if mode == RoutingMode.MANUAL:
        # Manual annotations are applied in tasks_loader, so this is
        # purely a no-op. Returning ``None`` here means we don't
        # OVERWRITE an existing manual hint with a fresh router call.
        return [None] * n
    if client is None:
        log.warning(
            "router: mode=auto but no Gemini client provided; "
            "skipping (treating as off)"
        )
        return [None] * n

    try:
        return route_via_gemini(
            client,
            step_texts,
            enable_browser_fast_path=enable_browser_fast_path,
        )
    except RouterUnavailable as exc:
        log.warning(
            "router: auto routing failed (%s); proceeding with no hints",
            exc,
        )
        return [None] * n


def parse_mode(raw: str | None) -> RoutingMode:
    """Parse ``TASK_ROUTING`` env value. Unknown values fall back to AUTO.

    We default to AUTO on unknown input rather than raising because the
    router is opt-out (graceful fallback when it errors anyway), so a
    typo in .env should still leave the agent running.
    """
    if not raw:
        return RoutingMode.AUTO
    try:
        return RoutingMode(raw.strip().lower())
    except ValueError:
        log.warning(
            "TASK_ROUTING=%r not recognised; expected one of "
            "auto/manual/off — defaulting to auto",
            raw,
        )
        return RoutingMode.AUTO


__all__ = [
    "RoutingComplexity",
    "RoutingHint",
    "RoutingMode",
    "RouterUnavailable",
    "ROUTER_SYSTEM_PROMPT",
    "apply_router",
    "parse_inline_annotation",
    "parse_mode",
    "route_via_gemini",
    # Pydantic schemas exported for tests + GeminiClient integration.
    "_RouterStepDecision",
    "_RouterResponseModel",
]


# We intentionally swallow the unused-import below — pydantic ValidationError
# is referenced through the GeminiClient call site only. Keeping it here
# (commented) makes it obvious why the module imports look slightly thin.
_ = ValidationError  # silence unused-import; kept available for future use
