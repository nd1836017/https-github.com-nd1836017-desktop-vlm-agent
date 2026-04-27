"""Task decomposer — split compound natural-language steps into atomic substeps.

Runs ONCE at run start, BEFORE the smart task router. Takes the loaded
``TaskStep`` list and asks Gemini to expand any compound instruction
("play the 2nd video on youtube", "log into Gmail and check my unread
mail") into atomic substeps the agent can execute one at a time.

Single-action lines pass through unchanged. Compound lines emit 2+
substeps, each carrying the original step's row metadata so DOWNLOAD
filenames and FOR_EACH_ROW row indexing keep working.

Three modes via ``TASK_DECOMPOSITION``:

* ``auto`` (default) — call Gemini once at run start, expand any
  compound steps, then continue. On any error the run proceeds with
  the original step list (graceful fallback).
* ``off`` — no decomposition; behavior identical to pre-decomposer
  builds.

Why a SEPARATE call from the router (rather than smashing both into
one Gemini call):

* The router and decomposer have different jobs. Decomposition cares
  about *what to do*; routing cares about *how cheap each thing is*.
  Cleaner abstraction = fewer regressions when one is updated.
* A user can disable decomposition without disabling routing, and
  vice versa. Useful when one of them misbehaves on a particular
  workflow.
* The two prompts can evolve independently.

The ~1 extra Gemini call at run start is cheap relative to the
per-step cost over the whole run.

Resume note: the decomposer is non-deterministic in principle (LLM).
If a user halts mid-run and resumes, the new run's expansion may not
match the old expansion's step count. ``run()`` validates
``existing_state.total_steps == len(steps)``; on mismatch, the resume
silently restarts from step 0. Documented in the PR description.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .tasks_loader import TaskStep

if TYPE_CHECKING:
    from .vlm import GeminiClient

log = logging.getLogger(__name__)


class DecompositionMode(str, Enum):
    """User-facing TASK_DECOMPOSITION values."""

    AUTO = "auto"
    OFF = "off"


# Pydantic schemas for Gemini structured output.

class _DecomposedStep(BaseModel):
    original_index: int = Field(
        description=(
            "Zero-based index into the input ``steps`` array. Each input "
            "step contributes 1 or more output entries with this same "
            "index — atomic input steps just emit one entry pointing at "
            "themselves; compound input steps emit 2 or more atomic "
            "entries all pointing at the same input index."
        )
    )
    text: str = Field(
        description=(
            "The atomic step text — exactly what the planner will see. "
            "ONE verifiable action per entry. Use plain natural language; "
            "do NOT emit literal command syntax (no CLICK [x,y], no "
            "PRESS [enter], etc.). The planner will translate to commands."
        )
    )


class _DecomposerResponseModel(BaseModel):
    steps: list[_DecomposedStep] = Field(
        description=(
            "Flat list of atomic steps. Each input step contributes >=1 "
            "entries, in order. Concatenating ``steps`` in order yields "
            "the full expanded plan."
        )
    )


# System prompt sent on every decomposer call. The "ONE verb per line"
# rule mirrors the rule we documented for human-authored tasks.txt files
# in the PR #20 README + planner prompt; the LLM has shown it learns
# that rhythm well when given concrete worked examples.
DECOMPOSER_SYSTEM_PROMPT = """You are a task-decomposition assistant for a desktop automation agent.

Given a numbered list of natural-language step instructions, split each step into the SMALLEST set of ATOMIC actions the agent can verify one at a time.

Atomic = ONE verifiable action per substep. The agent's verifier checks each substep against the screen after one command runs, so "open YouTube and play a video" must become at least two substeps because opening YouTube and playing a video are two distinct screen states.

Rules

A. Output array MUST be FLAT. Every output entry has an ``original_index`` pointing back at the input step it came from. Atomic input steps emit exactly one output entry (text == input). Compound input steps emit 2 or more entries all sharing the same ``original_index``, in execution order.

B. Each output ``text`` is ONE atomic, plain-English instruction. Examples:
   - "open Chrome"
   - "go to youtube.com"
   - "click the YouTube search bar at the top"
   - "type 'justin bieber' into the search box"
   - "press Enter to submit the search"
   - "click the second video result"

C. Do NOT emit literal command syntax. The planner translates to commands. WRONG: "BROWSER_GO [https://youtube.com]". RIGHT: "go to youtube.com".

D. ATOMIC means: focus → input → confirm are SEPARATE. "type X and press enter" -> two substeps. "click then type" -> two substeps. Never combine.

E. PRESERVE the original line when it's already atomic. Single-action inputs ("press Enter", "open Notepad", "click the Submit button") should emit one substep with the same text. Don't paraphrase or "improve" already-atomic steps.

F. Respect EXPLICIT command syntax. If the input ALREADY contains literal commands (BROWSER_GO [...], REMEMBER [name], DOWNLOAD [url], PAUSE [...], FOR_EACH_ROW), pass it through verbatim as a single substep. Don't try to decompose those — they're already atomic by construction.

G. Inline routing annotations like "[browser-fast] open youtube" should be treated as their core text: emit ONE substep with the original input text unchanged, INCLUDING the [tag] prefix. The router/loader handles annotations later.

H. Don't ADD steps that weren't implied by the user. If the input is "click submit", don't decompose into "find the submit button, hover over it, then click it". The verifier handles intermediate states.

I. Keep substep counts SMALL. 2-5 substeps for typical compound instructions. 6+ is suspicious — re-read the input and ask whether you're inventing actions.

Worked examples

Input:
0. play the 2nd video on youtube

Output:
[
  {"original_index": 0, "text": "open a new Chrome tab"},
  {"original_index": 0, "text": "go to youtube.com"},
  {"original_index": 0, "text": "scroll down so the video grid is visible"},
  {"original_index": 0, "text": "click the second video in the grid"}
]

Input:
0. log into gmail and check my unread emails

Output:
[
  {"original_index": 0, "text": "open a new Chrome tab"},
  {"original_index": 0, "text": "go to gmail.com"},
  {"original_index": 0, "text": "if a sign-in page appears, click 'Sign in'"},
  {"original_index": 0, "text": "wait for the inbox to load"},
  {"original_index": 0, "text": "click the 'Inbox' label in the left rail"}
]

Input:
0. open chrome
1. go to youtube
2. press Enter

Output:
[
  {"original_index": 0, "text": "open chrome"},
  {"original_index": 1, "text": "go to youtube"},
  {"original_index": 2, "text": "press Enter"}
]
(All three were already atomic. Output passes them through 1:1.)

Input:
0. BROWSER_GO [https://youtube.com]
1. click the search bar at the top of the page

Output:
[
  {"original_index": 0, "text": "BROWSER_GO [https://youtube.com]"},
  {"original_index": 1, "text": "click the search bar at the top of the page"}
]
(Literal command syntax preserved verbatim.)
"""


class DecomposerUnavailable(RuntimeError):
    """Raised when the Gemini decomposer call fails or returns garbage.

    Callers catch this and fall through to the original (unexpanded) step list.
    """


def _build_user_prompt(step_texts: list[str]) -> str:
    """Format the input list for the decomposer.

    Numbered list (0-indexed) so the model can refer to items by index.
    """
    lines = [
        f"{idx}. {text}" for idx, text in enumerate(step_texts)
    ]
    body = "\n".join(lines)
    return (
        "Decompose each of these steps into atomic substeps. Return a "
        "flat list. Every entry must have ``original_index`` pointing "
        "back at the input step it came from.\n\n"
        f"Steps:\n{body}"
    )


def _call_decomposer(
    client: GeminiClient,
    user_prompt: str,
) -> list[_DecomposedStep]:
    """Single Gemini call. Wraps GeminiClient.call_router_raw — same plumbing.

    The router and decomposer share the structured-output infrastructure
    on GeminiClient because both are "single Gemini call with a Pydantic
    response_schema, no image". Reusing avoids duplicating the SDK call /
    error-handling code.
    """
    # ``RouterUnavailable`` is raised by call_router_raw on transport /
    # schema errors. We re-raise as ``DecomposerUnavailable`` so the
    # caller can handle the two failure modes independently if they
    # ever want to (e.g. different fallback strategies).
    from .task_router import RouterUnavailable

    try:
        raw = client.call_router_raw(
            system_prompt=DECOMPOSER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_DecomposerResponseModel,
        )
    except RouterUnavailable as exc:
        raise DecomposerUnavailable(str(exc)) from exc

    if isinstance(raw, _DecomposerResponseModel):
        return raw.steps
    raise DecomposerUnavailable(
        f"decomposer got unexpected response type {type(raw).__name__}"
    )


def decompose_step_texts(
    client: GeminiClient,
    step_texts: list[str],
) -> list[_DecomposedStep]:
    """One Gemini call to decompose all steps. Returns flat ordered list.

    On any error raises ``DecomposerUnavailable``. The caller's fallback
    path is to skip decomposition and proceed with the original list.
    """
    if not step_texts:
        return []

    user_prompt = _build_user_prompt(step_texts)
    try:
        decomposed = _call_decomposer(client, user_prompt)
    except DecomposerUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — catch-all on purpose
        raise DecomposerUnavailable(
            f"decomposer call failed: {exc}"
        ) from exc

    # Validate the response: every entry's original_index must point at
    # a real input step, and every input step must contribute >=1 entry.
    if not decomposed:
        raise DecomposerUnavailable(
            "decomposer returned an empty list"
        )
    seen_indices: set[int] = set()
    for entry in decomposed:
        if entry.original_index < 0 or entry.original_index >= len(step_texts):
            raise DecomposerUnavailable(
                f"decomposer produced original_index={entry.original_index} "
                f"out of range [0, {len(step_texts)})"
            )
        if not entry.text.strip():
            raise DecomposerUnavailable(
                f"decomposer produced an empty text at "
                f"original_index={entry.original_index}"
            )
        seen_indices.add(entry.original_index)
    missing = set(range(len(step_texts))) - seen_indices
    if missing:
        raise DecomposerUnavailable(
            f"decomposer skipped input indices: {sorted(missing)}"
        )

    # Order check: original_index must be non-decreasing across the
    # output (each input's substeps are contiguous, in input order).
    last_idx = -1
    for entry in decomposed:
        if entry.original_index < last_idx:
            raise DecomposerUnavailable(
                "decomposer reordered input steps "
                f"({entry.original_index} after {last_idx})"
            )
        last_idx = entry.original_index

    return decomposed


def apply_decomposition(
    steps: list[TaskStep],
    decomposed: list[_DecomposedStep],
) -> list[TaskStep]:
    """Expand the input ``steps`` list using the decomposer output.

    For each ``DecomposedStep`` we build a new ``TaskStep`` inheriting
    the ORIGINAL step's row metadata (``row_index``, ``csv_name``,
    ``routing_hint``) but with the atomic text. This means a single
    FOR_EACH_ROW expansion of a compound line will produce N×M
    TaskSteps in the final run — exactly what we want.

    Manual ``[tag]`` annotations on the original step survive: the
    router will still see them because we pass through to the
    ``apply_router`` step next. (We're not the only owner of
    routing_hint; this function preserves whatever was set at load
    time, and the router will overwrite when applicable.)
    """
    if not decomposed:
        return list(steps)

    out: list[TaskStep] = []
    for entry in decomposed:
        original = steps[entry.original_index]
        out.append(replace(original, text=entry.text))
    return out


def parse_mode(raw: str | None) -> DecompositionMode:
    """Parse ``TASK_DECOMPOSITION`` env value. Unknown -> AUTO.

    We default to AUTO on unknown input rather than raising because the
    decomposer already has a graceful-fallback path on error, so a typo
    in .env should still leave the agent running.
    """
    if not raw:
        return DecompositionMode.AUTO
    try:
        return DecompositionMode(raw.strip().lower())
    except ValueError:
        log.warning(
            "TASK_DECOMPOSITION=%r not recognised; expected auto/off — "
            "defaulting to auto",
            raw,
        )
        return DecompositionMode.AUTO


def apply_decomposer(
    steps: list[TaskStep],
    *,
    mode: DecompositionMode,
    client: GeminiClient | None,
) -> list[TaskStep]:
    """High-level entry point used by ``agent.run``.

    Returns a (possibly longer) list of TaskSteps. ``mode=off`` is a
    no-op pass-through. ``mode=auto`` calls Gemini and expands compound
    steps; on any error the original list is returned unchanged with a
    warning.
    """
    if not steps or mode == DecompositionMode.OFF:
        return list(steps)
    if client is None:
        log.warning(
            "decomposer: mode=auto but no Gemini client provided; "
            "skipping (treating as off)"
        )
        return list(steps)

    step_texts = [s.text for s in steps]
    try:
        decomposed = decompose_step_texts(client, step_texts)
    except DecomposerUnavailable as exc:
        log.warning(
            "decomposer: auto decomposition failed (%s); proceeding with "
            "original step list",
            exc,
        )
        return list(steps)

    expanded = apply_decomposition(steps, decomposed)
    n_added = len(expanded) - len(steps)
    if n_added > 0:
        log.info(
            "decomposer: expanded %d input step(s) into %d atomic step(s) "
            "(+%d from decomposition)",
            len(steps),
            len(expanded),
            n_added,
        )
    else:
        log.info(
            "decomposer: %d input step(s) were already atomic — passing through",
            len(steps),
        )
    return expanded


__all__ = [
    "DecompositionMode",
    "DecomposerUnavailable",
    "DECOMPOSER_SYSTEM_PROMPT",
    "apply_decomposer",
    "apply_decomposition",
    "decompose_step_texts",
    "parse_mode",
    # Pydantic schemas exported for tests + GeminiClient integration.
    "_DecomposedStep",
    "_DecomposerResponseModel",
]
