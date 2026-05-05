"""Self-correction memory across runs.

After a run finishes successfully the agent condenses the action log
into a short natural-language hint ("to log in to Gmail, click the email
field, type the address, then press Tab and Enter") and saves it under
a signature derived from the original tasks.txt content. On the next
run the planner is seeded with that hint via the ``prior_run_hint``
parameter on :py:meth:`agent.vlm.GeminiClient.plan_action`.

Design:

* Storage is a single JSON file (configurable via ``RUN_MEMORY_DIR``)
  so the user can inspect / clear / commit it as they choose. Entries
  are append-only with last-N-per-signature eviction so the file
  doesn't grow unbounded.
* The signature is a hash of the *normalised* task list (whitespace
  + casefold). Cosmetic edits to ``tasks.txt`` still hit the cache;
  semantically distinct tasks don't collide.
* The end-of-run summary is one Gemini call. If it fails, we fall
  back to a deterministic action-list summary so the memory entry
  still gets written. Skipped on failure (only successful runs are
  remembered, otherwise we'd re-suggest broken paths).
* Lookup is exact-signature only for now. Fuzzy / semantic matching
  can be layered on later without changing the on-disk format.

Public surface (all importable from the top-level module):

    normalize_task_line(text) -> str
    compute_signature(tasks) -> str
    summarize_run_actions(actions) -> str
    RunMemoryEntry
    RunMemoryStore
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.vlm import GeminiClient

log = logging.getLogger(__name__)


# Bump this when the on-disk schema changes incompatibly. Older files
# are quietly ignored on load (treated as empty store) so old runs
# don't crash a fresh install.
_SCHEMA_VERSION = 1

# Hard caps to keep the JSON file readable by humans + reasonable to
# scan on startup. The defaults are conservative; per-store overrides
# come from the env-driven config layer.
_DEFAULT_MAX_PER_SIGNATURE = 3
_DEFAULT_MAX_AGE_DAYS = 30
_DEFAULT_FILENAME = "run_memory.json"

# Cap how many actions we feed to the summariser. A 200-step run with
# verbose history would overrun the prompt and waste tokens — the last
# 30 actions are plenty of context for "what worked".
_SUMMARIZE_MAX_ACTIONS = 30


# ---------------------------------------------------------------------------
# Pure helpers — fully unit-testable.
# ---------------------------------------------------------------------------

def normalize_task_line(text: str) -> str:
    """Normalise a single tasks.txt line for signature comparison.

    NFKC + casefold + collapse whitespace runs. Strips leading/trailing
    whitespace. Returns ``""`` for blank/None input. Mirrors the
    profile-picker normaliser so cross-feature comparisons behave
    consistently.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = " ".join(out.split())
    return out.casefold()


def compute_signature(tasks: list[str]) -> str:
    """Return a stable SHA-1 signature for the *content* of a task list.

    Tasks are normalised line-by-line then joined with ``"\\n"`` before
    hashing. Empty / whitespace-only lines are dropped so a trailing
    blank line doesn't shift the signature. Returns the empty string
    for an empty (post-normalisation) task list — callers can treat
    that as "do not record / look up".
    """
    if not tasks:
        return ""
    cleaned = [normalize_task_line(t) for t in tasks]
    cleaned = [t for t in cleaned if t]
    if not cleaned:
        return ""
    payload = "\n".join(cleaned).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def summarize_run_actions_deterministic(actions: list[str]) -> str:
    """Produce a deterministic 1-line summary of an action list.

    Used as a fallback when the VLM call fails. The summary lists each
    action's verb (everything before the first ``[``) joined by ", "
    so the output stays compact. TYPE and BROWSER_FILL values were
    already privacy-redacted upstream so this is safe to persist.
    """
    if not actions:
        return ""
    verbs: list[str] = []
    for raw in actions:
        token = raw.strip()
        if not token:
            continue
        # Split at the first space or '[' — both delimit the verb from
        # its arguments in the canonical render form.
        for delim in (" [", " ", "["):
            head = token.split(delim, 1)[0]
            if head and head != token:
                token = head
                break
        verbs.append(token)
    if not verbs:
        return ""
    return "Previously this task was completed with: " + ", ".join(verbs) + "."


# ---------------------------------------------------------------------------
# Memory entry + store.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunMemoryEntry:
    """One successful-run memory record.

    ``tasks_normalised`` is stored alongside the signature as a
    debugging aid — when a signature collision (extremely unlikely
    with sha1 over typical task lists) does happen, the user can see
    which run was overwritten.
    """

    signature: str
    tasks_normalised: list[str]
    summary: str
    actions: list[str] = field(default_factory=list)
    step_count: int = 0
    recorded_at: float = 0.0  # unix epoch seconds; float for sub-second
    run_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMemoryEntry | None:
        """Best-effort reconstruction from JSON. Returns None on bad shape."""
        try:
            return cls(
                signature=str(data["signature"]),
                tasks_normalised=[str(t) for t in data.get("tasks_normalised", [])],
                summary=str(data.get("summary", "")),
                actions=[str(a) for a in data.get("actions", [])],
                step_count=int(data.get("step_count", 0)),
                recorded_at=float(data.get("recorded_at", 0.0)),
                run_id=str(data.get("run_id", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("run_memory: dropping malformed entry: %s", exc)
            return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def age_days(self, now: float | None = None) -> float:
        """Age of this entry in days. ``now`` defaults to ``time.time()``."""
        if now is None:
            now = time.time()
        return max(0.0, (now - self.recorded_at) / 86_400.0)


class RunMemoryStore:
    """JSON-backed store of past successful runs.

    The store is intentionally small + simple: load on init, mutate
    in-memory, save explicitly. Concurrency is not supported — runs
    are sequential by design.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_per_signature: int = _DEFAULT_MAX_PER_SIGNATURE,
        max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
    ) -> None:
        self._path = path
        self._max_per_signature = max(1, int(max_per_signature))
        self._max_age_days = float(max_age_days)
        self._entries: list[RunMemoryEntry] = []
        self._loaded = False

    # ------------------------------------------------------------------ I/O

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        """Read entries from disk. Missing / corrupt files start empty."""
        self._loaded = True
        if not self._path.exists():
            self._entries = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "run_memory: %s unreadable (%s); starting empty.",
                self._path,
                exc,
            )
            self._entries = []
            return
        if not isinstance(raw, dict):
            log.warning(
                "run_memory: %s has unexpected top-level type %s; "
                "starting empty.",
                self._path,
                type(raw).__name__,
            )
            self._entries = []
            return
        version = raw.get("version")
        if version != _SCHEMA_VERSION:
            log.info(
                "run_memory: %s schema version %r != %d; starting empty.",
                self._path,
                version,
                _SCHEMA_VERSION,
            )
            self._entries = []
            return
        items = raw.get("entries", [])
        if not isinstance(items, list):
            self._entries = []
            return
        loaded: list[RunMemoryEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = RunMemoryEntry.from_dict(item)
            if entry is not None:
                loaded.append(entry)
        # Drop expired entries on load so a stale memory file doesn't
        # poison the next run.
        self._entries = [
            e for e in loaded if e.age_days() <= self._max_age_days
        ]

    def save(self) -> None:
        """Persist the current entries to disk atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _SCHEMA_VERSION,
            "entries": [e.to_dict() for e in self._entries],
        }
        # Write to a tmp file then rename — avoids a half-written file
        # if the process is killed mid-flush.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    # -------------------------------------------------------------- queries

    def all_entries(self) -> list[RunMemoryEntry]:
        """Return all currently-stored entries (a defensive copy)."""
        if not self._loaded:
            self.load()
        return list(self._entries)

    def lookup(self, signature: str) -> list[RunMemoryEntry]:
        """Return matching entries for ``signature``, newest first."""
        if not signature:
            return []
        if not self._loaded:
            self.load()
        matches = [e for e in self._entries if e.signature == signature]
        matches.sort(key=lambda e: e.recorded_at, reverse=True)
        return matches

    def latest(self, signature: str) -> RunMemoryEntry | None:
        """Convenience: most recent entry for ``signature`` or None."""
        matches = self.lookup(signature)
        return matches[0] if matches else None

    # -------------------------------------------------------------- mutation

    def record(self, entry: RunMemoryEntry) -> None:
        """Insert ``entry`` and apply per-signature + age eviction.

        The newest entry per signature wins on tie; older ones are
        dropped to ``max_per_signature``. Caller is responsible for
        calling :meth:`save` afterward.
        """
        if not self._loaded:
            self.load()
        if not entry.signature:
            log.debug("run_memory: refusing to record entry with empty signature.")
            return
        self._entries.append(entry)
        self._evict()

    def clear(self) -> None:
        """Drop every entry (caller persists via :meth:`save`)."""
        self._entries = []
        self._loaded = True

    def _evict(self) -> None:
        # Drop entries older than max_age_days.
        now = time.time()
        self._entries = [
            e for e in self._entries
            if (now - e.recorded_at) / 86_400.0 <= self._max_age_days
        ]
        # Then keep last-N per signature, newest first.
        kept: list[RunMemoryEntry] = []
        per_sig_counts: dict[str, int] = {}
        for entry in sorted(
            self._entries, key=lambda e: e.recorded_at, reverse=True
        ):
            count = per_sig_counts.get(entry.signature, 0)
            if count < self._max_per_signature:
                kept.append(entry)
                per_sig_counts[entry.signature] = count + 1
        # Restore chronological order so the file reads naturally.
        kept.sort(key=lambda e: e.recorded_at)
        self._entries = kept


# ---------------------------------------------------------------------------
# Summarisation — Gemini call wrapped so callers can pass any client.
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = """\
You are summarising a SUCCESSFUL automation run so a future run with the same goal can be seeded with what worked.

Goal of the run (from tasks.txt, one bullet per line):
{goals}

Action sequence the agent took, in order (one per line; values inside [] may be redacted):
{actions}

Write ONE concise paragraph (target 1-2 sentences, max 60 words) describing what the run did and the key actions that worked, like a postmortem note left for a future agent. Mention concrete elements (button labels, fields, key presses) but do NOT include specific pixel coordinates — those rarely transfer. Lead with the goal verb in lowercase ("to log in to gmail, …").

Respond with ONLY the paragraph. No preamble, no JSON, no quotes."""


def summarize_run_actions(
    *,
    client: GeminiClient | None,
    tasks: list[str],
    actions: list[str],
) -> str:
    """Ask the VLM client to summarise a successful run.

    Falls back to :func:`summarize_run_actions_deterministic` on any
    error (network, schema, empty response). The fallback summary is
    less useful for cross-task transfer but still better than nothing
    for the next run's planner.
    """
    deterministic = summarize_run_actions_deterministic(actions)
    if client is None:
        return deterministic
    if not actions:
        return deterministic
    trimmed = actions[-_SUMMARIZE_MAX_ACTIONS:]
    prompt = _SUMMARIZE_PROMPT.format(
        goals="\n".join(f"- {t}" for t in tasks if t.strip()) or "- (none)",
        actions="\n".join(trimmed),
    )
    try:
        text = client.summarize_text(prompt)  # type: ignore[attr-defined]
    except (AttributeError, Exception) as exc:  # noqa: BLE001
        log.warning(
            "run_memory: summariser failed (%s); using deterministic fallback.",
            exc,
        )
        return deterministic
    if not isinstance(text, str):
        return deterministic
    text = text.strip().strip('"').strip("'").strip()
    if not text:
        return deterministic
    return text


def format_prior_run_hint(entry: RunMemoryEntry) -> str:
    """Render a memory entry for injection into the planner prompt.

    Wording explicitly tells the planner the hint is advisory and that
    coordinates from past runs should NOT be replayed blindly — stale
    pixel positions are the most common failure mode for naive
    "remember what worked" caches.
    """
    if not entry.summary:
        return ""
    age = entry.age_days()
    when = (
        f"~{age:.1f}d ago" if age >= 1 else
        f"~{age * 24:.1f}h ago" if age * 24 >= 1 else
        "just now"
    )
    return (
        f"You have completed this task successfully before ({when}). "
        f"Here's a hint from that run: {entry.summary}\n"
        "Use this as a starting point, but ADAPT to the current screen — "
        "do not replay coordinates blindly."
    )


__all__ = [
    "RunMemoryEntry",
    "RunMemoryStore",
    "compute_signature",
    "format_prior_run_hint",
    "normalize_task_line",
    "summarize_run_actions",
    "summarize_run_actions_deterministic",
]
