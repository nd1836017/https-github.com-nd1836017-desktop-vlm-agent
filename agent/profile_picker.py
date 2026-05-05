"""Chrome profile-picker helpers — DOM-first label matching.

Chrome's ``chrome://profile-picker/`` page renders profiles as a grid of
visually similar avatar cards. Vision-only matching frequently picks the
wrong card because avatar colour/shape dominates the model's attention
over the small text label below. This module reads the actual DOM
labels via the existing CDP browser bridge so selection is grounded
in label text, not pixels.

Public surface:

* :func:`normalize_profile_label` — pure helper used by both bridge
  and tests. Trim + casefold (Unicode-safe) so "Nhan Doan",
  "  nhan  doan ", and "NHAN DOAN" all compare equal.
* :func:`match_profile_label` — given a list of available labels and a
  target name, return the 0-based index of the best match, or ``None``.
  Prefers an exact (post-normalize) match; falls back to a single
  case-insensitive substring match when there's no exact hit.
* :func:`enumerate_profile_labels` — runtime helper. Talks to the
  browser bridge and returns a list of visible profile card labels
  in DOM order. Returns ``None`` when the bridge isn't connected, the
  active tab is not the profile picker, or DOM extraction fails.
* :func:`click_profile_by_label` — high-level helper. Enumerates
  labels, picks the matching index, and dispatches a click on that
  card. Returns ``(ok, message)`` so the caller can log the result
  and fall back to the visual path.

Design:

* DOM structure of ``chrome://profile-picker/`` is composed of nested
  custom elements with shadow roots: ``profile-picker-app`` →
  ``profile-picker-main-view`` (shadow) → ``profile-card`` (shadow).
  Standard ``document.querySelectorAll`` does NOT cross shadow
  boundaries, so we ship a small recursive walker.
* The walker is intentionally permissive about tag names —
  Chrome's profile-picker internals have changed shape over Chrome
  versions (``profile-card``, ``profile-card-tile``, etc.), so we
  match on a substring of the tag name rather than an exact name.
* All CDP calls are wrapped in try/except: any failure returns a
  graceful fallback (``None`` for enumerate, ``(False, message)``
  for click). The agent loop can then fall back to vision/OCR.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.browser_bridge import BrowserBridge

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers — no I/O, fully unit-testable.
# ---------------------------------------------------------------------------

def normalize_profile_label(label: str) -> str:
    """Trim, NFKC-normalize, and casefold a profile label for comparison.

    Profile names can contain leading/trailing whitespace, multiple
    spaces between tokens (e.g. ``"Nhan  Doan"``), and unicode quirks
    (full-width characters, accented letters). NFKC normalization
    collapses compatibility characters; casefold is the unicode-aware
    equivalent of lower (it correctly handles e.g. German ß).
    """
    if not label:
        return ""
    # NFKC handles full-width / half-width / compat-decomposable chars.
    text = unicodedata.normalize("NFKC", label)
    # Collapse internal whitespace runs to a single space so
    # "Nhan  Doan" matches "Nhan Doan".
    text = " ".join(text.split())
    return text.casefold()


def match_profile_label(
    available: list[str],
    target: str,
) -> int | None:
    """Return the 0-based index of ``target`` in ``available`` or ``None``.

    Resolution order:

    1. Exact match after :func:`normalize_profile_label` on both sides.
       If multiple cards have the same normalized label we return the
       first one — Chrome itself doesn't allow duplicate exact names,
       but a stray whitespace-only diff between two cards would
       collapse to identical normalized form.
    2. Single substring match. The target is the haystack; each
       available label is the needle. Useful when the target is a
       full name like "Nhan Doan" and the card label is just "Nhan",
       or vice-versa. We require *exactly one* substring match —
       multiple hits is ambiguous and we'd rather fail loudly than
       click the wrong profile.

    Returns ``None`` when neither rule produces a confident match.
    """
    if not target or not available:
        return None
    norm_target = normalize_profile_label(target)
    if not norm_target:
        return None

    norm_available = [normalize_profile_label(label) for label in available]

    # Step 1: exact match.
    for idx, label in enumerate(norm_available):
        if label == norm_target:
            return idx

    # Step 2: bidirectional substring match.
    matches: list[int] = []
    for idx, label in enumerate(norm_available):
        if not label:
            continue
        if norm_target in label or label in norm_target:
            matches.append(idx)
    if len(matches) == 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Shadow-DOM walker JS — embedded once, formatted into Runtime.evaluate exprs.
# ---------------------------------------------------------------------------

# A tiny recursive walker that collects every element under ``root``
# (descending into open shadow roots), then keeps those whose
# ``tagName`` contains ``profile-card`` (case-insensitive). For each
# we return the visible text content, normalised — this mirrors how
# the user perceives the card: the top-line name underneath the
# avatar.
_WALK_PROFILE_CARDS_JS = r"""
(() => {
  function walk(root, out) {
    if (!root) return;
    out.push(root);
    const children = root.children || [];
    for (let i = 0; i < children.length; i++) walk(children[i], out);
    if (root.shadowRoot) walk(root.shadowRoot, out);
  }
  const all = [];
  walk(document.documentElement, all);
  const cards = [];
  for (const node of all) {
    const tag = (node.tagName || "").toLowerCase();
    // Match common Chrome profile-picker custom-element tag names.
    // Substring match keeps us forward-compatible across Chrome
    // versions (profile-card, profile-card-tile, profile-row, ...).
    if (tag.includes("profile-card") || tag === "profile-row") {
      const text = (node.innerText || node.textContent || "").trim();
      cards.push(text);
    }
  }
  return JSON.stringify(cards);
})()
"""


# Click the Nth profile card (0-based). Returns "OK" on success,
# "OUT_OF_RANGE" when the index is past the array length, "NOT_FOUND"
# when the walker found zero cards. The walker is duplicated here so
# the click is atomic — if the DOM is being rebuilt between
# enumerate and click, we'd rather fail than click stale state.
_CLICK_PROFILE_CARD_JS_TEMPLATE = r"""
(() => {
  function walk(root, out) {
    if (!root) return;
    out.push(root);
    const children = root.children || [];
    for (let i = 0; i < children.length; i++) walk(children[i], out);
    if (root.shadowRoot) walk(root.shadowRoot, out);
  }
  const all = [];
  walk(document.documentElement, all);
  const cards = [];
  for (const node of all) {
    const tag = (node.tagName || "").toLowerCase();
    if (tag.includes("profile-card") || tag === "profile-row") {
      cards.push(node);
    }
  }
  if (cards.length === 0) return "NOT_FOUND";
  const idx = %d;
  if (idx < 0 || idx >= cards.length) return "OUT_OF_RANGE";
  // Some profile-card variants nest a real <button> with the click
  // handler; click it directly when present so we trigger the
  // proper handler. Fall back to clicking the card itself.
  const card = cards[idx];
  let target = card.querySelector ? card.querySelector("button") : null;
  if (!target && card.shadowRoot) {
    target = card.shadowRoot.querySelector("button");
  }
  if (!target) target = card;
  target.click();
  return "OK";
})()
"""


# ---------------------------------------------------------------------------
# Bridge-aware helpers.
# ---------------------------------------------------------------------------

def is_profile_picker_url(url: str) -> bool:
    """Recognise the profile-picker URL across Chrome variants.

    Chrome and Chromium both use ``chrome://profile-picker/`` (with or
    without trailing slash, sometimes ``chrome://profile-picker/main-view``).
    Brave / Edge / etc. use the same scheme. We accept any URL whose
    path begins with ``//profile-picker``.
    """
    if not url:
        return False
    lowered = url.lower()
    return "://profile-picker" in lowered


def enumerate_profile_labels(
    bridge: BrowserBridge | None,
) -> list[str] | None:
    """Read visible profile-card labels from the active tab.

    Returns:
      * ``list[str]`` — labels in DOM order (may be empty if Chrome
        rendered the picker but no profiles exist; caller decides).
      * ``None`` — the bridge is unavailable, the active tab isn't the
        profile picker, or the JS walker raised. In all these cases
        the caller should fall back to vision/OCR.
    """
    if bridge is None or not bridge.is_connected():
        return None
    # Bridge has no public method to read the active URL; do it inline.
    try:
        url_result = bridge._send(  # noqa: SLF001 — internal use only
            "Runtime.evaluate",
            {"expression": "location.href", "returnByValue": True},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("profile_picker: could not read active tab URL: %s", exc)
        return None
    url_value = _extract_value(url_result)
    if not isinstance(url_value, str) or not is_profile_picker_url(url_value):
        log.info(
            "profile_picker: active tab is not the profile picker (url=%r); "
            "falling back to vision.",
            url_value,
        )
        return None
    try:
        result = bridge._send(  # noqa: SLF001
            "Runtime.evaluate",
            {"expression": _WALK_PROFILE_CARDS_JS, "returnByValue": True},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("profile_picker: walker eval failed: %s", exc)
        return None
    raw = _extract_value(result)
    if not isinstance(raw, str):
        log.warning(
            "profile_picker: walker returned non-string (%r); falling back.",
            raw,
        )
        return None
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("profile_picker: malformed JSON from walker: %s", exc)
        return None
    if not isinstance(labels, list):
        return None
    # Coerce each element to str for safety; drop anything that isn't.
    return [str(item) for item in labels if item is not None]


def click_profile_by_label(
    bridge: BrowserBridge | None,
    target_name: str,
) -> tuple[bool, str]:
    """High-level helper. Find the profile by label and click it.

    Returns ``(ok, message)`` so the caller can log + fall back.
    Failure modes (all return ``ok=False``):

    * Bridge unavailable / not on the picker page.
    * Walker found no profile cards.
    * No card label matches ``target_name``.
    * Click expression raised or returned an unexpected status.
    """
    labels = enumerate_profile_labels(bridge)
    if labels is None:
        return False, (
            "profile-picker: cannot read DOM labels "
            "(bridge unavailable or not on chrome://profile-picker/)"
        )
    if not labels:
        return False, "profile-picker: walker found zero profile cards"

    # Pre-click logging so failure postmortems show what we saw.
    log.info(
        "profile-picker: enumerated %d card label(s): %r",
        len(labels),
        labels,
    )

    idx = match_profile_label(labels, target_name)
    if idx is None:
        return False, (
            f"profile-picker: no card label matches {target_name!r} "
            f"(seen labels: {labels!r})"
        )
    chosen = labels[idx]
    log.info(
        "profile-picker: matched target %r → card #%d (label=%r); clicking.",
        target_name,
        idx,
        chosen,
    )
    expr = _CLICK_PROFILE_CARD_JS_TEMPLATE % idx
    try:
        result = bridge._send(  # noqa: SLF001
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True},
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"profile-picker: click eval raised: {exc}"
    status = _extract_value(result)
    if status == "OK":
        return True, (
            f"profile-picker: clicked card #{idx} (label={chosen!r}) "
            f"matching target {target_name!r}"
        )
    if status == "NOT_FOUND":
        return False, "profile-picker: walker found zero cards on click"
    if status == "OUT_OF_RANGE":
        return False, (
            f"profile-picker: card #{idx} out of range (DOM changed "
            f"between enumerate and click)"
        )
    return False, f"profile-picker: unexpected click status {status!r}"


def _extract_value(cdp_result: dict) -> object:
    """Pull the .result.value out of a Runtime.evaluate response."""
    inner = cdp_result.get("result") if isinstance(cdp_result, dict) else None
    if isinstance(inner, dict):
        return inner.get("value")
    return None


__all__ = [
    "normalize_profile_label",
    "match_profile_label",
    "is_profile_picker_url",
    "enumerate_profile_labels",
    "click_profile_by_label",
]
