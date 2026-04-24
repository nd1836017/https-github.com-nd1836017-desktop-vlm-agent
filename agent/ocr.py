"""OCR-based text-target finder for the `CLICK_TEXT [label]` primitive.

We use Tesseract via `pytesseract` to extract words + bounding boxes from a
screenshot, then fuzzy-match the requested label against consecutive-word
runs. The closest match (by a simple normalized-edit-distance ratio) whose
confidence clears a threshold wins.

Tesseract is a runtime dependency: the `pytesseract` Python wrapper is in
requirements.txt, but the `tesseract-ocr` system package must also be
installed. If either is missing, `find_text_center` returns None and the
caller falls back to the agent's replan-on-failure path.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from PIL import Image

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRMatch:
    """A matched run of consecutive OCR words."""

    text: str
    # Bounding box (left, top, right, bottom) on the image.
    bbox: tuple[int, int, int, int]
    score: float  # 0.0 to 1.0, higher is better
    confidence: float  # min tesseract confidence across the run

    def center(self) -> tuple[int, int]:
        left, top, right, bottom = self.bbox
        return ((left + right) // 2, (top + bottom) // 2)


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace + strip common punctuation."""
    # Keep letters, digits, and spaces. Collapse whitespace.
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _similarity(a: str, b: str) -> float:
    """Cheap symmetric similarity in [0, 1].

    We use `difflib.SequenceMatcher.ratio()`. It's not the fastest but has no
    extra dependency and behaves well on short UI labels.
    """
    from difflib import SequenceMatcher

    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def find_text_center(
    image: Image.Image,
    label: str,
    *,
    min_confidence: float = 50.0,
    min_similarity: float = 0.7,
    max_span: int = 6,
) -> OCRMatch | None:
    """Locate the best OCR text match for ``label`` in ``image``.

    Args:
        image: PIL image to OCR.
        label: human-readable label to search for.
        min_confidence: discard words with tesseract confidence below this.
        min_similarity: reject matches below this similarity to the label.
        max_span: max number of consecutive OCR words to try as a single run.

    Returns the best match or None. Never raises for missing tesseract — logs
    a warning and returns None so the caller can fall back.
    """
    try:
        import pytesseract
    except ImportError:
        log.warning("pytesseract not installed; CLICK_TEXT cannot run OCR.")
        return None

    try:
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
    except Exception as exc:  # noqa: BLE001 — tesseract can raise many things
        log.warning("pytesseract failed: %s: %s", type(exc).__name__, exc)
        return None

    words = data.get("text", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    # Build a filtered list of (idx, text, conf, bbox) for words with
    # non-empty text and readable confidence.
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for i, w in enumerate(words):
        if not w or not w.strip():
            continue
        try:
            conf = float(confs[i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_confidence:
            continue
        left = int(lefts[i])
        top = int(tops[i])
        right = left + int(widths[i])
        bottom = top + int(heights[i])
        kept.append((w, conf, (left, top, right, bottom)))

    if not kept:
        return None

    best: OCRMatch | None = None

    # Try every consecutive run of up to `max_span` words as a candidate
    # label. A short label like "Sign in" will match a 2-word run; a long
    # one like "Verify it's you" can match a 3-word run.
    for start in range(len(kept)):
        for span in range(1, max_span + 1):
            end = start + span
            if end > len(kept):
                break
            run = kept[start:end]
            text = " ".join(w for w, _, _ in run)
            score = _similarity(text, label)
            if score < min_similarity:
                continue
            min_conf = min(c for _, c, _ in run)
            # Union of bounding boxes.
            lefts_run = [b[0] for _, _, b in run]
            tops_run = [b[1] for _, _, b in run]
            rights_run = [b[2] for _, _, b in run]
            bots_run = [b[3] for _, _, b in run]
            bbox = (
                min(lefts_run),
                min(tops_run),
                max(rights_run),
                max(bots_run),
            )
            candidate = OCRMatch(
                text=text, bbox=bbox, score=score, confidence=min_conf
            )
            if best is None or candidate.score > best.score:
                best = candidate

    return best
