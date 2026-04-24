"""Tests for the OCR-backed text finder used by CLICK_TEXT."""
from __future__ import annotations

import sys
import types
from unittest import mock

from PIL import Image

from agent.ocr import _normalize, _similarity, find_text_center


def _fake_pytesseract(data_dict):
    """Build an installable fake `pytesseract` module returning ``data_dict``."""
    fake = types.SimpleNamespace()
    fake.Output = types.SimpleNamespace(DICT="DICT")
    fake.image_to_data = mock.MagicMock(return_value=data_dict)
    return fake


def _install_fake(data_dict):
    fake = _fake_pytesseract(data_dict)
    return mock.patch.dict(sys.modules, {"pytesseract": fake}), fake


def test_normalize_collapses_case_and_punctuation():
    assert _normalize("Hello, World!") == "hello world"


def test_similarity_identical():
    assert _similarity("sign in", "Sign In") == 1.0


def test_similarity_unrelated_is_low():
    assert _similarity("sign in", "cancel") < 0.5


def test_find_text_single_word_match():
    img = Image.new("RGB", (100, 100), "white")
    data = {
        "text": ["", "Submit", "Cancel"],
        "conf": ["-1", "95", "95"],
        "left": [0, 10, 50],
        "top": [0, 10, 10],
        "width": [0, 30, 30],
        "height": [0, 20, 20],
    }
    patch, _fake = _install_fake(data)
    with patch:
        match = find_text_center(img, "Submit")
    assert match is not None
    assert match.text == "Submit"
    assert match.center() == (25, 20)  # center of the 10,10,40,30 bbox


def test_find_text_multi_word_run():
    img = Image.new("RGB", (200, 100), "white")
    data = {
        "text": ["Sign", "in"],
        "conf": ["90", "90"],
        "left": [10, 60],
        "top": [10, 10],
        "width": [40, 30],
        "height": [20, 20],
    }
    patch, _fake = _install_fake(data)
    with patch:
        match = find_text_center(img, "Sign in")
    assert match is not None
    assert match.text == "Sign in"


def test_find_text_confidence_filter():
    img = Image.new("RGB", (200, 100), "white")
    data = {
        "text": ["Submit"],
        "conf": ["10"],  # Below default min_confidence=50
        "left": [10],
        "top": [10],
        "width": [30],
        "height": [20],
    }
    patch, _fake = _install_fake(data)
    with patch:
        match = find_text_center(img, "Submit")
    assert match is None


def test_find_text_returns_none_when_pytesseract_missing():
    img = Image.new("RGB", (100, 100), "white")
    # Remove pytesseract from sys.modules so the import inside find_text_center
    # fails.
    with mock.patch.dict(sys.modules, {"pytesseract": None}):
        match = find_text_center(img, "anything")
    assert match is None


def test_find_text_no_match_below_similarity_threshold():
    img = Image.new("RGB", (200, 100), "white")
    data = {
        "text": ["unrelated", "garbage"],
        "conf": ["95", "95"],
        "left": [10, 60],
        "top": [10, 10],
        "width": [40, 30],
        "height": [20, 20],
    }
    patch, _fake = _install_fake(data)
    with patch:
        match = find_text_center(img, "Sign in")
    assert match is None
