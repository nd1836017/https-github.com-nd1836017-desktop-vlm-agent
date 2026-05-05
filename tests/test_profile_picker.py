"""Unit tests for agent.profile_picker.

Pure-helper tests cover label normalization + matching. The bridge-aware
helpers (enumerate_profile_labels, click_profile_by_label) are tested
with a fake bridge so we don't need a real Chrome instance.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from agent.profile_picker import (
    click_profile_by_label,
    enumerate_profile_labels,
    is_profile_picker_url,
    match_profile_label,
    normalize_profile_label,
)

# -----------------------------------------------------------------------------
# normalize_profile_label
# -----------------------------------------------------------------------------

def test_normalize_strips_outer_whitespace():
    assert normalize_profile_label("  Nhan Doan  ") == "nhan doan"


def test_normalize_collapses_internal_whitespace():
    assert normalize_profile_label("Nhan   Doan") == "nhan doan"
    assert normalize_profile_label("Nhan\tDoan") == "nhan doan"
    assert normalize_profile_label("Nhan\n\tDoan") == "nhan doan"


def test_normalize_is_unicode_casefold_not_lower():
    # German ß: lower() returns "ß"; casefold returns "ss".
    assert normalize_profile_label("STRAßE") == "strasse"


def test_normalize_handles_full_width_chars_via_nfkc():
    # Full-width letters compatibility-decompose to ASCII.
    assert normalize_profile_label("Ｎｈａｎ") == "nhan"


def test_normalize_empty_returns_empty():
    assert normalize_profile_label("") == ""
    assert normalize_profile_label("   ") == ""


# -----------------------------------------------------------------------------
# match_profile_label — exact-match path
# -----------------------------------------------------------------------------

def test_match_exact_returns_correct_index():
    labels = ["Work", "Nhan Doan", "Personal"]
    assert match_profile_label(labels, "Nhan Doan") == 1


def test_match_exact_is_case_insensitive():
    labels = ["NHAN DOAN", "Other"]
    assert match_profile_label(labels, "nhan doan") == 0


def test_match_exact_ignores_extra_whitespace():
    labels = ["  Nhan  Doan  ", "Other"]
    assert match_profile_label(labels, "Nhan Doan") == 0


def test_match_returns_first_index_on_normalized_duplicates():
    # Two cards with identical normalized form (rare but possible).
    labels = ["Nhan Doan", "nhan doan"]
    assert match_profile_label(labels, "Nhan Doan") == 0


# -----------------------------------------------------------------------------
# match_profile_label — substring fallback
# -----------------------------------------------------------------------------

def test_match_substring_target_in_label():
    # Card label is "Nhan Doan (Work)"; user typed "Nhan Doan".
    labels = ["Other Person", "Nhan Doan (Work)"]
    assert match_profile_label(labels, "Nhan Doan") == 1


def test_match_substring_label_in_target():
    # Card label is just "Nhan"; user typed full "Nhan Doan".
    labels = ["Other", "Nhan"]
    assert match_profile_label(labels, "Nhan Doan") == 1


def test_match_substring_ambiguous_returns_none():
    # Two cards both contain the target: ambiguous, refuse.
    labels = ["Nhan Doan (Work)", "Nhan Doan (Personal)"]
    assert match_profile_label(labels, "Nhan Doan") is None


def test_match_no_substring_match_returns_none():
    labels = ["Alice", "Bob"]
    assert match_profile_label(labels, "Nhan Doan") is None


# -----------------------------------------------------------------------------
# match_profile_label — edge cases
# -----------------------------------------------------------------------------

def test_match_empty_target_returns_none():
    assert match_profile_label(["Alice"], "") is None
    assert match_profile_label(["Alice"], "   ") is None


def test_match_empty_available_returns_none():
    assert match_profile_label([], "Nhan Doan") is None


def test_match_skips_empty_labels_in_substring_path():
    # An empty label in available shouldn't match every target.
    labels = ["", "Nhan Doan"]
    assert match_profile_label(labels, "Nhan") == 1


# -----------------------------------------------------------------------------
# is_profile_picker_url
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "chrome://profile-picker/",
        "chrome://profile-picker",
        "chrome://profile-picker/main-view",
        "CHROME://Profile-Picker/",  # case insensitivity
    ],
)
def test_is_profile_picker_url_accepts_picker_variants(url: str):
    assert is_profile_picker_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://google.com",
        "chrome://newtab/",
        "chrome://settings/",
        "about:blank",
    ],
)
def test_is_profile_picker_url_rejects_others(url: str):
    assert is_profile_picker_url(url) is False


# -----------------------------------------------------------------------------
# Fake bridge for I/O paths.
# -----------------------------------------------------------------------------

class _FakeBridge:
    """Minimal stand-in for BrowserBridge.

    Records every CDP method call. Returns canned responses for the
    URL probe and the walker eval. Tests construct one per case with
    the desired URL + walker output (or an Exception to simulate a
    transport error).
    """

    def __init__(
        self,
        *,
        connected: bool = True,
        url: str = "chrome://profile-picker/",
        labels_json: str | Exception = '["Work", "Nhan Doan", "Personal"]',
        click_status: str | Exception = "OK",
    ) -> None:
        self._connected = connected
        self._url = url
        self._labels_json = labels_json
        self._click_status = click_status
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def is_connected(self) -> bool:
        return self._connected

    def _send(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method != "Runtime.evaluate":
            return {"result": {"value": None}}
        expr = params.get("expression", "")
        # URL probe.
        if expr == "location.href":
            return {"result": {"value": self._url}}
        # Walker enumerate.
        if "JSON.stringify(cards)" in expr:
            if isinstance(self._labels_json, Exception):
                raise self._labels_json
            return {"result": {"value": self._labels_json}}
        # Click expression.
        if "target.click()" in expr:
            if isinstance(self._click_status, Exception):
                raise self._click_status
            return {"result": {"value": self._click_status}}
        return {"result": {"value": None}}


# -----------------------------------------------------------------------------
# enumerate_profile_labels
# -----------------------------------------------------------------------------

def test_enumerate_returns_labels_when_on_picker_page():
    bridge = _FakeBridge(labels_json='["Work", "Nhan Doan"]')
    labels = enumerate_profile_labels(bridge)
    assert labels == ["Work", "Nhan Doan"]


def test_enumerate_returns_none_when_bridge_disconnected():
    bridge = _FakeBridge(connected=False)
    assert enumerate_profile_labels(bridge) is None


def test_enumerate_returns_none_when_bridge_is_none():
    assert enumerate_profile_labels(None) is None


def test_enumerate_returns_none_when_url_is_not_picker():
    bridge = _FakeBridge(url="https://google.com")
    assert enumerate_profile_labels(bridge) is None


def test_enumerate_returns_none_when_walker_raises():
    bridge = _FakeBridge(labels_json=RuntimeError("transport"))
    assert enumerate_profile_labels(bridge) is None


def test_enumerate_returns_none_on_malformed_json():
    bridge = _FakeBridge(labels_json="not-json")
    assert enumerate_profile_labels(bridge) is None


def test_enumerate_returns_empty_list_when_no_cards():
    bridge = _FakeBridge(labels_json="[]")
    assert enumerate_profile_labels(bridge) == []


# -----------------------------------------------------------------------------
# click_profile_by_label — end-to-end
# -----------------------------------------------------------------------------

def test_click_picks_correct_card_from_labels():
    bridge = _FakeBridge(
        labels_json=json.dumps(["Work", "Nhan Doan", "Personal"]),
    )
    ok, msg = click_profile_by_label(bridge, "Nhan Doan")
    assert ok is True
    assert "card #1" in msg
    assert "Nhan Doan" in msg
    # Final call should be the click expression with idx=1.
    click_calls = [
        c for c in bridge.calls
        if c[0] == "Runtime.evaluate" and "target.click()" in c[1]["expression"]
    ]
    assert len(click_calls) == 1
    assert "const idx = 1;" in click_calls[0][1]["expression"]


def test_click_fails_gracefully_when_no_match():
    bridge = _FakeBridge(labels_json=json.dumps(["Work", "Personal"]))
    ok, msg = click_profile_by_label(bridge, "Nhan Doan")
    assert ok is False
    assert "no card label matches" in msg
    assert "Nhan Doan" in msg


def test_click_fails_gracefully_when_disconnected():
    bridge = _FakeBridge(connected=False)
    ok, msg = click_profile_by_label(bridge, "Nhan Doan")
    assert ok is False
    assert "cannot read DOM labels" in msg


def test_click_fails_gracefully_on_walker_empty():
    bridge = _FakeBridge(labels_json="[]")
    ok, msg = click_profile_by_label(bridge, "Nhan Doan")
    assert ok is False
    assert "zero profile cards" in msg


def test_click_handles_out_of_range_status():
    # Walker returns 1 card but the click reports OUT_OF_RANGE
    # (DOM rebuilt between enumerate and click).
    bridge = _FakeBridge(
        labels_json=json.dumps(["Nhan Doan"]),
        click_status="OUT_OF_RANGE",
    )
    ok, msg = click_profile_by_label(bridge, "Nhan Doan")
    assert ok is False
    assert "out of range" in msg


def test_click_handles_unexpected_status():
    bridge = _FakeBridge(
        labels_json=json.dumps(["Nhan Doan"]),
        click_status="WEIRD",
    )
    ok, msg = click_profile_by_label(bridge, "Nhan Doan")
    assert ok is False
    assert "unexpected click status" in msg


def test_click_succeeds_via_substring_fallback():
    # Card label has extra suffix; substring fallback should match.
    bridge = _FakeBridge(
        labels_json=json.dumps(["Work (Office)", "Nhan Doan (Personal)"]),
    )
    ok, msg = click_profile_by_label(bridge, "Nhan Doan")
    assert ok is True
    assert "card #1" in msg
