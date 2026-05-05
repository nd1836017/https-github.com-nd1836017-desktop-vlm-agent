"""Tests for agent.run_memory.

Pure helpers (signature, normalisation, deterministic summary) are
covered first. Then the I/O store: load/save roundtrip, eviction
(per-signature + age), corrupt-file resilience, lookup ordering.
Finally summarize_run_actions is tested with a fake VLM client.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent.run_memory import (
    RunMemoryEntry,
    RunMemoryStore,
    compute_signature,
    format_prior_run_hint,
    normalize_task_line,
    summarize_run_actions,
    summarize_run_actions_deterministic,
)

# -----------------------------------------------------------------------------
# normalize_task_line
# -----------------------------------------------------------------------------

def test_normalize_strips_outer_whitespace():
    assert normalize_task_line("  open youtube  ") == "open youtube"


def test_normalize_collapses_internal_whitespace():
    assert normalize_task_line("open\t\tyoutube") == "open youtube"
    assert normalize_task_line("open  youtube") == "open youtube"


def test_normalize_casefold():
    assert normalize_task_line("OPEN Youtube") == "open youtube"


def test_normalize_unicode_compat():
    assert normalize_task_line("Ｏｐｅｎ") == "open"


def test_normalize_empty():
    assert normalize_task_line("") == ""
    assert normalize_task_line("   ") == ""


# -----------------------------------------------------------------------------
# compute_signature
# -----------------------------------------------------------------------------

def test_signature_is_stable():
    sig1 = compute_signature(["open youtube", "play first video"])
    sig2 = compute_signature(["open youtube", "play first video"])
    assert sig1 == sig2
    assert len(sig1) == 40  # sha1 hex


def test_signature_normalises_whitespace_and_case():
    a = compute_signature(["Open YouTube", "play  first VIDEO"])
    b = compute_signature(["open youtube", "play first video"])
    assert a == b


def test_signature_drops_blank_lines():
    a = compute_signature(["open youtube", "", "  ", "play first video"])
    b = compute_signature(["open youtube", "play first video"])
    assert a == b


def test_signature_changes_when_task_changes():
    a = compute_signature(["open youtube"])
    b = compute_signature(["open gmail"])
    assert a != b


def test_signature_empty_task_list():
    assert compute_signature([]) == ""
    assert compute_signature(["", "  "]) == ""


# -----------------------------------------------------------------------------
# summarize_run_actions_deterministic
# -----------------------------------------------------------------------------

def test_deterministic_summary_pulls_verbs():
    actions = [
        "BROWSER_GO [https://gmail.com]",
        "TYPE [<REDACTED, 12 chars>]",
        "PRESS [tab]",
        "PRESS [enter]",
    ]
    summary = summarize_run_actions_deterministic(actions)
    assert summary.startswith("Previously this task was completed with:")
    # Every verb should appear.
    for verb in ("BROWSER_GO", "TYPE", "PRESS"):
        assert verb in summary


def test_deterministic_summary_empty_actions():
    assert summarize_run_actions_deterministic([]) == ""
    assert summarize_run_actions_deterministic(["", "  "]) == ""


# -----------------------------------------------------------------------------
# RunMemoryEntry
# -----------------------------------------------------------------------------

def test_entry_to_from_dict_roundtrip():
    entry = RunMemoryEntry(
        signature="abc123",
        tasks_normalised=["open youtube"],
        summary="to open youtube, click the address bar then type and press enter.",
        actions=["BROWSER_GO [https://youtube.com]"],
        step_count=1,
        recorded_at=1700000000.0,
        run_id="run-x",
    )
    rebuilt = RunMemoryEntry.from_dict(entry.to_dict())
    assert rebuilt == entry


def test_entry_from_dict_drops_malformed():
    assert RunMemoryEntry.from_dict({"signature": "x"}) == RunMemoryEntry(
        signature="x", tasks_normalised=[], summary="", actions=[],
        step_count=0, recorded_at=0.0, run_id="",
    )
    # Missing the required signature field returns None.
    assert RunMemoryEntry.from_dict({}) is None


def test_entry_age_days():
    now = time.time()
    e = RunMemoryEntry(
        signature="x", tasks_normalised=[], summary="", actions=[],
        step_count=0, recorded_at=now - 86400 * 5, run_id="",
    )
    assert 4.9 <= e.age_days(now) <= 5.1


# -----------------------------------------------------------------------------
# RunMemoryStore — load/save/lookup
# -----------------------------------------------------------------------------

def _make_entry(*, sig: str, when: float, summary: str = "hint") -> RunMemoryEntry:
    return RunMemoryEntry(
        signature=sig,
        tasks_normalised=["task"],
        summary=summary,
        actions=["BROWSER_GO [u]"],
        step_count=1,
        recorded_at=when,
        run_id="",
    )


def test_store_load_missing_file_starts_empty(tmp_path: Path):
    store = RunMemoryStore(tmp_path / "memory.json")
    store.load()
    assert store.all_entries() == []


def test_store_load_corrupt_starts_empty(tmp_path: Path):
    p = tmp_path / "memory.json"
    p.write_text("not json", encoding="utf-8")
    store = RunMemoryStore(p)
    store.load()
    assert store.all_entries() == []


def test_store_load_wrong_schema_starts_empty(tmp_path: Path):
    p = tmp_path / "memory.json"
    p.write_text(json.dumps({"version": 999, "entries": []}), encoding="utf-8")
    store = RunMemoryStore(p)
    store.load()
    assert store.all_entries() == []


def test_store_save_then_load_roundtrip(tmp_path: Path):
    p = tmp_path / "memory.json"
    store = RunMemoryStore(p)
    entry = _make_entry(sig="abc", when=time.time())
    store.record(entry)
    store.save()

    fresh = RunMemoryStore(p)
    fresh.load()
    assert len(fresh.all_entries()) == 1
    assert fresh.all_entries()[0].signature == "abc"


def test_store_lookup_returns_newest_first(tmp_path: Path):
    p = tmp_path / "memory.json"
    store = RunMemoryStore(p)
    older = _make_entry(sig="abc", when=time.time() - 100, summary="old")
    newer = _make_entry(sig="abc", when=time.time() - 1, summary="new")
    store.record(older)
    store.record(newer)
    matches = store.lookup("abc")
    # Both kept (max_per_signature default = 3); newer first.
    assert [m.summary for m in matches] == ["new", "old"]


def test_store_latest_picks_newest(tmp_path: Path):
    p = tmp_path / "memory.json"
    store = RunMemoryStore(p)
    now = time.time()
    store.record(_make_entry(sig="abc", when=now - 100, summary="first"))
    store.record(_make_entry(sig="abc", when=now - 1, summary="latest"))
    assert store.latest("abc").summary == "latest"


def test_store_latest_returns_none_for_missing(tmp_path: Path):
    p = tmp_path / "memory.json"
    store = RunMemoryStore(p)
    assert store.latest("does-not-exist") is None


def test_store_lookup_empty_signature_returns_empty(tmp_path: Path):
    store = RunMemoryStore(tmp_path / "m.json")
    assert store.lookup("") == []


# -----------------------------------------------------------------------------
# Eviction: per-signature + age
# -----------------------------------------------------------------------------

def test_evict_caps_per_signature(tmp_path: Path):
    p = tmp_path / "memory.json"
    store = RunMemoryStore(p, max_per_signature=2)
    base = time.time() - 100
    # 4 entries for the same signature; only the newest 2 should survive.
    for i in range(4):
        store.record(_make_entry(sig="abc", when=base + i, summary=f"s{i}"))
    matches = store.lookup("abc")
    assert len(matches) == 2
    summaries = [m.summary for m in matches]
    # Newest first; "s3" + "s2" survive.
    assert summaries == ["s3", "s2"]


def test_evict_does_not_cross_signatures(tmp_path: Path):
    p = tmp_path / "memory.json"
    store = RunMemoryStore(p, max_per_signature=1)
    now = time.time()
    store.record(_make_entry(sig="abc", when=now - 10))
    store.record(_make_entry(sig="def", when=now - 5))
    # Both kept — different signatures.
    assert len(store.all_entries()) == 2


def test_evict_drops_old_entries_on_load(tmp_path: Path):
    p = tmp_path / "memory.json"
    # Manually craft a file with one expired and one fresh entry.
    fresh_t = time.time() - 60
    expired_t = time.time() - 86400 * 100  # 100 days old
    payload = {
        "version": 1,
        "entries": [
            _make_entry(sig="fresh", when=fresh_t).to_dict(),
            _make_entry(sig="old", when=expired_t).to_dict(),
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    store = RunMemoryStore(p, max_age_days=30)
    store.load()
    assert [e.signature for e in store.all_entries()] == ["fresh"]


def test_evict_drops_old_entries_on_record(tmp_path: Path):
    p = tmp_path / "memory.json"
    store = RunMemoryStore(p, max_age_days=30)
    store.record(_make_entry(sig="old", when=time.time() - 86400 * 100))
    store.record(_make_entry(sig="new", when=time.time()))
    sigs = {e.signature for e in store.all_entries()}
    assert sigs == {"new"}


# -----------------------------------------------------------------------------
# Mutation: clear / record-empty-signature
# -----------------------------------------------------------------------------

def test_clear_drops_everything(tmp_path: Path):
    store = RunMemoryStore(tmp_path / "m.json")
    store.record(_make_entry(sig="abc", when=time.time()))
    store.clear()
    assert store.all_entries() == []


def test_record_with_empty_signature_is_dropped(tmp_path: Path):
    store = RunMemoryStore(tmp_path / "m.json")
    store.record(_make_entry(sig="", when=time.time()))
    assert store.all_entries() == []


# -----------------------------------------------------------------------------
# format_prior_run_hint
# -----------------------------------------------------------------------------

def test_format_prior_run_hint_includes_summary_and_age():
    e = _make_entry(sig="abc", when=time.time() - 3600, summary="my hint")
    text = format_prior_run_hint(e)
    assert "my hint" in text
    assert "ADAPT" in text or "adapt" in text


def test_format_prior_run_hint_empty_summary_returns_empty():
    e = _make_entry(sig="abc", when=time.time(), summary="")
    assert format_prior_run_hint(e) == ""


# -----------------------------------------------------------------------------
# summarize_run_actions — fake VLM client
# -----------------------------------------------------------------------------

class _FakeVLM:
    def __init__(self, response: str | Exception):
        self.response = response
        self.calls: list[str] = []

    def summarize_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_summarize_uses_vlm_when_available():
    vlm = _FakeVLM(response="to open youtube, click and press enter.")
    out = summarize_run_actions(
        client=vlm,
        tasks=["open youtube"],
        actions=["BROWSER_GO [https://youtube.com]", "PRESS [enter]"],
    )
    assert out == "to open youtube, click and press enter."
    # Prompt should include both the tasks and actions.
    assert "open youtube" in vlm.calls[0]
    assert "BROWSER_GO" in vlm.calls[0]


def test_summarize_strips_quotes_and_whitespace():
    vlm = _FakeVLM(response='  "to open youtube, etc"  ')
    out = summarize_run_actions(
        client=vlm,
        tasks=["open youtube"],
        actions=["X"],
    )
    assert out == "to open youtube, etc"


def test_summarize_falls_back_on_vlm_error():
    vlm = _FakeVLM(response=RuntimeError("boom"))
    out = summarize_run_actions(
        client=vlm,
        tasks=["open youtube"],
        actions=["BROWSER_GO [u]", "PRESS [enter]"],
    )
    # Falls back to the deterministic summary.
    assert out.startswith("Previously this task")
    assert "BROWSER_GO" in out


def test_summarize_falls_back_on_empty_response():
    vlm = _FakeVLM(response="   ")
    out = summarize_run_actions(
        client=vlm,
        tasks=["t"],
        actions=["TYPE [x]"],
    )
    assert out.startswith("Previously this task")


def test_summarize_returns_deterministic_when_no_client():
    out = summarize_run_actions(
        client=None,
        tasks=["t"],
        actions=["PRESS [enter]"],
    )
    assert out.startswith("Previously this task")


def test_summarize_no_actions_no_vlm_call():
    vlm = _FakeVLM(response="should-not-be-used")
    out = summarize_run_actions(
        client=vlm,
        tasks=["t"],
        actions=[],
    )
    assert out == ""
    assert vlm.calls == []


def test_summarize_truncates_long_action_log():
    # 50 actions; summariser caps at 30 in the prompt.
    actions = [f"PRESS [k{i}]" for i in range(50)]
    vlm = _FakeVLM(response="ok")
    summarize_run_actions(
        client=vlm,
        tasks=["t"],
        actions=actions,
    )
    assert vlm.calls, "VLM should have been called"
    # The earliest 20 must NOT appear in the prompt.
    prompt = vlm.calls[0]
    assert "PRESS [k0]" not in prompt
    assert "PRESS [k49]" in prompt


# -----------------------------------------------------------------------------
# Atomic save: writes via .tmp rename so a crash mid-write doesn't corrupt.
# -----------------------------------------------------------------------------

def test_save_uses_tmp_rename(tmp_path: Path):
    p = tmp_path / "memory.json"
    store = RunMemoryStore(p)
    store.record(_make_entry(sig="abc", when=time.time()))
    store.save()
    assert p.exists()
    # No leftover .tmp file after a clean save.
    assert not (p.with_suffix(p.suffix + ".tmp")).exists()
