"""Regression tests for 10 medium-severity Devin Review findings against PR #16's
groundwork (REMEMBER/RECALL + Tier 4) — fixed in this PR.

Each block at the top of a test names the bug number from the review and the
specific failure it guards against. Bug 11 was raised but withdrawn by the
user before fixing; that gap in numbering is intentional.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import agent.agent as agent_module
from agent.artifacts import ArtifactWriter
from agent.executor import execute, execute_click_pixels
from agent.files import (
    _filename_from_url,
    format_features_summary,
    inspect_features,
)
from agent.history import History
from agent.parser import ScrollCommand, TypeCommand
from agent.screen import CropResult, ScreenGeometry, annotate_candidates
from agent.state import AgentState, save_state
from agent.tasks_loader import TaskStep
from agent.vlm import PlanResponseModel, plan_response_to_command


# ---------------------------------------------------------------------------
# Bug #5: AgentHalt was defined but never raised — dead code.
# Removing it should not surface as an export the codebase relied on.
# ---------------------------------------------------------------------------
def test_bug5_agent_halt_no_longer_exported():
    """The unused AgentHalt class should no longer exist on agent.agent."""
    assert not hasattr(agent_module, "AgentHalt"), (
        "AgentHalt was dead code — its presence was misleading. "
        "If you're re-adding it, make sure something actually raises it."
    )


# ---------------------------------------------------------------------------
# Bug #6: execute_click_pixels post-click sleep is INTENTIONAL (matches the
# regular CLICK branch in execute()). The reviewer was concerned about
# "unnecessary latency"; the docstring now documents why this is correct
# and not a bug. Behavior unchanged — this test pins the behavior.
# ---------------------------------------------------------------------------
def test_bug6_execute_click_pixels_still_sleeps_animation_buffer():
    sleeps: list[float] = []
    fake_pyautogui = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake_pyautogui),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
        patch("agent.executor.random.uniform", return_value=0.0),
    ):
        execute_click_pixels(100, 200, animation_buffer_seconds=1.5)
    # The animation buffer sleep should still fire — clicks animate.
    assert 1.5 in sleeps, sleeps


def test_bug6_execute_click_pixels_can_skip_buffer_when_caller_opts_out():
    sleeps: list[float] = []
    fake_pyautogui = MagicMock()
    with (
        patch("agent.executor._pyautogui", return_value=fake_pyautogui),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
        patch("agent.executor.random.uniform", return_value=0.0),
    ):
        execute_click_pixels(100, 200, animation_buffer_seconds=0.0)
    # 0.0 sleeps are still recorded by time.sleep but should not waste time.
    assert all(s == 0.0 for s in sleeps), sleeps


# ---------------------------------------------------------------------------
# Bug #7: annotate_candidates badge collapsed near top of screen.
# When a candidate's top edge was within 34 px of the image top, the badge
# rectangle was drawn OVER the click target. The fix flips the badge to
# below the candidate box when there isn't enough room above.
# ---------------------------------------------------------------------------
def _make_crop_region(full_w: int = 200, full_h: int = 200) -> CropResult:
    """Crop covering the full image so candidate norm coords map 1:1."""
    return CropResult(
        image=Image.new("RGB", (full_w, full_h)),
        origin_px=(0, 0),
        size_px=(full_w, full_h),
    )


def _is_red(rgb: tuple[int, int, int]) -> bool:
    return rgb[0] > 200 and rgb[1] < 100 and rgb[2] < 100


def test_bug7_badge_does_not_overlap_candidate_at_top_of_screen():
    """Candidate at the very top — badge must end up below the box, not on it."""
    # Use a 1000x1000 image so the 0..1000 normalized grid maps 1:1.
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    crop = _make_crop_region(1000, 1000)
    # px=500, py=0. Box at y in [-20, 20].
    annotated = annotate_candidates(img, crop, [(500, 0)], box_size_px=40)
    pixels = annotated.load()
    # Under the fix, badge top is at y=20 (bottom of box), badge_h=34.
    # Look for red badge pixels in y in [20, 54], x in [480, 512].
    has_red_below_box = any(
        _is_red(pixels[x, y])
        for y in range(20, 55)
        for x in range(480, 512)
    )
    assert has_red_below_box, "Badge must be drawn below the candidate box"


def test_bug7_badge_above_candidate_when_room_above():
    """Candidate well below top — badge stays above (no behavior change)."""
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    crop = _make_crop_region(1000, 1000)
    # px=500, py=500. Box at y in [480, 520]. Badge above at y in [446, 480].
    annotated = annotate_candidates(img, crop, [(500, 500)], box_size_px=40)
    pixels = annotated.load()
    has_red_above_box = any(
        _is_red(pixels[x, y])
        for y in range(446, 480)
        for x in range(480, 512)
    )
    assert has_red_above_box


# ---------------------------------------------------------------------------
# Bug #8: inspect_features used max(rows, row_index) for csv_row_count and
# omitted the actual expanded step count, making the summary line confusing.
# Now we track csv_step_count separately and the summary shows real numbers.
# ---------------------------------------------------------------------------
def _step(text: str, *, row_index: int | None = None, csv_name: str | None = None):
    return TaskStep(text=text, row_index=row_index, csv_name=csv_name)


def test_bug8_csv_step_count_reports_expanded_total():
    # 10 rows x 3 inner steps = 30 expanded steps.
    steps = []
    for row in range(1, 11):
        for inner in ("step a", "step b", "step c"):
            steps.append(_step(f"{inner} row {row}", row_index=row, csv_name="data.csv"))
    feats = inspect_features(steps)
    assert feats.csv_row_count == 10
    assert feats.csv_step_count == 30
    summary = format_features_summary(feats, total_steps=30)
    # Real numbers in the user-facing message.
    assert "10 rows" in summary
    assert "3 inner" in summary
    assert "30 expanded" in summary
    assert "from data.csv" in summary


def test_bug8_csv_step_count_zero_when_no_loop():
    feats = inspect_features([_step("plain step")])
    assert feats.csv_row_count == 0
    assert feats.csv_step_count == 0
    assert not feats.uses_csv_loop


# ---------------------------------------------------------------------------
# Bug #9: _filename_from_url used ``"." not in base`` which mishandled
# names like "invoice-v2" (real extensionless filename with no real
# extension). The fix uses Path.suffix to detect a real extension.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com/download", "download.bin"),  # no dot -> .bin
        ("https://example.com/file.pdf", "file.pdf"),  # real ext -> as-is
        ("https://example.com/invoice-v2", "invoice-v2.bin"),  # no real ext -> .bin
        ("https://example.com/run.final", "run.final"),  # ".final" treated as ext
        ("https://example.com/", "download.bin"),  # empty path
    ],
)
def test_bug9_filename_from_url(url: str, expected: str):
    assert _filename_from_url(url) == expected


# ---------------------------------------------------------------------------
# Bug #10: History(window=0) silently dropped records via deque(maxlen=0).
# Now the docstring documents window=0 as the "disabled" mode and the
# methods short-circuit explicitly.
# ---------------------------------------------------------------------------
def test_bug10_history_window_zero_disabled_explicit():
    h = History(window=0)
    assert h.disabled is True
    h.record("step", "action", True, "ok")
    assert len(h) == 0
    assert list(iter(h)) == []
    assert h.summary() == ""


def test_bug10_history_window_negative_still_raises():
    with pytest.raises(ValueError, match="non-negative"):
        History(window=-1)


def test_bug10_history_window_positive_unchanged():
    h = History(window=2)
    assert h.disabled is False
    h.record("step1", "a", True, "ok1")
    h.record("step2", "b", True, "ok2")
    h.record("step3", "c", True, "ok3")
    # Drops oldest.
    assert len(h) == 2
    assert "step1" not in h.summary()
    assert "step2" in h.summary()
    assert "step3" in h.summary()


# ---------------------------------------------------------------------------
# Bug #12: SCROLL with a bad direction returned None silently from the
# JSON-mode planner path. Now it logs a warning before returning None so
# malformed plans are visible during debugging.
# ---------------------------------------------------------------------------
def test_bug12_scroll_invalid_direction_logs_warning(caplog):
    resp = PlanResponseModel(command="SCROLL", direction="sideways", amount=3)
    with caplog.at_level(logging.WARNING, logger="agent.vlm"):
        result = plan_response_to_command(resp)
    assert result is None
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("invalid direction" in r.getMessage() for r in warnings), [
        r.getMessage() for r in warnings
    ]


def test_bug12_scroll_valid_direction_still_works():
    resp = PlanResponseModel(command="SCROLL", direction="DOWN", amount=5)
    result = plan_response_to_command(resp)
    assert isinstance(result, ScrollCommand)
    assert result.direction == "down"
    assert result.amount == 5


# ---------------------------------------------------------------------------
# Bug #13: TYPE post-buffer was hardcoded 0.25s. For long pastes the page
# may still be running input handlers when the next screenshot fires.
# Buffer now scales with text length, capped at 2.0s.
# ---------------------------------------------------------------------------
def _run_type(text: str) -> list[float]:
    sleeps: list[float] = []
    fake_pyautogui = MagicMock()
    geometry = ScreenGeometry(width=1920, height=1080)
    cmd = TypeCommand(text=text)
    with (
        patch("agent.executor._pyautogui", return_value=fake_pyautogui),
        patch("agent.executor.time.sleep", side_effect=sleeps.append),
        patch("agent.executor.random.uniform", return_value=0.0),
    ):
        execute(cmd, geometry)
    return sleeps


def test_bug13_type_post_buffer_short_string():
    sleeps = _run_type("hi")
    # Last sleep is the post-type settle — should be the floor (0.25s).
    assert sleeps[-1] == pytest.approx(0.25, abs=1e-6)


def test_bug13_type_post_buffer_scales_with_length():
    # 200 chars * 0.005 = 1.0s, between 0.25 floor and 2.0 cap.
    sleeps = _run_type("x" * 200)
    assert sleeps[-1] == pytest.approx(1.0, abs=1e-6)


def test_bug13_type_post_buffer_capped_at_two_seconds():
    # 1000 chars * 0.005 = 5.0s, capped at 2.0s.
    sleeps = _run_type("x" * 1000)
    assert sleeps[-1] == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Bug #14: save_state used a bare `raise` in the except block, so the
# operator never saw the path that was being written. The fix wraps the
# original exception with a descriptive OSError that names the path.
# ---------------------------------------------------------------------------
def test_bug14_save_state_reraises_with_path_context(tmp_path):
    state = AgentState(
        version=2,
        tasks_file="tasks.txt",
        total_steps=1,
        last_completed_step=0,
        variables={},
    )
    bad_path = tmp_path / "ok.json"

    # Force os.replace to fail. Patch the symbol used inside the module.
    with (
        patch("agent.state.os.replace", side_effect=OSError("disk full")),
        pytest.raises(OSError) as excinfo,
    ):
        save_state(bad_path, state)

    msg = str(excinfo.value)
    assert "Failed to write checkpoint" in msg
    # The original error should appear chained.
    assert "disk full" in msg or excinfo.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Bug #15: ArtifactWriter timestamps used datetime.now() (local time)
# which can collide or sort backwards across DST. The fix uses UTC and
# suffixes the folder name with 'Z' to make the time zone explicit.
# ---------------------------------------------------------------------------
def test_bug15_artifact_writer_timestamp_is_utc(tmp_path):
    writer = ArtifactWriter.create(enabled=True, base_dir=str(tmp_path))
    assert writer.run_dir is not None
    name = writer.run_dir.name
    # YYYYMMDD-HHMMSSZ — 16 chars, ends with Z.
    assert name.endswith("Z"), f"Expected UTC suffix, got {name!r}"
    assert len(name) == 16, f"Unexpected timestamp shape: {name!r}"
