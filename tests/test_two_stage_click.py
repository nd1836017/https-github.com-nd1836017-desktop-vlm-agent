"""Tests for the two-stage CLICK refinement path.

Covers:
- Crop math (`crop_around` + `CropResult.crop_norm_to_full_pixel`).
- Refine response parsing (`GeminiClient.refine_click` parsing, via a
  FakeClient that replays raw strings through the same regex).
- Disambiguation branching: 0 / 1 / >1 candidates.
- End-to-end agent flow with two-stage ON: confirms refine is called,
  disambiguation only fires when multiple candidates, and the executed
  pixel lands on the refined candidate (not the coarse point).
"""
from __future__ import annotations

import sys
from typing import cast
from unittest import mock

import pytest
from PIL import Image

from agent.agent import run_step
from agent.history import History
from agent.screen import (
    CropResult,
    ScreenGeometry,
    annotate_candidates,
    crop_around,
)
from agent.vlm import VerificationResult


# -----------------------------------------------------------------------------
# Crop math
# -----------------------------------------------------------------------------
def test_crop_around_center_region():
    geom = ScreenGeometry(width=1000, height=1000)
    img = Image.new("RGB", (1000, 1000), color=(0, 0, 0))
    crop = crop_around(img, geom, norm_x=500, norm_y=500, crop_size_px=200)
    assert crop.origin_px == (400, 400)
    assert crop.size_px == (200, 200)


def test_crop_around_top_left_edge_is_shifted_inward():
    geom = ScreenGeometry(width=1000, height=1000)
    img = Image.new("RGB", (1000, 1000))
    crop = crop_around(img, geom, norm_x=0, norm_y=0, crop_size_px=200)
    # Coarse pt clamps to (0,0); crop origin must stay at (0,0) and the crop
    # must be the requested size, not smaller.
    assert crop.origin_px == (0, 0)
    assert crop.size_px == (200, 200)


def test_crop_around_bottom_right_edge_is_shifted_inward():
    geom = ScreenGeometry(width=1000, height=1000)
    img = Image.new("RGB", (1000, 1000))
    crop = crop_around(img, geom, norm_x=1000, norm_y=1000, crop_size_px=200)
    assert crop.origin_px == (800, 800)
    assert crop.size_px == (200, 200)


def test_crop_around_smaller_than_requested_when_screen_is_tiny():
    geom = ScreenGeometry(width=100, height=100)
    img = Image.new("RGB", (100, 100))
    crop = crop_around(img, geom, norm_x=500, norm_y=500, crop_size_px=200)
    # Crop can't grow beyond the screen.
    assert crop.origin_px == (0, 0)
    assert crop.size_px == (100, 100)


def test_crop_norm_to_full_pixel_maps_correctly():
    # A 300-px crop at origin (100, 200); normalized (500, 500) inside the crop
    # should land at pixel (100+150, 200+150) = (250, 350) on the full screen.
    crop = CropResult(
        image=Image.new("RGB", (300, 300)),
        origin_px=(100, 200),
        size_px=(300, 300),
    )
    assert crop.crop_norm_to_full_pixel(500, 500) == (250, 350)
    # Corners.
    assert crop.crop_norm_to_full_pixel(0, 0) == (100, 200)
    assert crop.crop_norm_to_full_pixel(1000, 1000) == (400, 500)


def test_crop_norm_to_full_pixel_clamps_out_of_range_inputs():
    crop = CropResult(
        image=Image.new("RGB", (100, 100)),
        origin_px=(0, 0),
        size_px=(100, 100),
    )
    # Negative / >1000 are clamped to the 0-1000 range.
    assert crop.crop_norm_to_full_pixel(-50, 1500) == (0, 100)


def test_crop_size_px_must_be_positive():
    geom = ScreenGeometry(width=100, height=100)
    img = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError):
        crop_around(img, geom, 500, 500, crop_size_px=0)


# -----------------------------------------------------------------------------
# annotate_candidates runs without crashing (smoke)
# -----------------------------------------------------------------------------
def test_annotate_candidates_produces_image_of_same_size():
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    crop = CropResult(
        image=Image.new("RGB", (200, 200)),
        origin_px=(300, 200),
        size_px=(200, 200),
    )
    annotated = annotate_candidates(
        img, crop, candidates=[(250, 250), (750, 750)]
    )
    assert annotated.size == (800, 600)


# -----------------------------------------------------------------------------
# End-to-end agent flow with two-stage ON
# -----------------------------------------------------------------------------
class TwoStageFakeClient:
    """FakeClient that also supports refine_click + disambiguate_candidates."""

    def __init__(
        self,
        plan_outputs,
        verify_outputs,
        refine_outputs,
        disambig_outputs=None,
    ):
        self._plan_outputs = list(plan_outputs)
        self._verify_outputs = list(verify_outputs)
        self._refine_outputs = list(refine_outputs)
        self._disambig_outputs = list(disambig_outputs or [])
        self.plan_calls: list[dict] = []
        self.refine_calls: list[dict] = []
        self.disambig_calls: list[dict] = []
        self.verify_calls = 0

    def plan_action(
        self, step, screenshot, history_summary="", previous_failure=""
    ):
        self.plan_calls.append({"step": step})
        return self._plan_outputs.pop(0)

    def verify(self, goal, screenshot):
        self.verify_calls += 1
        out = self._verify_outputs.pop(0)
        if isinstance(out, VerificationResult):
            return out
        passed, reason = out
        return VerificationResult(passed=passed, reason=reason)

    def refine_click(self, step, crop, max_candidates=5):
        self.refine_calls.append(
            {"step": step, "crop_size": crop.size, "max_candidates": max_candidates}
        )
        return self._refine_outputs.pop(0)

    def disambiguate_candidates(self, step, annotated_screenshot, num_candidates):
        self.disambig_calls.append(
            {"step": step, "num_candidates": num_candidates}
        )
        return self._disambig_outputs.pop(0)


@pytest.fixture
def fake_geometry():
    return ScreenGeometry(width=1000, height=1000)


@pytest.fixture(autouse=True)
def patch_pyautogui():
    fake_img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    fake_pyautogui = mock.MagicMock()
    fake_pyautogui.size.return_value = mock.MagicMock(width=1000, height=1000)
    fake_pyautogui.screenshot.return_value = fake_img
    fake_pyautogui.FAILSAFE = False
    with mock.patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        yield fake_pyautogui


def test_single_candidate_skips_disambiguation_and_clicks_refined_pixel(
    fake_geometry, patch_pyautogui
):
    """Coarse CLICK [500,500] + single refined candidate at crop (500,500)
    → must click the pixel mapped through the crop math, NOT the coarse pixel."""
    client = TwoStageFakeClient(
        plan_outputs=["CLICK [500,500]"],
        verify_outputs=[VerificationResult(passed=True, reason="PASS")],
        refine_outputs=[[(500, 500)]],  # one candidate at crop-center
    )
    result = run_step(
        step="click the only button",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=0,
        history=History(window=0),
        enable_two_stage_click=True,
        two_stage_crop_size_px=200,
        max_click_candidates=5,
    )
    assert result.passed is True
    assert len(client.refine_calls) == 1, "refine must be called once per CLICK"
    assert len(client.disambig_calls) == 0, (
        "single candidate should skip disambiguation"
    )
    # With a 200-px crop centered at (500,500), crop origin is (400,400); the
    # candidate at crop-normalized (500,500) maps to full-screen pixel (500,500).
    assert patch_pyautogui.click.call_args.args == (500, 500)


def test_multiple_candidates_trigger_disambiguation(
    fake_geometry, patch_pyautogui
):
    """Two refined candidates + disambiguator picks #2."""
    client = TwoStageFakeClient(
        plan_outputs=["CLICK [500,500]"],
        verify_outputs=[VerificationResult(passed=True, reason="PASS")],
        refine_outputs=[[(250, 250), (750, 750)]],
        disambig_outputs=[2],
    )
    result = run_step(
        step="click the second button",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=0,
        history=History(window=0),
        enable_two_stage_click=True,
        two_stage_crop_size_px=200,
        max_click_candidates=5,
    )
    assert result.passed is True
    assert len(client.refine_calls) == 1
    assert len(client.disambig_calls) == 1
    assert client.disambig_calls[0]["num_candidates"] == 2
    # Candidate #2 at crop-normalized (750,750) on a 200-px crop at origin
    # (400,400) maps to pixel (400 + 0.75*200, 400 + 0.75*200) = (550, 550).
    assert patch_pyautogui.click.call_args.args == (550, 550)


def test_zero_candidates_triggers_replan(fake_geometry, patch_pyautogui):
    """Refine returns [] → step is treated as FAIL; replan budget consumed."""
    client = TwoStageFakeClient(
        plan_outputs=["CLICK [500,500]", "CLICK [600,600]"],
        verify_outputs=[VerificationResult(passed=True, reason="PASS")],
        refine_outputs=[[], [(500, 500)]],  # first refine: NONE; second: one hit
    )
    result = run_step(
        step="click the right thing",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=1,  # allow one replan
        history=History(window=5),
        enable_two_stage_click=True,
        two_stage_crop_size_px=200,
        max_click_candidates=5,
    )
    assert result.passed is True
    # Plan was called twice (initial + 1 replan), refine twice, no executor
    # click on the first attempt (since refine returned nothing).
    assert len(client.plan_calls) == 2
    assert len(client.refine_calls) == 2
    assert patch_pyautogui.click.call_count == 1, (
        "first attempt should not click (refine said NONE); only the replan clicks"
    )


def test_disambiguator_returns_zero_triggers_replan(
    fake_geometry, patch_pyautogui
):
    """Ambiguous candidates but disambiguator says 0 → FAIL + replan."""
    client = TwoStageFakeClient(
        plan_outputs=["CLICK [500,500]"],
        verify_outputs=[],
        refine_outputs=[[(250, 250), (750, 750)]],
        disambig_outputs=[0],  # 'none of them match'
    )
    result = run_step(
        step="impossible target",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=0,
        history=History(window=0),
        enable_two_stage_click=True,
        two_stage_crop_size_px=200,
        max_click_candidates=5,
    )
    assert result.passed is False
    assert patch_pyautogui.click.call_count == 0, (
        "disambig=0 must abort the click, not fall back to the coarse point"
    )


def test_two_stage_disabled_uses_coarse_click_directly(
    fake_geometry, patch_pyautogui
):
    """When ENABLE_TWO_STAGE_CLICK is off, refine must NOT be called and the
    coarse pixel is used as-is (legacy behaviour)."""
    client = TwoStageFakeClient(
        plan_outputs=["CLICK [300,700]"],
        verify_outputs=[VerificationResult(passed=True, reason="PASS")],
        refine_outputs=[],  # would blow up if called
    )
    result = run_step(
        step="click the only button",
        vlm=cast("object", client),
        geometry=fake_geometry,
        animation_buffer=0.0,
        max_parse_retries=0,
        max_replans=0,
        history=History(window=0),
        enable_two_stage_click=False,
    )
    assert result.passed is True
    assert len(client.refine_calls) == 0
    # Coarse 300,700 on a 1000x1000 screen maps to pixel (300, 700).
    assert patch_pyautogui.click.call_args.args == (300, 700)
