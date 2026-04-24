"""Tests for coordinate scaling."""
from __future__ import annotations

from agent.screen import ScreenGeometry


def test_scale_corners_1080p():
    g = ScreenGeometry(width=1920, height=1080)
    assert g.to_pixels(0, 0) == (0, 0)
    assert g.to_pixels(1000, 1000) == (1919, 1079)


def test_scale_center_1080p():
    g = ScreenGeometry(width=1920, height=1080)
    px, py = g.to_pixels(500, 500)
    # Allow small rounding tolerance.
    assert abs(px - 960) <= 1
    assert abs(py - 540) <= 1


def test_scale_4k():
    g = ScreenGeometry(width=3840, height=2160)
    assert g.to_pixels(0, 0) == (0, 0)
    assert g.to_pixels(1000, 1000) == (3839, 2159)
    px, py = g.to_pixels(250, 750)
    assert abs(px - 960) <= 1
    assert abs(py - 1620) <= 1


def test_scale_clamps_out_of_range():
    g = ScreenGeometry(width=1920, height=1080)
    assert g.to_pixels(-10, -10) == (0, 0)
    assert g.to_pixels(9999, 9999) == (1919, 1079)
