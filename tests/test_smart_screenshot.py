"""Tests for PR S Phase 1 smart-screenshot optimizations.

Covers:
- Layer 1: ``image_to_jpeg_bytes`` downsamples + JPEG-encodes correctly.
- Layer 1 wiring: ``GeminiClient._generate`` swaps PIL Images for a
  small JPEG ``Part`` before calling the SDK.
- Layer 2: identical-frame skip drops the screenshot from
  ``plan_action`` only when both the env flag is on AND consecutive
  signatures match. First call always sends, and a non-matching
  signature always sends.
- Config: env-var loading + bounds validation.
"""
from __future__ import annotations

import io
from unittest import mock

import pytest
from PIL import Image

from agent.screen import (
    downsample_for_vlm,
    image_signature,
    image_to_jpeg_bytes,
)
from agent.vlm import GeminiClient, PlanResponseModel

# ----------------------------------------------------------------- Layer 1


def _make_image(w: int, h: int, *, mode: str = "RGB") -> Image.Image:
    img = Image.new(mode, (w, h), color=(73, 109, 137) if mode == "RGB" else 200)
    return img


def test_downsample_skips_when_already_small():
    img = _make_image(800, 600)
    out = downsample_for_vlm(img, max_dim=1280)
    assert out.size == (800, 600)


def test_downsample_resizes_long_edge_to_max_dim():
    img = _make_image(1920, 1080)
    out = downsample_for_vlm(img, max_dim=1280)
    # Long edge (width) should be exactly 1280, aspect preserved.
    assert out.width == 1280
    # Tolerate 1px rounding on the short edge.
    assert abs(out.height - 720) <= 1


def test_downsample_handles_portrait_orientation():
    img = _make_image(1080, 1920)
    out = downsample_for_vlm(img, max_dim=1280)
    assert out.height == 1280
    assert abs(out.width - 720) <= 1


def test_downsample_disabled_when_max_dim_zero():
    img = _make_image(4000, 4000)
    out = downsample_for_vlm(img, max_dim=0)
    assert out.size == (4000, 4000)


def test_downsample_strips_alpha_to_rgb():
    img = _make_image(200, 200, mode="RGBA")
    out = downsample_for_vlm(img, max_dim=1280)
    # JPEG encoding can't accept RGBA — we must convert.
    assert out.mode == "RGB"


def test_image_to_jpeg_bytes_starts_with_jpeg_magic():
    img = _make_image(1920, 1080)
    data = image_to_jpeg_bytes(img, quality=80, max_dim=1280)
    # JPEG SOI marker.
    assert data[:2] == b"\xff\xd8"


def test_image_to_jpeg_bytes_is_smaller_than_png_for_typical_screenshot():
    """Sanity check the real win: q=80 + 1280px is much smaller than a
    full-resolution PNG of the same content. We synthesize a noisy
    image (deterministic) because solid-color images compress
    extremely well in PNG and don't reflect real screenshot bytes.
    """
    import random

    rng = random.Random(0xC0FFEE)
    img = Image.new("RGB", (1920, 1080))
    pixels = img.load()
    # Sparse noise + structure: mostly a base color with noisy lines
    # every 8th row, mimicking text/UI density without being random
    # enough to defeat JPEG entirely.
    for y in range(img.height):
        if y % 8 == 0:
            for x in range(img.width):
                pixels[x, y] = (
                    rng.randint(0, 255),
                    rng.randint(0, 255),
                    rng.randint(0, 255),
                )
        else:
            for x in range(img.width):
                pixels[x, y] = (200, 210, 220)

    jpeg = image_to_jpeg_bytes(img, quality=80, max_dim=1280)

    png_buf = io.BytesIO()
    img.save(png_buf, format="PNG")
    png = png_buf.getvalue()

    # JPEG @ q=80 of a 1280px-downsampled image should be at least
    # 2x smaller than the full-res PNG of the same content. Real
    # desktop screenshots compress 5-30x; this synthetic image has a
    # row of pure noise every 8 lines (worst case for JPEG), so we
    # only require 2x as a robust floor.
    assert len(jpeg) * 2 < len(png), (
        f"expected at least 2x reduction, got "
        f"{len(jpeg)} JPEG vs {len(png)} PNG"
    )


def test_image_to_jpeg_bytes_is_decodable():
    img = _make_image(1920, 1080)
    data = image_to_jpeg_bytes(img, quality=80, max_dim=1280)
    decoded = Image.open(io.BytesIO(data))
    decoded.load()
    assert decoded.format == "JPEG"
    assert decoded.size == (1280, 720)


# ----------------------------------------------------- Layer 1 client wiring


@pytest.fixture
def stub_client():
    """A GeminiClient backed by a fake SDK that records the call."""
    with mock.patch("agent.vlm.genai.Client"):
        client = GeminiClient(
            api_key="fake",
            model_name="fake-model",
            enable_json_output=True,
            image_max_dim=1280,
            image_quality=80,
            skip_identical_frames=False,
        )
    resp = mock.MagicMock()
    resp.text = '{"command":"WAIT","seconds":1}'
    resp.parsed = PlanResponseModel(command="WAIT", seconds=1)
    client._client.models.generate_content = mock.MagicMock(return_value=resp)
    return client


def test_generate_replaces_pil_image_with_jpeg_part(stub_client):
    img = _make_image(1920, 1080)
    stub_client.plan_action("dummy step", img)

    call_kwargs = stub_client._client.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]

    # No PIL Images in the outgoing payload.
    assert not any(isinstance(c, Image.Image) for c in contents)

    # Exactly one Part with JPEG bytes — the screenshot.
    image_parts = [c for c in contents if hasattr(c, "inline_data")]
    assert len(image_parts) == 1
    blob = image_parts[0].inline_data
    assert blob.mime_type == "image/jpeg"
    assert blob.data[:2] == b"\xff\xd8"


def test_generate_passes_strings_through_unchanged(stub_client):
    img = _make_image(800, 600)
    stub_client.plan_action("click the button", img)

    call_kwargs = stub_client._client.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]
    # First entry is the string prompt, untouched.
    assert isinstance(contents[0], str)
    assert "click the button" in contents[0]


# ------------------------------------------------------- Layer 2 (identical-frame skip)


def _stub_plan_response(client: GeminiClient) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.text = '{"command":"WAIT","seconds":1}'
    resp.parsed = PlanResponseModel(command="WAIT", seconds=1)
    client._client.models.generate_content = mock.MagicMock(return_value=resp)
    return client._client.models.generate_content


def _count_image_parts(contents) -> int:
    return sum(1 for c in contents if hasattr(c, "inline_data"))


def test_first_plan_call_always_sends_screenshot_even_when_skip_enabled():
    with mock.patch("agent.vlm.genai.Client"):
        client = GeminiClient(
            api_key="fake",
            model_name="fake-model",
            enable_json_output=True,
            skip_identical_frames=True,
        )
    sdk = _stub_plan_response(client)

    img = _make_image(1280, 720)
    client.plan_action("step 1", img)

    contents = sdk.call_args.kwargs["contents"]
    assert _count_image_parts(contents) == 1


def test_identical_consecutive_replan_screenshots_are_dropped_when_flag_on():
    """During a replan on the same step, an identical screen drops the image."""
    with mock.patch("agent.vlm.genai.Client"):
        client = GeminiClient(
            api_key="fake",
            model_name="fake-model",
            enable_json_output=True,
            skip_identical_frames=True,
        )
    sdk = _stub_plan_response(client)

    img = _make_image(1280, 720)
    client.plan_action("press the button", img)
    # Replan on the same step: previous_failure is non-empty.
    client.plan_action(
        "press the button", img, previous_failure="nothing happened"
    )

    second_call_contents = sdk.call_args_list[-1].kwargs["contents"]
    # Screenshot was dropped on the second call.
    assert _count_image_parts(second_call_contents) == 0
    # And the planner was told why.
    prompt = second_call_contents[0]
    assert "IDENTICAL" in prompt


def test_identical_screen_on_fresh_step_is_not_dropped():
    """Across step boundaries — i.e. plan_action without previous_failure —
    even if the screen happens to fingerprint the same as the last call,
    we MUST send the image. Otherwise the planner gets zero visual
    context for the new goal and a bogus "last action didn't change the
    UI" message.
    """
    with mock.patch("agent.vlm.genai.Client"):
        client = GeminiClient(
            api_key="fake",
            model_name="fake-model",
            enable_json_output=True,
            skip_identical_frames=True,
        )
    sdk = _stub_plan_response(client)

    img = _make_image(1280, 720)
    # First step: a TYPE in a small field that doesn't perceptibly
    # change the 16x16 fingerprint.
    client.plan_action("type a name", img)
    # Different step, same screenshot — must NOT skip.
    client.plan_action("press Enter", img)

    second_call_contents = sdk.call_args_list[-1].kwargs["contents"]
    assert _count_image_parts(second_call_contents) == 1
    prompt = second_call_contents[0]
    assert "IDENTICAL" not in prompt


def test_changed_screenshot_is_not_dropped():
    with mock.patch("agent.vlm.genai.Client"):
        client = GeminiClient(
            api_key="fake",
            model_name="fake-model",
            enable_json_output=True,
            skip_identical_frames=True,
        )
    sdk = _stub_plan_response(client)

    img_a = Image.new("RGB", (1280, 720), color=(10, 20, 30))
    img_b = Image.new("RGB", (1280, 720), color=(200, 50, 50))
    # Sanity: signatures actually differ.
    assert image_signature(img_a) != image_signature(img_b)

    client.plan_action("step 1", img_a)
    client.plan_action("step 2", img_b)

    second_call_contents = sdk.call_args_list[-1].kwargs["contents"]
    assert _count_image_parts(second_call_contents) == 1


def test_skip_disabled_by_default_keeps_screenshot_on_repeats():
    with mock.patch("agent.vlm.genai.Client"):
        client = GeminiClient(
            api_key="fake",
            model_name="fake-model",
            enable_json_output=True,
            skip_identical_frames=False,
        )
    sdk = _stub_plan_response(client)

    img = _make_image(1280, 720)
    client.plan_action("step 1", img)
    client.plan_action("step 2", img)

    for call in sdk.call_args_list:
        assert _count_image_parts(call.kwargs["contents"]) == 1


# --------------------------------------------------------------- Config


def test_config_loads_smart_screenshot_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("VLM_IMAGE_MAX_DIM", raising=False)
    monkeypatch.delenv("VLM_IMAGE_QUALITY", raising=False)
    monkeypatch.delenv("VLM_SKIP_IDENTICAL_FRAMES", raising=False)
    monkeypatch.chdir(tmp_path)

    from agent.config import Config

    cfg = Config.load()
    assert cfg.vlm_image_max_dim == 1280
    assert cfg.vlm_image_quality == 80
    assert cfg.vlm_skip_identical_frames is False


def test_config_smart_screenshot_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("VLM_IMAGE_MAX_DIM", "2048")
    monkeypatch.setenv("VLM_IMAGE_QUALITY", "60")
    monkeypatch.setenv("VLM_SKIP_IDENTICAL_FRAMES", "true")
    monkeypatch.chdir(tmp_path)

    from agent.config import Config

    cfg = Config.load()
    assert cfg.vlm_image_max_dim == 2048
    assert cfg.vlm_image_quality == 60
    assert cfg.vlm_skip_identical_frames is True


def test_config_rejects_invalid_quality(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("VLM_IMAGE_QUALITY", "0")
    monkeypatch.chdir(tmp_path)

    from agent.config import Config

    with pytest.raises(ValueError, match="VLM_IMAGE_QUALITY"):
        Config.load()
