"""Screenshot capture and 0-1000 coordinate scaling.

The VLM emits coordinates on a normalized 0-1000 grid. This module captures
the primary monitor and converts normalized coordinates to native pixels,
regardless of the display's actual resolution or DPI scaling.

It also provides crop helpers used by the two-stage CLICK refinement path:
a coarse CLICK is turned into a small crop centered on the target, which is
sent back to the VLM for a more precise pick.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# The VLM is prompted to emit coordinates on this normalized grid.
NORMALIZED_SCALE = 1000


@dataclass(frozen=True)
class ScreenGeometry:
    width: int
    height: int

    @property
    def scale_x(self) -> float:
        return self.width / NORMALIZED_SCALE

    @property
    def scale_y(self) -> float:
        return self.height / NORMALIZED_SCALE

    def to_pixels(self, nx: int, ny: int) -> tuple[int, int]:
        """Convert 0-1000 normalized coords to native pixel coords."""
        nx = max(0, min(NORMALIZED_SCALE, int(nx)))
        ny = max(0, min(NORMALIZED_SCALE, int(ny)))
        px = int(round(nx * self.scale_x))
        py = int(round(ny * self.scale_y))
        # Clamp to valid pixel range.
        px = max(0, min(self.width - 1, px))
        py = max(0, min(self.height - 1, py))
        return px, py


def detect_geometry() -> ScreenGeometry:
    """Detect the primary monitor's native resolution via pyautogui."""
    import pyautogui  # lazy: requires a display server

    size = pyautogui.size()
    geom = ScreenGeometry(width=int(size.width), height=int(size.height))
    log.info("Detected screen geometry: %dx%d", geom.width, geom.height)
    return geom


def capture_screenshot() -> Image.Image:
    """Capture the primary monitor as a PIL image."""
    import pyautogui  # lazy: requires a display server

    return pyautogui.screenshot()


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@dataclass(frozen=True)
class CropResult:
    """A cropped region of the full screenshot.

    `origin_px` is the pixel coordinate of the crop's top-left corner on the
    full screen. `size_px` is the actual `(width, height)` of the crop (it
    can be smaller than the requested size if the coarse point was near an
    edge).
    """

    image: Image.Image
    origin_px: tuple[int, int]
    size_px: tuple[int, int]

    def crop_norm_to_full_pixel(self, cx_norm: int, cy_norm: int) -> tuple[int, int]:
        """Map `(cx, cy)` in the crop's own 0-1000 grid to full-screen pixels."""
        cx_norm = max(0, min(NORMALIZED_SCALE, int(cx_norm)))
        cy_norm = max(0, min(NORMALIZED_SCALE, int(cy_norm)))
        cw, ch = self.size_px
        rel_x = int(round(cx_norm * cw / NORMALIZED_SCALE))
        rel_y = int(round(cy_norm * ch / NORMALIZED_SCALE))
        return self.origin_px[0] + rel_x, self.origin_px[1] + rel_y


def crop_around(
    screenshot: Image.Image,
    geometry: ScreenGeometry,
    norm_x: int,
    norm_y: int,
    crop_size_px: int,
) -> CropResult:
    """Crop a square region of up to `crop_size_px` centered on `(norm_x, norm_y)`.

    The coarse point is on the full-screen 0-1000 grid. The returned crop is
    clamped to the screen bounds and shifted inward when the point is near
    an edge, so the crop is always as close to `crop_size_px x crop_size_px`
    as the screen allows.
    """
    if crop_size_px <= 0:
        raise ValueError("crop_size_px must be positive")

    cx_px, cy_px = geometry.to_pixels(norm_x, norm_y)
    half = crop_size_px // 2
    w, h = geometry.width, geometry.height

    # Initial window centered on the coarse point.
    left = cx_px - half
    top = cy_px - half
    right = left + crop_size_px
    bottom = top + crop_size_px

    # Shift into bounds: clamp the far edge, then re-derive the near edge so the
    # crop stays as close to `crop_size_px` wide/tall as possible.
    if right > w:
        right = w
    if bottom > h:
        bottom = h
    left = max(0, right - crop_size_px)
    top = max(0, bottom - crop_size_px)
    # Recompute right/bottom in case left/top had to move to 0.
    right = min(w, left + crop_size_px)
    bottom = min(h, top + crop_size_px)

    cropped = screenshot.crop((left, top, right, bottom))
    return CropResult(
        image=cropped,
        origin_px=(left, top),
        size_px=(right - left, bottom - top),
    )


def annotate_candidates(
    screenshot: Image.Image,
    crop: CropResult,
    candidates: list[tuple[int, int]],
    box_size_px: int = 40,
) -> Image.Image:
    """Draw numbered red rectangles on a copy of `screenshot` around each candidate.

    Each candidate is `(cx, cy)` on the crop's own 0-1000 grid. The output
    image has 1-based labels `1, 2, ...` rendered next to each rectangle so
    the disambiguator VLM can reference them by number.
    """
    from PIL import ImageDraw, ImageFont

    annotated = screenshot.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
        )
    except OSError:
        font = ImageFont.load_default()

    half = box_size_px // 2
    for idx, (cx_norm, cy_norm) in enumerate(candidates, start=1):
        px, py = crop.crop_norm_to_full_pixel(cx_norm, cy_norm)
        left, top = px - half, py - half
        right, bottom = px + half, py + half
        # Red rectangle outline.
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=4)
        # White-filled badge with the 1-based number.
        label = str(idx)
        label_box = (left, max(0, top - 34), left + 32, max(34, top))
        draw.rectangle(label_box, fill=(255, 255, 255), outline=(255, 0, 0), width=2)
        draw.text(
            (label_box[0] + 6, label_box[1] + 2),
            label,
            fill=(255, 0, 0),
            font=font,
        )

    return annotated
