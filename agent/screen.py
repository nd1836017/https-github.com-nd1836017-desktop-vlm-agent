"""Screenshot capture and 0-1000 coordinate scaling.

The VLM emits coordinates on a normalized 0-1000 grid. This module captures
the primary monitor and converts normalized coordinates to native pixels,
regardless of the display's actual resolution or DPI scaling.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

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
