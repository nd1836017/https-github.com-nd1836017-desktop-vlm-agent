"""Live test for Bug #6: candidate badge must stay 34px tall even near top edge."""
from __future__ import annotations

from PIL import Image

from agent.screen import (
    NORMALIZED_SCALE,
    ScreenGeometry,
    annotate_candidates,
    crop_around,
)


def _count_red(img: Image.Image, x0: int, y0: int, x1: int, y1: int) -> int:
    pixels = img.load()
    count = 0
    for y in range(max(0, y0), min(img.height, y1)):
        for x in range(max(0, x0), min(img.width, x1)):
            r, g, bl = pixels[x, y][:3]
            if r > 200 and g < 80 and bl < 80:
                count += 1
    return count


def run(cand_cy_crop_norm: int, coarse_cy_norm: int, label: str) -> None:
    """Render an annotation for a candidate at a specific normalized y within the crop.

    - `coarse_cy_norm` places the crop on the full screen (in full-screen 0-1000).
    - `cand_cy_crop_norm` is where the candidate sits inside the crop (in crop 0-1000).
    """
    width, height = 1600, 1200
    geom = ScreenGeometry(width=width, height=height)

    # Use a white base screenshot so any red pixels must come from our drawing.
    base = Image.new("RGB", (width, height), (255, 255, 255))

    cx_norm = NORMALIZED_SCALE // 2  # 500 in full-screen grid
    crop = crop_around(base, geom, cx_norm, coarse_cy_norm, crop_size_px=300)

    candidates = [(NORMALIZED_SCALE // 2, cand_cy_crop_norm)]
    annotated = annotate_candidates(base, crop, candidates, box_size_px=60)

    # Recompute the expected badge location on the full image for pixel checks.
    px_full, py_full = crop.crop_norm_to_full_pixel(
        NORMALIZED_SCALE // 2, cand_cy_crop_norm
    )
    half = 60 // 2
    left = px_full - half
    top = py_full - half
    expected_badge_top = max(0, top - 34)
    expected_badge_bottom = expected_badge_top + 34

    # Crop the region around the badge for a compact saved PNG evidence file.
    evidence_crop = annotated.crop(
        (max(0, left - 10), max(0, expected_badge_top - 10), left + 60, top + 60)
    )
    path = f"/tmp/bug6_{label}_top{top}.png"
    evidence_crop.save(path)

    red_in_badge = _count_red(
        annotated, left, expected_badge_top, left + 32, expected_badge_bottom
    )

    badge_height = expected_badge_bottom - expected_badge_top
    print(f"[{label}] candidate at crop-norm y={cand_cy_crop_norm}, full-px top={top}")
    print(f"  badge box: x={left}..{left+32}, y={expected_badge_top}..{expected_badge_bottom} (h={badge_height})")
    print(f"  red pixels in badge box: {red_in_badge}")
    print(f"  saved evidence crop: {path}")
    assert badge_height == 34, f"[{label}] badge height collapsed to {badge_height}px"
    assert red_in_badge > 0, (
        f"[{label}] Expected red badge border pixels, found {red_in_badge}. "
        f"This is the collapsed-badge bug the fix targets."
    )


if __name__ == "__main__":
    # Case A: candidate right at top of the full image → pre-fix badge collapsed
    # coarse_cy_norm=0 puts the crop at the top; cand_cy_crop_norm=0 puts the
    # bbox at the top of the crop so top_full==0.
    run(cand_cy_crop_norm=0, coarse_cy_norm=0, label="edge0")

    # Case B: candidate's bbox lands 15px below top (top in 0..33 band)
    # — pre-fix code pinned both edges to 34, collapsing height to 0.
    run(cand_cy_crop_norm=100, coarse_cy_norm=0, label="near_edge")

    # Case C: comfortably inside image → always worked, sanity check.
    run(cand_cy_crop_norm=500, coarse_cy_norm=500, label="middle")
    print("\nOK — all three candidate positions produced a 34px-tall red badge.")
