"""Text rendering helpers that convert glyph grids to grayscale canvases."""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import List, Sequence

from snn_ocr import bitfont, pgm

GrayGrid = List[List[int]]


def _clamp(value: float, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(round(value))))


def _deterministic_rng(text: str, width: int, height: int) -> Random:
    seed = (hash(text) ^ (width << 16) ^ height) & 0xFFFFFFFF
    return Random(seed)


def render_text(
    text: str,
    w: int,
    h: int,
    *,
    font: str = "5x7",
    jitter: int = 1,
    noise: float = 0.01,
    contrast: float = 1.0,
    scale: int | None = None,
    shear: int = 0,
    dilate: int = 0,
    foreground: int = 255,
    background: int = 0,
    seed: int | None = None,
) -> GrayGrid:
    """Render a single text row into a grayscale canvas of shape (h, w)."""
    if w <= 0 or h <= 0:
        raise ValueError("Target dimensions must be positive")
    if not text:
        return [[background for _ in range(w)] for _ in range(h)]
    bitfont.validate_charset(font, required=text)
    base = bitfont.render_text_row(text, size=font)
    base_h = len(base)
    base_w = len(base[0])
    if base_h > h or base_w > w:
        raise ValueError("Target canvas too small for requested text")
    if scale is None:
        auto_scale = max(1, min(w // base_w, h // base_h))
    else:
        auto_scale = max(1, scale)
    scaled = bitfont.scale_nn(base, auto_scale, auto_scale)
    if shear:
        scaled = bitfont.shear_x(scaled, shear)
    if dilate > 0:
        scaled = bitfont.dilate3x3(scaled, n=dilate)
    scaled_h = len(scaled)
    scaled_w = len(scaled[0])
    if scaled_h > h or scaled_w > w:
        raise ValueError("Transformed glyph exceeds target canvas; increase w/h or reduce transforms")
    glyph_gray = bitfont.to_gray(scaled, fg=foreground, bg=background)
    canvas: GrayGrid = [[background for _ in range(w)] for _ in range(h)]
    rng = _deterministic_rng(text if seed is None else f"{text}:{seed}", w, h)
    jitter = max(0, jitter)
    shift_x = rng.randint(-jitter, jitter) if jitter else 0
    shift_y = rng.randint(-jitter, jitter) if jitter else 0
    origin_x = max(0, min((w - scaled_w) // 2 + shift_x, w - scaled_w))
    origin_y = max(0, min((h - scaled_h) // 2 + shift_y, h - scaled_h))
    for y in range(scaled_h):
        canvas_row = canvas[origin_y + y]
        glyph_row = glyph_gray[y]
        for x in range(scaled_w):
            value = glyph_row[x]
            if value != background:
                canvas_row[origin_x + x] = value
    if noise > 0:
        amplitude = int(255 * min(1.0, max(0.0, noise)))
        for y in range(h):
            row = canvas[y]
            for x in range(w):
                delta = rng.randint(-amplitude, amplitude)
                row[x] = _clamp(row[x] + delta)
    if contrast != 1.0:
        for y in range(h):
            row = canvas[y]
            for x in range(w):
                row[x] = _clamp(background + (row[x] - background) * contrast)
    assert len(canvas) == h
    assert all(len(row) == w for row in canvas)
    return canvas


def ascii_preview(gray: GrayGrid) -> str:
    """Return an ASCII representation of a grayscale grid."""
    if not gray:
        return ""
    gradient = " .:-=+*#%@"
    v_min = min(min(row) for row in gray)
    v_max = max(max(row) for row in gray)
    if v_max == v_min:
        return "\n".join(gradient[-1] * len(row) for row in gray)
    span = v_max - v_min
    lines: List[str] = []
    for row in gray:
        line_chars = []
        for value in row:
            norm = (value - v_min) / span
            idx = min(len(gradient) - 1, max(0, int(round(norm * (len(gradient) - 1)))))
            line_chars.append(gradient[idx])
        lines.append("".join(line_chars))
    return "\n".join(lines)


def _self_check() -> None:
    sample = render_text("TEST", 64, 32, jitter=0, noise=0.0)
    assert len(sample) == 32
    assert len(sample[0]) == 64


if __name__ == "__main__":
    _self_check()
    canvas = render_text(
        "HELLO",
        160,
        48,
        jitter=1,
        noise=0.03,
        contrast=1.15,
        scale=2,
        shear=1,
        dilate=1,
        background=24,
    )
    out_path = Path("hello.pgm")
    pgm.save_pgm(out_path, canvas)
    print(f"Saved {out_path.resolve()}")
    print(ascii_preview(canvas))
