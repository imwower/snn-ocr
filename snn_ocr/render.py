"""Text rendering helpers that convert glyph grids to grayscale canvases."""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import List

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
) -> GrayGrid:
    """Render a single text row into a grayscale canvas of shape (h, w)."""
    if w <= 0 or h <= 0:
        raise ValueError("Target dimensions must be positive")
    if not text:
        return [[0 for _ in range(w)] for _ in range(h)]
    base = bitfont.render_text_row(text, size=font)
    base_h = len(base)
    base_w = len(base[0])
    if base_h > h or base_w > w:
        raise ValueError("Target canvas too small for requested text")
    scale = max(1, min(w // base_w, h // base_h))
    scaled = bitfont.scale_nn(base, scale, scale) if scale > 1 else [row.copy() for row in base]
    scaled_h = len(scaled)
    scaled_w = len(scaled[0])
    canvas: GrayGrid = [[0 for _ in range(w)] for _ in range(h)]
    rng = _deterministic_rng(text, w, h)
    jitter = max(0, jitter)
    shift_x = rng.randint(-jitter, jitter) if jitter else 0
    shift_y = rng.randint(-jitter, jitter) if jitter else 0
    origin_x = max(0, min((w - scaled_w) // 2 + shift_x, w - scaled_w))
    origin_y = max(0, min((h - scaled_h) // 2 + shift_y, h - scaled_h))
    for y in range(scaled_h):
        canvas_row = canvas[origin_y + y]
        glyph_row = scaled[y]
        for x in range(scaled_w):
            if glyph_row[x]:
                canvas_row[origin_x + x] = 255
    if noise > 0:
        noise = max(0.0, min(1.0, noise))
        for y in range(h):
            for x in range(w):
                if rng.random() < noise:
                    canvas[y][x] = 255 if rng.random() < 0.5 else 0
    if contrast != 1.0:
        for y in range(h):
            row = canvas[y]
            for x in range(w):
                row[x] = _clamp((row[x] - 128) * contrast + 128)
    assert len(canvas) == h
    assert all(len(row) == w for row in canvas)
    return canvas


def ascii_preview(gray: GrayGrid) -> str:
    """Return an ASCII representation of a grayscale grid."""
    return "\n".join("".join("#" if value > 128 else "." for value in row) for row in gray)


def _self_check() -> None:
    sample = render_text("TEST", 64, 32, jitter=0, noise=0.0)
    assert len(sample) == 32
    assert len(sample[0]) == 64


if __name__ == "__main__":
    _self_check()
    canvas = render_text("HELLO", 128, 32, jitter=1, noise=0.02, contrast=1.2)
    out_path = Path("hello.pgm")
    pgm.save_pgm(out_path, canvas)
    print(f"Saved {out_path.resolve()}")
    print(ascii_preview(canvas))
