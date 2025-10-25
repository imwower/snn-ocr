"""Text rendering helpers that convert glyph grids to grayscale canvases."""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import List, Sequence, Tuple

from snn_ocr import bitfont, pgm

GrayGrid = List[List[int]]


def _clamp(value: float, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(round(value))))


def _deterministic_rng(text: str, width: int, height: int) -> Random:
    seed = (hash(text) ^ (width << 16) ^ height) & 0xFFFFFFFF
    return Random(seed)


def _transform_line(
    glyph: bitfont.Grid,
    *,
    scale_factor: int,
    shear: int,
    dilate: int,
    foreground: int,
    background: int,
) -> GrayGrid:
    scaled = bitfont.scale_nn(glyph, scale_factor, scale_factor)
    if shear:
        scaled = bitfont.shear_x(scaled, shear)
    if dilate > 0:
        scaled = bitfont.dilate3x3(scaled, n=dilate)
    return bitfont.to_gray(scaled, fg=foreground, bg=background)


def _prepare_lines(text: str) -> List[str]:
    if text == "":
        return [" "]
    segments = text.split("\n")
    return [segment if segment else " " for segment in segments]


def _base_glyph(line: str, font: str) -> bitfont.Grid:
    bitfont.validate_charset(font, required=line)
    grid = bitfont.render_text_row(line, size=font)
    if not grid:
        grid = bitfont.render_text_row(" ", size=font)
    return grid


def _auto_scale(
    base_height: int,
    base_widths: Sequence[int],
    line_count: int,
    w: int,
    h: int,
    spacing_units: int,
) -> int:
    height_denom = line_count * base_height + max(0, line_count - 1) * spacing_units
    if height_denom <= 0:
        return 1
    height_limit = h // height_denom
    width_limit = min(w // max(1, width) for width in base_widths) if base_widths else w
    candidate = max(1, min(height_limit, width_limit))
    return candidate


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
    """Render text (single or multi-line) into a grayscale canvas of shape (h, w)."""
    if w <= 0 or h <= 0:
        raise ValueError("Target dimensions must be positive")
    if not text:
        return [[background for _ in range(w)] for _ in range(h)]
    lines = _prepare_lines(text)
    glyphs_raw: List[Tuple[str, bitfont.Grid]] = []
    for line in lines:
        glyphs_raw.append((line, _base_glyph(line, font)))
    base_height = len(glyphs_raw[0][1]) if glyphs_raw and glyphs_raw[0][1] else 1
    base_widths = [len(g[1][0]) if g[1] else 1 for g in glyphs_raw]
    spacing_units = max(1, base_height // 2)
    scale_factor = max(1, scale) if scale is not None else _auto_scale(
        base_height,
        base_widths,
        len(lines),
        w,
        h,
        spacing_units,
    )
    jitter = max(0, jitter)
    rng = _deterministic_rng(text if seed is None else f"{text}:{seed}", w, h)

    def _fit(scale_value: int) -> Tuple[int, int, int, List[GrayGrid]]:
        glyphs: List[GrayGrid] = []
        max_width = 0
        for _, glyph in glyphs_raw:
            transformed = _transform_line(
                glyph,
                scale_factor=scale_value,
                shear=shear,
                dilate=max(0, dilate),
                foreground=foreground,
                background=background,
            )
            glyphs.append(transformed)
            if transformed:
                max_width = max(max_width, len(transformed[0]))
        spacing_px = max(1, spacing_units * scale_value)
        total_height = sum(len(g) for g in glyphs)
        if len(glyphs) > 1:
            total_height += spacing_px * (len(glyphs) - 1)
        return (max_width if glyphs else 0), total_height, spacing_px, glyphs

    max_width, content_height, spacing_px, glyphs_scaled = _fit(scale_factor)
    if (max_width > w or content_height > h) and scale is None:
        candidate = scale_factor - 1
        while candidate >= 1:
            max_width, content_height, spacing_px, glyphs_scaled = _fit(candidate)
            if max_width <= w and content_height <= h:
                break
            candidate -= 1
        else:
            raise ValueError("Unable to fit multiline text within target canvas")
        scale_factor = candidate
    elif max_width > w or content_height > h:
        raise ValueError("Transformed glyph exceeds target canvas; adjust width/height")

    canvas: GrayGrid = [[background for _ in range(w)] for _ in range(h)]
    shift_x = rng.randint(-jitter, jitter) if jitter else 0
    shift_y = rng.randint(-jitter, jitter) if jitter else 0
    top_margin = max(0, min((h - content_height) // 2 + shift_y, h - content_height))
    cursor_y = top_margin
    for idx, glyph in enumerate(glyphs_scaled):
        if not glyph:
            if idx != len(glyphs_scaled) - 1:
                cursor_y += spacing_px
            continue
        glyph_h = len(glyph)
        glyph_w = len(glyph[0])
        origin_x = max(0, min((w - glyph_w) // 2 + shift_x, w - glyph_w))
        for y in range(glyph_h):
            canvas_row = canvas[cursor_y + y]
            glyph_row = glyph[y]
            for x in range(glyph_w):
                value = glyph_row[x]
                if value != background:
                    canvas_row[origin_x + x] = value
        cursor_y += glyph_h
        if idx != len(glyphs_scaled) - 1:
            cursor_y += spacing_px

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
    multiline = render_text("SNN\nOCR", 160, 64, jitter=1, noise=0.02, contrast=1.1)
    print("Two-line preview:\n" + ascii_preview(multiline))
