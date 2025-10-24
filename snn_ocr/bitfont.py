"""Bitmap font utilities for the SNN-OCR project.

This module provides minimal 5x7 and 7x9 bitmap glyphs along with helpers for
rendering single-line text, applying simple geometric transforms, and converting
boolean grids into grayscale intensity maps.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from types import MappingProxyType

Grid = List[List[int]]


def _parse_patterns(patterns: Mapping[str, Sequence[str]]) -> Mapping[str, Tuple[Tuple[int, ...], ...]]:
    """Convert textual bitmap patterns into immutable integer grids."""
    parsed: Dict[str, Tuple[Tuple[int, ...], ...]] = {}
    for ch, rows in patterns.items():
        if not rows:
            raise ValueError(f"Pattern for {ch!r} is empty")
        row_length = len(rows[0])
        numeric_rows: List[Tuple[int, ...]] = []
        for row in rows:
            if len(row) != row_length:
                raise ValueError(f"Inconsistent row length for {ch!r}")
            numeric_rows.append(tuple(1 if pix == "1" else 0 for pix in row))
        parsed[ch] = tuple(numeric_rows)
    return MappingProxyType(parsed)


_PATTERNS_5X7: Mapping[str, Sequence[str]] = MappingProxyType(
    {
        "0": (
            "01110",
            "10001",
            "10011",
            "10101",
            "11001",
            "10001",
            "01110",
        ),
        "1": (
            "00100",
            "01100",
            "00100",
            "00100",
            "00100",
            "00100",
            "01110",
        ),
        "2": (
            "01110",
            "10001",
            "00001",
            "00010",
            "00100",
            "01000",
            "11111",
        ),
        "3": (
            "01110",
            "10001",
            "00001",
            "00110",
            "00001",
            "10001",
            "01110",
        ),
        "4": (
            "00010",
            "00110",
            "01010",
            "10010",
            "11111",
            "00010",
            "00010",
        ),
        "5": (
            "11111",
            "10000",
            "11110",
            "00001",
            "00001",
            "10001",
            "01110",
        ),
        "6": (
            "00110",
            "01000",
            "10000",
            "11110",
            "10001",
            "10001",
            "01110",
        ),
        "7": (
            "11111",
            "00001",
            "00010",
            "00100",
            "01000",
            "01000",
            "01000",
        ),
        "8": (
            "01110",
            "10001",
            "10001",
            "01110",
            "10001",
            "10001",
            "01110",
        ),
        "9": (
            "01110",
            "10001",
            "10001",
            "01111",
            "00001",
            "00010",
            "11100",
        ),
        "A": (
            "01110",
            "10001",
            "10001",
            "11111",
            "10001",
            "10001",
            "10001",
        ),
        "B": (
            "11110",
            "10001",
            "10001",
            "11110",
            "10001",
            "10001",
            "11110",
        ),
        "C": (
            "01110",
            "10001",
            "10000",
            "10000",
            "10000",
            "10001",
            "01110",
        ),
        "D": (
            "11100",
            "10010",
            "10001",
            "10001",
            "10001",
            "10010",
            "11100",
        ),
        "E": (
            "11111",
            "10000",
            "10000",
            "11100",
            "10000",
            "10000",
            "11111",
        ),
        "F": (
            "11111",
            "10000",
            "10000",
            "11100",
            "10000",
            "10000",
            "10000",
        ),
        "G": (
            "01110",
            "10001",
            "10000",
            "10111",
            "10001",
            "10001",
            "01110",
        ),
        "H": (
            "10001",
            "10001",
            "10001",
            "11111",
            "10001",
            "10001",
            "10001",
        ),
        "I": (
            "01110",
            "00100",
            "00100",
            "00100",
            "00100",
            "00100",
            "01110",
        ),
        "J": (
            "00001",
            "00001",
            "00001",
            "00001",
            "10001",
            "10001",
            "01110",
        ),
        "K": (
            "10001",
            "10010",
            "10100",
            "11000",
            "10100",
            "10010",
            "10001",
        ),
        "L": (
            "10000",
            "10000",
            "10000",
            "10000",
            "10000",
            "10000",
            "11111",
        ),
        "M": (
            "10001",
            "11011",
            "10101",
            "10101",
            "10001",
            "10001",
            "10001",
        ),
        "N": (
            "10001",
            "10001",
            "11001",
            "10101",
            "10011",
            "10001",
            "10001",
        ),
        "O": (
            "01110",
            "10001",
            "10001",
            "10001",
            "10001",
            "10001",
            "01110",
        ),
        "P": (
            "11110",
            "10001",
            "10001",
            "11110",
            "10000",
            "10000",
            "10000",
        ),
        "Q": (
            "01110",
            "10001",
            "10001",
            "10001",
            "10101",
            "10010",
            "01101",
        ),
        "R": (
            "11110",
            "10001",
            "10001",
            "11110",
            "10100",
            "10010",
            "10001",
        ),
        "S": (
            "01111",
            "10000",
            "10000",
            "01110",
            "00001",
            "00001",
            "11110",
        ),
        "T": (
            "11111",
            "00100",
            "00100",
            "00100",
            "00100",
            "00100",
            "00100",
        ),
        "U": (
            "10001",
            "10001",
            "10001",
            "10001",
            "10001",
            "10001",
            "01110",
        ),
        "V": (
            "10001",
            "10001",
            "10001",
            "10001",
            "10001",
            "01010",
            "00100",
        ),
        "W": (
            "10001",
            "10001",
            "10001",
            "10101",
            "10101",
            "10101",
            "01010",
        ),
        "X": (
            "10001",
            "01010",
            "00100",
            "00100",
            "01010",
            "10001",
            "10001",
        ),
        "Y": (
            "10001",
            "01010",
            "00100",
            "00100",
            "00100",
            "00100",
            "00100",
        ),
        "Z": (
            "11111",
            "00001",
            "00010",
            "00100",
            "01000",
            "10000",
            "11111",
        ),
        "a": (
            "00000",
            "00000",
            "01110",
            "00001",
            "01111",
            "10001",
            "01111",
        ),
        "b": (
            "10000",
            "10000",
            "11110",
            "10001",
            "10001",
            "10001",
            "11110",
        ),
        "c": (
            "00000",
            "00000",
            "01110",
            "10001",
            "10000",
            "10001",
            "01110",
        ),
        "d": (
            "00001",
            "00001",
            "01111",
            "10001",
            "10001",
            "10001",
            "01111",
        ),
        "e": (
            "00000",
            "00000",
            "01110",
            "10001",
            "11111",
            "10000",
            "01110",
        ),
        "f": (
            "00110",
            "01001",
            "01000",
            "11100",
            "01000",
            "01000",
            "01000",
        ),
        "g": (
            "00000",
            "00000",
            "01111",
            "10001",
            "10001",
            "01111",
            "00001",
        ),
        "h": (
            "10000",
            "10000",
            "11110",
            "10001",
            "10001",
            "10001",
            "10001",
        ),
        "i": (
            "00100",
            "00000",
            "01100",
            "00100",
            "00100",
            "00100",
            "01110",
        ),
        "j": (
            "00010",
            "00000",
            "00110",
            "00010",
            "00010",
            "10010",
            "01100",
        ),
        "k": (
            "10000",
            "10000",
            "10010",
            "10100",
            "11000",
            "10100",
            "10010",
        ),
        "l": (
            "01100",
            "00100",
            "00100",
            "00100",
            "00100",
            "00100",
            "01110",
        ),
        "m": (
            "00000",
            "00000",
            "11010",
            "10101",
            "10101",
            "10101",
            "10101",
        ),
        "n": (
            "00000",
            "00000",
            "11110",
            "10001",
            "10001",
            "10001",
            "10001",
        ),
        "o": (
            "00000",
            "00000",
            "01110",
            "10001",
            "10001",
            "10001",
            "01110",
        ),
        "p": (
            "00000",
            "00000",
            "11110",
            "10001",
            "10001",
            "11110",
            "10000",
        ),
        "q": (
            "00000",
            "00000",
            "01111",
            "10001",
            "10001",
            "01111",
            "00001",
        ),
        "r": (
            "00000",
            "00000",
            "10110",
            "11001",
            "10000",
            "10000",
            "10000",
        ),
        "s": (
            "00000",
            "00000",
            "01111",
            "10000",
            "01110",
            "00001",
            "11110",
        ),
        "t": (
            "01000",
            "01000",
            "11100",
            "01000",
            "01000",
            "01001",
            "00110",
        ),
        "u": (
            "00000",
            "00000",
            "10001",
            "10001",
            "10001",
            "10011",
            "01101",
        ),
        "v": (
            "00000",
            "00000",
            "10001",
            "10001",
            "10001",
            "01010",
            "00100",
        ),
        "w": (
            "00000",
            "00000",
            "10001",
            "10101",
            "10101",
            "10101",
            "01010",
        ),
        "x": (
            "00000",
            "00000",
            "10001",
            "01010",
            "00100",
            "01010",
            "10001",
        ),
        "y": (
            "00000",
            "00000",
            "10001",
            "10001",
            "10001",
            "01111",
            "00001",
        ),
        "z": (
            "00000",
            "00000",
            "11111",
            "00010",
            "00100",
            "01000",
            "11111",
        ),
        " ": (
            "00000",
            "00000",
            "00000",
            "00000",
            "00000",
            "00000",
            "00000",
        ),
        ".": (
            "00000",
            "00000",
            "00000",
            "00000",
            "00000",
            "01100",
            "01100",
        ),
        "!": (
            "00100",
            "00100",
            "00100",
            "00100",
            "00100",
            "00000",
            "00100",
        ),
        "?": (
            "01110",
            "10001",
            "00001",
            "00010",
            "00100",
            "00000",
            "00100",
        ),
    }
)


def _expand_to_7x9(pattern: Sequence[str]) -> Tuple[str, ...]:
    """Embed a 5x7 pattern into a 7x9 canvas with quiet borders."""
    width = len(pattern[0])
    padded: List[str] = ["0" * (width + 2)]
    padded.extend(f"0{row}0" for row in pattern)
    padded.append("0" * (width + 2))
    return tuple(padded)


_PATTERNS_7X9: Dict[str, Tuple[str, ...]] = {
    ch: _expand_to_7x9(rows) for ch, rows in _PATTERNS_5X7.items()
}

FONT_5X7 = _parse_patterns(_PATTERNS_5X7)
FONT_7X9 = _parse_patterns(_PATTERNS_7X9)
FONTS: Mapping[str, Mapping[str, Tuple[Tuple[int, ...], ...]]] = MappingProxyType(
    {"5x7": FONT_5X7, "7x9": FONT_7X9}
)


def get_bitmap(ch: str, size: str = "5x7") -> Grid:
    """Return a copy of the bitmap for a single character."""
    if len(ch) != 1:
        raise ValueError("Expected a single character")
    font = FONTS.get(size)
    if font is None:
        raise KeyError(f"Unsupported size {size!r}")
    glyph = font.get(ch)
    if glyph is None:
        raise KeyError(f"Character {ch!r} missing from font {size}")
    return [list(row) for row in glyph]


def render_text_row(text: str, size: str = "5x7", inner: int = 1, outer: int = 2) -> Grid:
    """Render text as a single bitmap row with configurable spacing."""
    if inner < 0 or outer < 0:
        raise ValueError("inner and outer spacings must be non-negative")
    if not text:
        return []
    glyphs = [get_bitmap(ch, size=size) for ch in text]
    height = len(glyphs[0])
    if any(len(glyph) != height for glyph in glyphs):
        raise ValueError("Mixed glyph heights are not supported")
    result: Grid = [[] for _ in range(height)]
    for row in result:
        row.extend([0] * outer)
    for index, glyph in enumerate(glyphs):
        for y, row in enumerate(glyph):
            result[y].extend(row)
        if index != len(glyphs) - 1:
            for row in result:
                row.extend([0] * inner)
    for row in result:
        row.extend([0] * outer)
    return result


def scale_nn(grid: Grid, sx: int, sy: int) -> Grid:
    """Nearest-neighbour scaling."""
    if sx <= 0 or sy <= 0:
        raise ValueError("Scale factors must be positive")
    if not grid:
        return []
    scaled: Grid = []
    for row in grid:
        stretched_row = [pix for pix in row for _ in range(sx)]
        for _ in range(sy):
            scaled.append(stretched_row.copy())
    return scaled


def dilate3x3(grid: Grid, n: int = 1) -> Grid:
    """Apply n iterations of 3x3 dilation."""
    if n <= 0:
        return [row.copy() for row in grid]
    if not grid:
        return []
    height = len(grid)
    width = len(grid[0])
    current = [row.copy() for row in grid]
    for _ in range(n):
        expanded = [[0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                if current[y][x]:
                    expanded[y][x] = 1
                    continue
                for dy in (-1, 0, 1):
                    ny = y + dy
                    if ny < 0 or ny >= height:
                        continue
                    for dx in (-1, 0, 1):
                        nx = x + dx
                        if 0 <= nx < width and current[ny][nx]:
                            expanded[y][x] = 1
                            break
                    if expanded[y][x]:
                        break
        current = expanded
    return current


def shear_x(grid: Grid, px: int) -> Grid:
    """Shear along the x-axis by an integer amount across the height."""
    if not grid:
        return []
    height = len(grid)
    width = len(grid[0])
    if height == 1 or px == 0:
        return [row.copy() for row in grid]
    denominator = max(height - 1, 1)
    shifts = [(row * px) // denominator for row in range(height)]
    min_shift = min(shifts)
    max_shift = max(shifts)
    new_width = width + (max_shift - min_shift)
    sheared = [[0] * new_width for _ in range(height)]
    for y, row in enumerate(grid):
        offset = shifts[y] - min_shift
        for x, pix in enumerate(row):
            if pix:
                sheared[y][x + offset] = 1
    return sheared


def to_gray(grid: Grid, fg: int = 255, bg: int = 0) -> Grid:
    """Map boolean bitmap values to grayscale intensities."""
    if fg == bg:
        raise ValueError("Foreground and background intensities must differ")
    return [[fg if pix else bg for pix in row] for row in grid]


def _run_self_checks() -> None:
    """Basic invariants used as lightweight regression checks."""
    assert len(get_bitmap("A", "5x7")) == 7
    assert len(get_bitmap("A", "7x9")) == 9
    assert len(get_bitmap("0", "5x7")[0]) == 5
    assert len(get_bitmap("0", "7x9")[0]) == 7
    demo = render_text_row("Aa0", size="5x7")
    assert len(demo) == 7
    assert all(len(row) == len(demo[0]) for row in demo)
    scaled = scale_nn(get_bitmap("B"), 2, 2)
    assert len(scaled) == 14 and len(scaled[0]) == 10
    dilated = dilate3x3(get_bitmap("I"), 1)
    assert len(dilated) == 7 and len(dilated[0]) == 5
    sheared = shear_x(get_bitmap("W"), 2)
    assert len(sheared[0]) >= len(get_bitmap("W")[0])


def _grid_to_ascii(grid: Grid) -> str:
    """Convert a bitmap grid into ASCII art for quick previews."""
    return "\n".join("".join("#" if pix else "." for pix in row) for row in grid)


if __name__ == "__main__":
    _run_self_checks()
    demo_grid = render_text_row("Ab9", size="7x9")
    demo_gray = to_gray(demo_grid)
    print("ASCII preview of 'Ab9':")
    print(_grid_to_ascii(demo_grid))
    print("\nGrayscale sample values:")
    for row in demo_gray:
        print(row)
