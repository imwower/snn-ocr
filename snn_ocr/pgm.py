"""Lightweight Netpbm (PGM/PPM) read/write helpers built with the Python standard library."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

GridGray = List[List[int]]
PixelRGB = Tuple[int, int, int]
GridRGB = List[List[PixelRGB]]


def _validate_gray(grid: GridGray) -> None:
    if not grid or not grid[0]:
        raise ValueError("Gray grid must be non-empty")
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("Inconsistent gray row width")
        for value in row:
            if not 0 <= value <= 255:
                raise ValueError("Gray values must lie in 0..255")


def _validate_rgb(grid: GridRGB) -> None:
    if not grid or not grid[0]:
        raise ValueError("RGB grid must be non-empty")
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("Inconsistent RGB row width")
        for r, g, b in row:
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError("RGB channel values must lie in 0..255")


def save_pgm(path: Path, gray: GridGray) -> None:
    """Write a P5 binary PGM image to disk."""
    _validate_gray(gray)
    height = len(gray)
    width = len(gray[0])
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    body = bytearray(value for row in gray for value in row)
    Path(path).write_bytes(header + body)


def save_ppm(path: Path, rgb: GridRGB) -> None:
    """Write a P6 binary PPM image to disk."""
    _validate_rgb(rgb)
    height = len(rgb)
    width = len(rgb[0])
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    body = bytearray(component for row in rgb for pixel in row for component in pixel)
    Path(path).write_bytes(header + body)


def _parse_header(buffer: bytes) -> Tuple[str, int, int, int, int]:
    """Return (magic, width, height, maxval, offset) for Netpbm data."""
    if len(buffer) < 11:
        raise ValueError("Buffer too small to be a Netpbm image")
    magic = buffer[:2]
    if magic not in (b"P5", b"P6"):
        raise ValueError("Unsupported Netpbm magic number")
    tokens: List[str] = []
    idx = 2
    current = bytearray()
    while idx < len(buffer) and len(tokens) < 3:
        byte = buffer[idx]
        if byte == 35:  # Comment start '#'
            while idx < len(buffer) and buffer[idx] not in (10, 13):
                idx += 1
        elif byte in (9, 10, 13, 32):
            if current:
                tokens.append(current.decode("ascii"))
                current.clear()
        else:
            current.append(byte)
        idx += 1
    if current:
        tokens.append(current.decode("ascii"))
    if len(tokens) < 3:
        raise ValueError("Incomplete Netpbm header")
    width, height, maxval = map(int, tokens[:3])
    if maxval != 255:
        raise ValueError("Only maxval=255 images are supported")
    return magic.decode("ascii"), width, height, maxval, idx


def load_pgm(path: Path) -> GridGray:
    """Load a P5 binary PGM image into a 2D grayscale grid."""
    raw = Path(path).read_bytes()
    magic, width, height, _, offset = _parse_header(raw)
    if magic != "P5":
        raise ValueError("Expected P5 magic for PGM image")
    payload = raw[offset:]
    expected = width * height
    if len(payload) != expected:
        raise ValueError("Unexpected payload length for PGM image")
    grid: GridGray = []
    cursor = 0
    for _ in range(height):
        row = list(payload[cursor : cursor + width])
        grid.append(row)
        cursor += width
    return grid


def _checkerboard(width: int = 64, height: int = 32, cell: int = 8) -> GridGray:
    grid: GridGray = []
    for y in range(height):
        row: List[int] = []
        for x in range(width):
            toggle = ((x // cell) + (y // cell)) % 2
            row.append(255 if toggle else 0)
        grid.append(row)
    return grid


def _self_check() -> None:
    pattern = _checkerboard()
    assert len(pattern) == 32 and len(pattern[0]) == 64
    assert sum(len(row) for row in pattern) == 64 * 32
    tmp_path = Path("_tmp_check.pgm")
    save_pgm(tmp_path, pattern)
    loaded = load_pgm(tmp_path)
    assert loaded == pattern
    tmp_path.unlink(missing_ok=True)


def _grid_to_ascii(grid: GridGray) -> str:
    return "\n".join("".join("#" if cell else "." for cell in row) for row in grid)


if __name__ == "__main__":
    _self_check()
    demo = _checkerboard()
    out_path = Path("checkerboard.pgm")
    save_pgm(out_path, demo)
    print(f"Saved {out_path.resolve()}")
    preview_rows = demo[:8]
    print(_grid_to_ascii(preview_rows))
