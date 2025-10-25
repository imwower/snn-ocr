"""Lightweight Netpbm (PGM/PPM) read/write helpers built with the Python standard library."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

GridGray = List[List[int]]
PixelRGB = Tuple[int, int, int]
GridRGB = List[List[PixelRGB]]

_WHITESPACE = b" \t\r\n"


def _validate_gray(grid: GridGray) -> Tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("Gray grid must be non-empty")
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("Inconsistent gray row width")
        for value in row:
            if not 0 <= value <= 255:
                raise ValueError("Gray values must lie in 0..255")
    return len(grid), width


def _validate_rgb(grid: GridRGB) -> Tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("RGB grid must be non-empty")
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("Inconsistent RGB row width")
        for r, g, b in row:
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError("RGB channel values must lie in 0..255")
    return len(grid), width


def _write_header(handle, magic: str, width: int, height: int) -> None:
    handle.write(f"{magic}\n{width} {height}\n255\n".encode("ascii"))


def save_pgm(path: Path, gray: GridGray) -> None:
    """Write a P5 binary PGM image to disk."""
    height, width = _validate_gray(gray)
    target = Path(path)
    with target.open("wb") as handle:
        _write_header(handle, "P5", width, height)
        for row in gray:
            handle.write(bytes(row))


def save_ppm(path: Path, rgb: GridRGB) -> None:
    """Write a P6 binary PPM image to disk."""
    height, width = _validate_rgb(rgb)
    target = Path(path)
    with target.open("wb") as handle:
        _write_header(handle, "P6", width, height)
        for row in rgb:
            handle.write(bytes(component for pixel in row for component in pixel))


def _read_token(handle) -> str:
    token = bytearray()
    while True:
        ch = handle.read(1)
        if not ch:
            if token:
                return token.decode("ascii")
            raise ValueError("Unexpected EOF while reading Netpbm header")
        if ch == b"#":
            handle.readline()
            continue
        if ch in _WHITESPACE:
            if token:
                return token.decode("ascii")
            continue
        token.append(ch[0])


def _read_header(path: Path) -> Tuple[str, int, int, int, int]:
    with Path(path).open("rb") as handle:
        magic = _read_token(handle)
        if magic not in ("P5", "P6"):
            raise ValueError("Unsupported Netpbm format")
        width = int(_read_token(handle))
        height = int(_read_token(handle))
        maxval = int(_read_token(handle))
        if maxval != 255:
            raise ValueError("Only maxval=255 Netpbm images are supported")
        data_offset = handle.tell()
    return magic, width, height, maxval, data_offset


def load_pgm(path: Path) -> GridGray:
    """Load a P5 binary PGM image into a 2D grayscale grid."""
    target = Path(path)
    magic, width, height, _, data_offset = _read_header(target)
    if magic != "P5":
        raise ValueError("Expected P5 magic for PGM image")
    expected = width * height
    raw = target.read_bytes()
    payload = raw[data_offset:]
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
