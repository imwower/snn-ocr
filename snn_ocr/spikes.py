"""Spike encoding utilities (TTFS and Poisson) with ON/OFF channel splitting."""
from __future__ import annotations

from random import Random
from typing import List, Tuple

GrayGrid = List[List[int]]
SpikeTensor = List[List[List[int]]]


def _normalize(gray: GrayGrid) -> Tuple[List[List[float]], int, int]:
    height = len(gray)
    width = len(gray[0]) if height else 0
    if height == 0 or width == 0:
        raise ValueError("Grayscale grid must be non-empty")
    normalized: List[List[float]] = []
    for row in gray:
        if len(row) != width:
            raise ValueError("Inconsistent row width in grayscale grid")
        normalized.append([value / 255.0 for value in row])
    return normalized, height, width


def encode_ttfs(gray: GrayGrid, T: int) -> SpikeTensor:
    """Encode grayscale intensities via Time-To-First-Spike (TTFS)."""
    if T <= 0:
        raise ValueError("Number of steps T must be positive")
    normalized, height, width = _normalize(gray)
    spikes: SpikeTensor = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(T)]
    for y in range(height):
        for x in range(width):
            intensity = normalized[y][x]
            if intensity <= 0.0:
                continue
            if intensity >= 1.0:
                step = 0
            else:
                step = int((1.0 - intensity) * (T - 1))
            spikes[step][y][x] = 1
    return spikes


def encode_poisson(gray: GrayGrid, T: int, rate_scale: float = 1.0, seed: int = 0) -> SpikeTensor:
    """Rate-code grayscale values into binary spikes using a Poisson process."""
    if T <= 0:
        raise ValueError("Number of steps T must be positive")
    if rate_scale <= 0:
        raise ValueError("rate_scale must be positive")
    normalized, height, width = _normalize(gray)
    spikes: SpikeTensor = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(T)]
    rng = Random(seed)
    for t in range(T):
        for y in range(height):
            for x in range(width):
                rate = normalized[y][x] * rate_scale
                if rng.random() < rate:
                    spikes[t][y][x] = 1
    return spikes


def split_on_off(spikes: SpikeTensor) -> Tuple[SpikeTensor, SpikeTensor]:
    """Split spikes into ON (positive change) and OFF (negative change) streams."""
    if not spikes:
        raise ValueError("Spike tensor must be non-empty")
    T = len(spikes)
    height = len(spikes[0])
    width = len(spikes[0][0]) if height else 0
    on = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(T)]
    off = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(T)]
    prev = [[0 for _ in range(width)] for _ in range(height)]
    for t in range(T):
        frame = spikes[t]
        if len(frame) != height or any(len(row) != width for row in frame):
            raise ValueError("Inconsistent frame dimensions in spike tensor")
        for y in range(height):
            for x in range(width):
                value = frame[y][x]
                if value not in (0, 1):
                    raise ValueError("Spike values must be binary")
                if value == 1 and prev[y][x] == 0:
                    on[t][y][x] = 1
                elif value == 0 and prev[y][x] == 1:
                    off[t][y][x] = 1
                prev[y][x] = value
    return on, off


def _self_check() -> None:
    dummy = [[0, 128, 255]]
    ttfs = encode_ttfs(dummy, T=4)
    assert sum(sum(sum(row) for row in frame) for frame in ttfs) == 2
    poisson = encode_poisson(dummy, T=4, seed=1)
    assert len(poisson) == 4
    on, off = split_on_off(ttfs)
    assert len(on) == len(off) == len(ttfs)


if __name__ == "__main__":
    _self_check()
    sample = [[int(255 * (x % 2)) for x in range(8)] for _ in range(8)]
    ttfs_spikes = encode_ttfs(sample, T=4)
    poisson_spikes = encode_poisson(sample, T=4, seed=42)
    on_channel, off_channel = split_on_off(ttfs_spikes)
    for name, tensor in [("TTFS", ttfs_spikes), ("Poisson", poisson_spikes), ("ON", on_channel), ("OFF", off_channel)]:
        counts = [sum(sum(row) for row in frame) for frame in tensor]
        print(f"{name} spike counts per step: {counts}")
