"""Spike encoding utilities (TTFS and Poisson) with ON/OFF channel splitting."""
from __future__ import annotations

import math
from random import Random
from typing import List, Sequence, Tuple

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
        normalized.append([min(1.0, max(0.0, value / 255.0)) for value in row])
    return normalized, height, width


def _validate_spike_tensor(spikes: SpikeTensor) -> Tuple[int, int, int]:
    if not spikes:
        raise ValueError("Spike tensor must be non-empty")
    T = len(spikes)
    height = len(spikes[0])
    width = len(spikes[0][0]) if height else 0
    if height == 0 or width == 0:
        raise ValueError("Spike frames must be non-empty")
    for frame in spikes:
        if len(frame) != height:
            raise ValueError("Inconsistent frame height in spike tensor")
        for row in frame:
            if len(row) != width:
                raise ValueError("Inconsistent frame width in spike tensor")
    return T, height, width


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
            step = int(round((1.0 - intensity) * (T - 1)))
            step = max(0, min(T - 1, step))
            spikes[step][y][x] = 1
    return spikes


def encode_poisson(
    gray: GrayGrid,
    T: int,
    rate_scale: float = 1.0,
    seed: int = 0,
) -> SpikeTensor:
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
                lam = normalized[y][x] * rate_scale
                prob = 1.0 - math.exp(-lam)
                prob = max(0.0, min(1.0, prob))
                if rng.random() < prob:
                    spikes[t][y][x] = 1
    return spikes


def split_on_off(spikes: SpikeTensor) -> Tuple[SpikeTensor, SpikeTensor]:
    """Split spikes into ON (rise) and OFF (fall) streams."""
    T, height, width = _validate_spike_tensor(spikes)
    on = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(T)]
    off = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(T)]
    prev = [[0 for _ in range(width)] for _ in range(height)]
    for t in range(T):
        frame = spikes[t]
        for y in range(height):
            for x in range(width):
                value = frame[y][x]
                if value not in (0, 1):
                    raise ValueError("Spike values must be binary")
                if value == 1 and prev[y][x] == 0:
                    on[t][y][x] = 1
                if value == 0 and prev[y][x] == 1:
                    off[t][y][x] = 1
                prev[y][x] = value
    return on, off


def apply_micro_saccade(
    spikes: SpikeTensor,
    dx: int | Sequence[int],
    dy: int | Sequence[int],
) -> SpikeTensor:
    """Apply a per-frame shift to mimic micro-saccades."""
    T, height, width = _validate_spike_tensor(spikes)

    def _expand(offset: int | Sequence[int]) -> List[int]:
        if isinstance(offset, int):
            return [offset for _ in range(T)]
        if isinstance(offset, (list, tuple)):
            if len(offset) != T:
                raise ValueError("Offset sequence length must match T")
            return [int(value) for value in offset]
        raise TypeError("Offsets must be int or sequence of ints")

    dx_seq = _expand(dx)
    dy_seq = _expand(dy)
    shifted: SpikeTensor = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(T)]
    for t in range(T):
        offset_x = dx_seq[t]
        offset_y = dy_seq[t]
        for y in range(height):
            for x in range(width):
                if spikes[t][y][x] == 0:
                    continue
                new_x = x + offset_x
                new_y = y + offset_y
                if 0 <= new_x < width and 0 <= new_y < height:
                    shifted[t][new_y][new_x] = 1
    return shifted


def _self_check() -> None:
    dummy = [[0, 128, 255]]
    ttfs = encode_ttfs(dummy, T=4)
    assert sum(sum(sum(row) for row in frame) for frame in ttfs) == 2
    poisson = encode_poisson(dummy, T=4, seed=1)
    assert len(poisson) == 4
    on, off = split_on_off(ttfs)
    assert len(on) == len(off) == len(ttfs)
    shifted = apply_micro_saccade(ttfs, dx=1, dy=0)
    assert len(shifted) == len(ttfs)


if __name__ == "__main__":
    _self_check()
    sample = [[int(255 * (x % 2)) for x in range(8)] for _ in range(8)]
    ttfs_spikes = encode_ttfs(sample, T=4)
    poisson_spikes = encode_poisson(sample, T=4, seed=42)
    micro_spikes = apply_micro_saccade(poisson_spikes, dx=[0, 1, 0, -1], dy=[0, 0, 1, 0])
    on_channel, off_channel = split_on_off(ttfs_spikes)
    for name, tensor in [
        ("TTFS", ttfs_spikes),
        ("Poisson", poisson_spikes),
        ("Micro-saccade", micro_spikes),
        ("ON", on_channel),
        ("OFF", off_channel),
    ]:
        counts = [sum(sum(row) for row in frame) for frame in tensor]
        print(f"{name} spike counts per step: {counts}")
