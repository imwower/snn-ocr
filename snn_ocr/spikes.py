"""Spike encoding utilities (TTFS and Poisson) with ON/OFF channel splitting."""
from __future__ import annotations

import math
from random import Random
from typing import Dict, List, Sequence, Tuple

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


def spike_histogram(spikes: SpikeTensor) -> List[int]:
    """Return per-timestep spike counts."""
    return [sum(sum(row) for row in frame) for frame in spikes]


def assert_ttfs_single_spike(spikes: SpikeTensor) -> None:
    """Ensure TTFS tensors fire at most once per pixel (across time)."""
    T, height, width = _validate_spike_tensor(spikes)
    seen = [[0 for _ in range(width)] for _ in range(height)]
    for t in range(T):
        frame = spikes[t]
        for y in range(height):
            for x in range(width):
                if frame[y][x]:
                    if seen[y][x]:
                        raise AssertionError(f"TTFS pixel ({x},{y}) fired more than once")
                    seen[y][x] = 1


def _random_offsets(
    T: int,
    jitter: int,
    seed: int | None,
    width: int,
    height: int,
) -> Tuple[List[int], List[int]]:
    if jitter <= 0:
        return [0] * T, [0] * T
    limit_x = min(jitter, max(0, width - 1))
    limit_y = min(jitter, max(0, height - 1))
    rng = Random(seed)
    dx = [rng.randint(-limit_x, limit_x) for _ in range(T)] if limit_x else [0] * T
    dy = [rng.randint(-limit_y, limit_y) for _ in range(T)] if limit_y else [0] * T
    return dx, dy


def encode_ttfs(
    gray: GrayGrid,
    T: int,
    *,
    jitter: int = 0,
    seed: int | None = None,
    return_stats: bool = False,
) -> SpikeTensor | Tuple[SpikeTensor, Dict[str, object]]:
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
    assert_ttfs_single_spike(spikes)
    if jitter > 0:
        dx, dy = _random_offsets(T, jitter, seed, width, height)
        spikes = apply_micro_saccade(spikes, dx=dx, dy=dy)
    if return_stats:
        hist = spike_histogram(spikes)
        summary = {
            "histogram": hist,
            "total": sum(hist),
            "jitter": jitter,
        }
        return spikes, summary
    return spikes


def encode_poisson(
    gray: GrayGrid,
    T: int,
    rate_scale: float = 1.0,
    seed: int = 0,
    *,
    jitter: int = 0,
    return_stats: bool = False,
) -> SpikeTensor | Tuple[SpikeTensor, Dict[str, object]]:
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
    if jitter > 0:
        spikes = apply_random_micro_saccade(spikes, jitter=jitter, seed=seed)
    if return_stats:
        hist = spike_histogram(spikes)
        summary = {"histogram": hist, "total": sum(hist), "jitter": jitter}
        return spikes, summary
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


def apply_random_micro_saccade(
    spikes: SpikeTensor,
    jitter: int = 1,
    seed: int | None = None,
) -> SpikeTensor:
    """Convenience wrapper that samples offsets in [-jitter, jitter]."""
    if jitter <= 0:
        return spikes
    T, height, width = _validate_spike_tensor(spikes)
    dx, dy = _random_offsets(T, jitter, seed, width, height)
    return apply_micro_saccade(spikes, dx=dx, dy=dy)


def alpha_schedule(layers: int, base_tau: float = 4.0, decay: float = 0.85) -> List[float]:
    """Return per-layer alpha = exp(-1/tau) schedule for multi-tau processing."""
    if layers <= 0:
        raise ValueError("layers must be positive")
    if base_tau <= 0.0:
        raise ValueError("base_tau must be positive")
    if not (0.0 < decay <= 1.0):
        raise ValueError("decay must be in (0, 1]")
    alphas: List[float] = []
    current_tau = base_tau
    for _ in range(layers):
        alpha = math.exp(-1.0 / current_tau)
        alphas.append(alpha)
        current_tau *= decay
    return alphas


def spike_summary(spikes: SpikeTensor) -> Dict[str, object]:
    hist = spike_histogram(spikes)
    total = sum(hist)
    return {
        "histogram": hist,
        "total": total,
        "mean_per_step": total / max(1, len(hist)),
    }


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
    import argparse

    _self_check()

    parser = argparse.ArgumentParser(description="Spike encoder diagnostics.")
    parser.add_argument("--mode", choices=["ttfs", "poisson"], default="ttfs")
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--jitter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-tau", type=int, default=0, help="Print alpha schedule for N layers when >0.")
    args = parser.parse_args()

    width = 8
    sample = [[int(255 * (x / (width - 1))) for x in range(width)]]
    if args.mode == "ttfs":
        result = encode_ttfs(sample, args.timesteps, jitter=args.jitter, seed=args.seed, return_stats=True)
    else:
        result = encode_poisson(
            sample,
            args.timesteps,
            rate_scale=args.rate_scale,
            seed=args.seed,
            jitter=args.jitter,
            return_stats=True,
        )
    spikes_tensor, stats = result  # type: ignore[assignment]
    print(f"[{args.mode.upper()}] histogram per step: {stats['histogram']}")
    print(f"Total spikes: {stats['total']}")
    if args.show_tau > 0:
        print("Alpha schedule:", alpha_schedule(args.show_tau))
