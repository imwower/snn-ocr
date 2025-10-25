"""Sequence head layers for SNN OCR: depthwise conv, pointwise mixing, optional attention."""
from __future__ import annotations

import math
from random import Random
from typing import List, Tuple

SequenceTensor = List[List[List[float]]]  # T x W x C


def _validate_seq(seq: SequenceTensor) -> Tuple[int, int, int]:
    if not seq:
        raise ValueError("sequence tensor must be non-empty")
    T = len(seq)
    W = len(seq[0])
    if W == 0:
        raise ValueError("Width dimension must be positive")
    C = len(seq[0][0])
    if C == 0:
        raise ValueError("Channel dimension must be positive")
    for t in range(T):
        if len(seq[t]) != W:
            raise ValueError("Inconsistent width across time")
        for w in range(W):
            if len(seq[t][w]) != C:
                raise ValueError("Inconsistent channel count")
    return T, W, C


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _layer_norm_column(column: List[float], eps: float = 1e-5) -> List[float]:
    mean = sum(column) / len(column)
    variance = sum((value - mean) ** 2 for value in column) / len(column)
    scale = 1.0 / math.sqrt(variance + eps)
    return [(value - mean) * scale for value in column]


def _layer_norm(seq: SequenceTensor) -> SequenceTensor:
    T, W, C = _validate_seq(seq)
    normalized: SequenceTensor = []
    for t in range(T):
        step: List[List[float]] = []
        for w in range(W):
            step.append(_layer_norm_column(seq[t][w][:]))
        normalized.append(step)
    return normalized


def _residual_norm(residual: SequenceTensor, update: SequenceTensor) -> SequenceTensor:
    T, W, C = _validate_seq(residual)
    _validate_seq(update)
    combined: SequenceTensor = []
    for t in range(T):
        step: List[List[float]] = []
        for w in range(W):
            column = [residual[t][w][c] + update[t][w][c] for c in range(C)]
            step.append(_layer_norm_column(column))
        combined.append(step)
    return combined


def dwconv1d_spike(seq: SequenceTensor, k: int = 3) -> SequenceTensor:
    """Depthwise convolution across width with spike-like non-linearity."""
    T, W, C = _validate_seq(seq)
    if k <= 0 or k % 2 == 0:
        raise ValueError("Kernel size k must be positive and odd")
    pad = k // 2
    kernel = [
        1.0 - abs(offset) / (pad + 1) for offset in range(-pad, pad + 1)
    ]  # triangular weights
    kernel_sum = sum(kernel)
    result: SequenceTensor = []
    for t in range(T):
        step_out: List[List[float]] = []
        for w in range(W):
            column_out: List[float] = []
            for c in range(C):
                acc = 0.0
                for offset, weight in zip(range(-pad, pad + 1), kernel):
                    idx = w + offset
                    if 0 <= idx < W:
                        acc += seq[t][idx][c] * weight
                acc /= kernel_sum
                column_out.append(_sigmoid(acc))
            step_out.append(column_out)
        result.append(step_out)
    return result


def pointwise_spike(seq: SequenceTensor) -> SequenceTensor:
    """Pointwise channel mixing with a spike-inspired activation."""
    T, W, C = _validate_seq(seq)
    result: SequenceTensor = []
    for t in range(T):
        step_out: List[List[float]] = []
        for w in range(W):
            column = seq[t][w]
            mean = sum(column) / C
            row_out: List[float] = []
            for c in range(C):
                neighbor = column[(c + 1) % C]
                mixed = 0.7 * column[c] + 0.3 * neighbor - mean
                row_out.append(_sigmoid(mixed))
            step_out.append(row_out)
        result.append(step_out)
    return result


def _positive_feature(vec: List[float]) -> List[float]:
    dim = len(vec)
    if dim == 0:
        return []
    eps = 1e-3
    scale = 1.0 / dim
    return [max(0.0, value) * scale + eps for value in vec]


def linear_attention(
    seq: SequenceTensor,
    heads: int = 2,
    key_dim: int | None = None,
) -> SequenceTensor:
    """Lightweight linear attention using positive feature kernels."""
    T, W, C = _validate_seq(seq)
    if heads <= 0:
        raise ValueError("heads must be positive")
    if key_dim is None:
        key_dim = max(1, C // heads)
    key_dim = max(1, min(key_dim, C))
    result: SequenceTensor = []
    for t in range(T):
        step_out = [seq[t][w][:] for w in range(W)]
        for head in range(heads):
            start = head * key_dim
            if start >= C:
                break
            end = min(C, start + key_dim)
            dim = end - start
            if dim <= 0:
                continue
            context = [0.0 for _ in range(dim)]
            norm = [1e-6 for _ in range(dim)]
            for j in range(W):
                slice_j = seq[t][j][start:end]
                phi_k = _positive_feature(slice_j)
                for d in range(dim):
                    context[d] += phi_k[d] * slice_j[d]
                    norm[d] += phi_k[d]
            for w in range(W):
                slice_w = seq[t][w][start:end]
                phi_q = _positive_feature(slice_w)
                for d in range(dim):
                    attn = phi_q[d] * context[d] / norm[d]
                    step_out[w][start + d] += attn
        result.append(step_out)
    return result


def merge_columns_blockwise(
    sequence: List[List[float]],
    *,
    max_merge: int = 4,
    window: int = 16,
) -> List[List[float]]:
    """Merge adjacent low-information columns using a sliding window threshold."""
    if not sequence:
        return []
    if max_merge <= 0:
        return [column[:] for column in sequence]
    feature_dim = len(sequence[0])
    info_scores: List[float] = []
    for column in sequence:
        values = column[:-1] if len(column) > 1 else column[:]
        if not values:
            info_scores.append(0.0)
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        info_scores.append(variance)
    window = max(1, window)

    def local_threshold(idx: int) -> float:
        start = max(0, idx - window + 1)
        subset = info_scores[start : idx + 1]
        return sum(subset) / max(1, len(subset))

    merged: List[List[float]] = []
    counts: List[int] = []
    for idx, column in enumerate(sequence):
        if (
            merged
            and info_scores[idx] <= local_threshold(idx)
            and counts[-1] < max_merge
        ):
            prev_count = counts[-1]
            combined = merged[-1]
            new_count = prev_count + 1
            if len(combined) != len(column):
                raise ValueError("Feature dimension changed within sequence")
            for dim in range(feature_dim):
                combined[dim] = (combined[dim] * prev_count + column[dim]) / new_count
            counts[-1] = new_count
        else:
            merged.append(column[:])
            counts.append(1)
    return merged


class SpikingSeqHead:
    """Two-layer spike sequence head with optional attention and stability regularizers."""

    def __init__(
        self,
        dw_kernel: int = 3,
        use_attention: bool = True,
        heads: int = 2,
        key_dim: int | None = None,
        *,
        attn_type: str = "linear",
        drop_path: float = 0.0,
        smooth_lambda: float = 0.0,
        smooth_mode: str = "l2",
        huber_delta: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.dw_kernel = dw_kernel
        self.use_attention = use_attention
        self.heads = heads
        self.key_dim = key_dim
        self.attn_type = attn_type.lower()
        if self.attn_type not in ("linear", "none"):
            raise ValueError("attn_type must be 'linear' or 'none'")
        self.drop_path = max(0.0, min(0.999, drop_path))
        self.smooth_lambda = max(0.0, smooth_lambda)
        self.smooth_mode = smooth_mode.lower()
        if self.smooth_mode not in ("l2", "huber", "none"):
            raise ValueError("smooth_mode must be 'l2', 'huber', or 'none'")
        self.huber_delta = max(1e-4, huber_delta)
        self.training = True
        self._rng = Random(seed)
        self._last_temporal_penalty = 0.0

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def regularization(self) -> float:
        return self._last_temporal_penalty

    def _apply_droppath(self, residual: SequenceTensor, update: SequenceTensor) -> SequenceTensor:
        if not self.training or self.drop_path <= 0.0:
            return _residual_norm(residual, update)
        if self._rng.random() < self.drop_path:
            return residual
        keep_prob = 1.0 - self.drop_path
        scaled: SequenceTensor = []
        for t in range(len(update)):
            step: List[List[float]] = []
            for w in range(len(update[t])):
                step.append([value / keep_prob for value in update[t][w]])
            scaled.append(step)
        return _residual_norm(residual, scaled)

    def _temporal_penalty(self, seq: SequenceTensor) -> float:
        if self.smooth_lambda <= 0.0 or self.smooth_mode == "none":
            return 0.0
        T, W, C = _validate_seq(seq)
        if T < 2:
            return 0.0
        total = 0.0
        count = 0
        delta = self.huber_delta
        for t in range(1, T):
            prev = seq[t - 1]
            curr = seq[t]
            for w in range(W):
                for c in range(C):
                    diff = curr[w][c] - prev[w][c]
                    if self.smooth_mode == "huber":
                        abs_diff = abs(diff)
                        if abs_diff <= delta:
                            total += 0.5 * diff * diff
                        else:
                            total += delta * (abs_diff - 0.5 * delta)
                    else:
                        total += diff * diff
                    count += 1
        return self.smooth_lambda * total / max(1, count)

    def forward(self, seq: SequenceTensor, *, use_attention: bool | None = None) -> SequenceTensor:
        _validate_seq(seq)
        attn_requested = self.use_attention if use_attention is None else use_attention
        attn_mode = self.attn_type if attn_requested else "none"
        x = seq
        for _ in range(2):
            conv = dwconv1d_spike(x, k=self.dw_kernel)
            x = self._apply_droppath(x, conv)
            pw = pointwise_spike(x)
            if attn_mode == "linear":
                attn = linear_attention(pw, heads=self.heads, key_dim=self.key_dim)
                x = self._apply_droppath(pw, attn)
            else:
                x = _layer_norm(pw)
        self._last_temporal_penalty = self._temporal_penalty(x)
        return x


def _self_check() -> None:
    sample = [[[0.1, 0.2, 0.3] for _ in range(5)] for _ in range(2)]
    smoothed = dwconv1d_spike(sample, k=3)
    mixed = pointwise_spike(smoothed)
    attn = linear_attention(mixed, heads=2)
    assert len(attn) == len(sample)
    assert len(attn[0]) == len(sample[0])
    assert len(attn[0][0]) == len(sample[0][0])
    head = SpikingSeqHead(dw_kernel=3, use_attention=True, heads=2)
    out = head.forward(sample)
    assert len(out) == len(sample)
    assert len(out[0]) == len(sample[0])


if __name__ == "__main__":
    import argparse

    _self_check()

    parser = argparse.ArgumentParser(description="Spiking sequence head stability demo.")
    parser.add_argument("--attn", choices=["none", "linear"], default="linear")
    parser.add_argument("--steps", type=int, default=100, help="Number of random forward passes.")
    parser.add_argument("--drop-path", type=float, default=0.0, help="DropPath rate during training.")
    parser.add_argument("--smooth", type=float, default=0.0, help="Temporal smoothing weight.")
    parser.add_argument("--smooth-mode", choices=["l2", "huber", "none"], default="l2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = Random(args.seed)
    head = SpikingSeqHead(
        dw_kernel=3,
        use_attention=True,
        heads=2,
        attn_type=args.attn,
        drop_path=args.drop_path,
        smooth_lambda=args.smooth,
        smooth_mode=args.smooth_mode,
        seed=args.seed + 42,
    )
    head.train()
    max_abs = 0.0
    for step_idx in range(args.steps):
        seq_input: SequenceTensor = []
        for _ in range(3):
            step: List[List[float]] = []
            for _ in range(8):
                step.append([rng.uniform(-1.0, 1.0) for _ in range(4)])
            seq_input.append(step)
        output = head.forward(seq_input)
        for frame in output:
            for column in frame:
                for value in column:
                    max_abs = max(max_abs, abs(value))
        if math.isnan(max_abs):
            raise RuntimeError("Numerical instability detected.")
    print(
        f"Demo complete (attn={args.attn}, drop_path={args.drop_path}, smooth={args.smooth}): "
        f"max|value|={max_abs:.4f}, last_reg={head.regularization():.6f}"
    )
