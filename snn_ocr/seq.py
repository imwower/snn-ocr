"""Sequence head layers for SNN OCR: depthwise conv, pointwise mixing, optional attention."""
from __future__ import annotations

from random import Random
from typing import List, Sequence, Tuple

SequenceTensor = List[List[List[float]]]  # T x W x C


def _validate_seq(seq: SequenceTensor) -> Tuple[int, int, int]:
    if not seq:
        raise ValueError("sequence tensor must be non-empty")
    T = len(seq)
    W = len(seq[0])
    C = len(seq[0][0]) if W else 0
    for t in range(T):
        if len(seq[t]) != W:
            raise ValueError("Inconsistent width across time")
        for w in range(W):
            if len(seq[t][w]) != C:
                raise ValueError("Inconsistent channel count")
    return T, W, C


def dwconv1d_spike(seq: SequenceTensor, k: int = 3) -> SequenceTensor:
    """Depthwise separable 1D convolution across width dimension."""
    T, W, C = _validate_seq(seq)
    if k <= 0 or k % 2 == 0:
        raise ValueError("Kernel size k must be positive and odd")
    pad = k // 2
    result: SequenceTensor = []
    for t in range(T):
        step_out: List[List[float]] = []
        for w in range(W):
            column_out: List[float] = []
            for c in range(C):
                acc = 0.0
                count = 0
                for offset in range(-pad, pad + 1):
                    idx = w + offset
                    if 0 <= idx < W:
                        acc += seq[t][idx][c]
                        count += 1
                column_out.append(acc / count if count else 0.0)
            step_out.append(column_out)
        result.append(step_out)
    return result


def pointwise_spike(seq: SequenceTensor, out_channels: int) -> SequenceTensor:
    """Pointwise linear mixing across channels."""
    T, W, C = _validate_seq(seq)
    if out_channels <= 0:
        raise ValueError("out_channels must be positive")
    rng = Random(123)
    weights: List[List[float]] = [
        [rng.uniform(-0.1, 0.1) for _ in range(C)] for _ in range(out_channels)
    ]
    biases: List[float] = [0.0 for _ in range(out_channels)]
    result: SequenceTensor = []
    for t in range(T):
        step_out: List[List[float]] = []
        for w in range(W):
            column: List[float] = []
            for out_c in range(out_channels):
                acc = biases[out_c]
                for in_c in range(C):
                    acc += seq[t][w][in_c] * weights[out_c][in_c]
                column.append(acc)
            step_out.append(column)
        result.append(step_out)
    return result


def linear_attention(seq: SequenceTensor, heads: int = 2) -> SequenceTensor:
    """Simplified linear attention across width dimension."""
    T, W, C = _validate_seq(seq)
    if heads <= 0:
        raise ValueError("heads must be positive")
    head_dim = max(1, C // heads)
    rng = Random(321)
    weights_q = [
        [rng.uniform(-0.05, 0.05) for _ in range(head_dim)] for _ in range(heads)
    ]
    weights_k = [
        [rng.uniform(-0.05, 0.05) for _ in range(head_dim)] for _ in range(heads)
    ]
    weights_v = [
        [rng.uniform(-0.05, 0.05) for _ in range(head_dim)] for _ in range(heads)
    ]
    result: SequenceTensor = []
    for t in range(T):
        step_out: List[List[float]] = []
        for w in range(W):
            col_out = seq[t][w][:]  # residual connection
            for head in range(heads):
                start = head * head_dim
                end = min(C, start + head_dim)
                if start >= end:
                    continue
                q = sum(seq[t][w][c] * weights_q[head][c - start] for c in range(start, end))
                numerator = 0.0
                denominator = 1e-6
                for j in range(W):
                    k_val = sum(seq[t][j][c] * weights_k[head][c - start] for c in range(start, end))
                    v_val = sum(seq[t][j][c] * weights_v[head][c - start] for c in range(start, end))
                    kernel = max(0.0, q * k_val)
                    numerator += kernel * v_val
                    denominator += kernel
                attention = numerator / denominator
                for c in range(start, end):
                    col_out[c] += attention / (end - start)
            step_out.append(col_out)
        result.append(step_out)
    return result


def _self_check() -> None:
    sample = [[[0.1, 0.2, 0.3] for _ in range(5)] for _ in range(2)]
    smoothed = dwconv1d_spike(sample, k=3)
    mixed = pointwise_spike(smoothed, out_channels=4)
    attn = linear_attention(mixed, heads=2)
    assert len(attn) == len(sample)
    assert len(attn[0]) == len(sample[0])
    assert len(attn[0][0]) == 4


if __name__ == "__main__":
    _self_check()
    rng = Random(0)
    T, W, C = 3, 6, 4
    seq: SequenceTensor = []
    for _ in range(T):
        step: List[List[float]] = []
        for _ in range(W):
            step.append([rng.uniform(-1.0, 1.0) for _ in range(C)])
        seq.append(step)
    conv = dwconv1d_spike(seq, k=3)
    mixed = pointwise_spike(conv, out_channels=6)
    attn = linear_attention(mixed, heads=3)
    print(f"Final logits shape: T={len(attn)}, W={len(attn[0])}, C={len(attn[0][0])}")
