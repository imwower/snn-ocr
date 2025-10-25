"""Optical Token Compressor: collapse features along height and merge low-information columns."""
from __future__ import annotations

import math
from random import Random
from typing import Dict, List, Sequence, Tuple

FeatureColumn = List[float]
FeatureStep = List[FeatureColumn]
FeatureTensor = List[List[List[List[float]]]]  # T x H x W x C
SequenceTensor = List[List[List[float]]]  # T x W' x C


def _validate_features(features: FeatureTensor) -> Tuple[int, int, int, int]:
    if not features:
        raise ValueError("features must be non-empty")
    T = len(features)
    H = len(features[0])
    W = len(features[0][0])
    C = len(features[0][0][0])
    for t in range(T):
        if len(features[t]) != H:
            raise ValueError("Inconsistent height across time steps")
        for h in range(H):
            if len(features[t][h]) != W:
                raise ValueError("Inconsistent width across rows")
            for w in range(W):
                if len(features[t][h][w]) != C:
                    raise ValueError("Inconsistent channel dimension")
    return T, H, W, C


def _column_information(features: FeatureTensor, gate: str) -> List[float]:
    T, H, W, C = _validate_features(features)
    info_values: List[float] = []
    eps = 1e-8
    for w in range(W):
        values: List[float] = []
        for t in range(T):
            for h in range(H):
                values.extend(features[t][h][w])
        if gate == "var":
            mean = sum(values) / (len(values) or 1)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) or 1)
            info_values.append(variance)
        elif gate == "entropy":
            pos = [abs(v) for v in values]
            total = sum(pos)
            if total <= eps:
                info_values.append(0.0)
            else:
                probs = [p / total for p in pos]
                entropy = -sum(p * math.log(p + eps) for p in probs)
                info_values.append(entropy)
        else:
            raise ValueError("gate must be 'var' or 'entropy'")
    return info_values


def _reduce_height(features: FeatureTensor, target_h: int) -> Tuple[SequenceTensor, int]:
    T, H, W, C = _validate_features(features)
    if target_h <= 0:
        raise ValueError("target_h must be positive")
    reduced: SequenceTensor = []
    for t in range(T):
        step: List[List[float]] = []
        for w in range(W):
            column_vec = [0.0 for _ in range(C)]
            count = 0
            for h in range(H):
                row = features[t][h][w]
                for c in range(C):
                    column_vec[c] += row[c]
                count += 1
            if count > 0:
                column_vec = [value / count for value in column_vec]
            step.append(column_vec)
        reduced.append(step)
    return reduced, W


def compress_height(
    features: FeatureTensor,
    target_h: int = 1,
    gate: str = "var",
    max_merge: int = 4,
) -> Tuple[SequenceTensor, Tuple[int, int, int], List[Dict[str, float | int]]]:
    """Compress features to (T, W', C) by collapsing height and merging columns.

    Returns a tuple of (sequence, (T, W_prime, C), merge_log) where merge_log
    contains records with start/end indices and the number of merged columns.
    """
    if max_merge <= 0:
        raise ValueError("max_merge must be positive")
    reduced, original_width = _reduce_height(features, target_h=target_h)
    info_values = _column_information(features, gate=gate)
    mean_info = sum(info_values) / (len(info_values) or 1)
    groups: List[List[List[float]]] = []
    counts: List[int] = []
    columns: List[List[int]] = []
    for w in range(original_width):
        column_vectors = [reduced[t][w][:] for t in range(len(reduced))]
        if (
            groups
            and info_values[w] <= mean_info
            and counts[-1] < max_merge
        ):
            prev_count = counts[-1]
            previous = groups[-1]
            new_count = prev_count + 1
            for t in range(len(previous)):
                for c in range(len(previous[t])):
                    previous[t][c] = (previous[t][c] * prev_count + column_vectors[t][c]) / new_count
            counts[-1] = new_count
            columns[-1].append(w)
        else:
            groups.append(column_vectors)
            counts.append(1)
            columns.append([w])
    sequence: SequenceTensor = []
    for t in range(len(reduced)):
        step: List[List[float]] = []
        for group in groups:
            step.append(group[t][:])
        sequence.append(step)
    merge_log: List[Dict[str, float | int]] = []
    for idx, col_group in enumerate(columns):
        if not col_group:
            continue
        info_mean = sum(info_values[col] for col in col_group) / len(col_group)
        merge_log.append(
            {
                "start": col_group[0],
                "end": col_group[-1],
                "count": len(col_group),
                "info_mean": info_mean,
            }
        )
    T, _, _, C = _validate_features(features)
    W_prime = len(sequence[0]) if sequence else 0
    shape = (T, W_prime, C)
    return sequence, shape, merge_log


def ascii_heatmap(values: Sequence[float]) -> str:
    """Render values as ASCII heat using a small gradient."""
    if not values:
        return ""
    chars = " .:-=+*#%@"
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max, rel_tol=1e-9, abs_tol=1e-12):
        return chars[-1] * len(values)
    output_chars: List[str] = []
    for value in values:
        norm = (value - v_min) / (v_max - v_min)
        idx = min(len(chars) - 1, max(0, int(round(norm * (len(chars) - 1)))))
        output_chars.append(chars[idx])
    return "".join(output_chars)


def _self_check() -> None:
    rng = Random(0)
    features: FeatureTensor = []
    for t in range(3):
        time_slice: List[List[List[float]]] = []
        for h in range(2):
            row: List[List[float]] = []
            for w in range(5):
                row.append([rng.random(), rng.random()])
            time_slice.append(row)
        features.append(time_slice)
    reduced, shape, log = compress_height(features, target_h=1, gate="var", max_merge=2)
    assert len(reduced) == 3
    assert len(reduced[0]) <= 5
    info = _column_information(features, gate="var")
    assert len(info) == 5
    assert shape[1] == len(reduced[0])
    assert len(log) == len(reduced[0])


if __name__ == "__main__":
    _self_check()
    rng = Random(13)
    T, H, W, C = 4, 3, 8, 2
    tensor: FeatureTensor = []
    for t in range(T):
        time_slice: List[List[List[float]]] = []
        for h in range(H):
            row: List[List[float]] = []
            for w in range(W):
                base = 1.0 if w % 3 == 0 else 0.2
                row.append([base + 0.1 * rng.random() for _ in range(C)])
            time_slice.append(row)
        tensor.append(time_slice)
    info_values = _column_information(tensor, gate="var")
    compressed, shape, log = compress_height(tensor, target_h=1, gate="var", max_merge=2)
    compression_ratio = (shape[1] / W) if W else 0.0
    print(f"W -> W': {W} -> {shape[1]} (ratio {compression_ratio:.2f})")
    print("Column information heatmap:")
    print(ascii_heatmap(info_values))
    print("Merge log:")
    for entry in log:
        print(
            f"  cols {entry['start']}..{entry['end']} (count={entry['count']}) "
            f"info≈{entry['info_mean']:.4f}"
        )
