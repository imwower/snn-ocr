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
        elif gate == "laplace":
            energy = 0.0
            count = 0
            for t in range(T):
                for h in range(H):
                    current = features[t][h][w]
                    if w > 0:
                        left = features[t][h][w - 1]
                        for c in range(C):
                            energy += abs(current[c] - left[c])
                            count += 1
                    if w + 1 < W:
                        right = features[t][h][w + 1]
                        for c in range(C):
                            energy += abs(current[c] - right[c])
                            count += 1
            info_values.append(energy / (count or 1))
        else:
            raise ValueError("gate must be 'var', 'entropy', or 'laplace'")
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


def _segments_to_sequence(
    reduced: SequenceTensor,
    segments: List[Tuple[int, int]],
) -> SequenceTensor:
    if not reduced:
        return []
    sequence: SequenceTensor = []
    C = len(reduced[0][0]) if reduced[0] else 0
    for t in range(len(reduced)):
        step: List[List[float]] = []
        for start, end in segments:
            count = max(1, end - start + 1)
            accum = [0.0 for _ in range(C)]
            for w in range(start, end + 1):
                column = reduced[t][w]
                for c in range(C):
                    accum[c] += column[c]
            step.append([value / count for value in accum])
        sequence.append(step)
    return sequence


def _build_merge_log(
    info_values: Sequence[float],
    segments: List[Tuple[int, int]],
    column_map: List[int],
    curve: List[Dict[str, float]] | None,
) -> List[Dict[str, float | int | List[int] | List[Dict[str, float]]]]:
    merge_log: List[Dict[str, float | int | List[int] | List[Dict[str, float]]]] = []
    for seg_idx, (start, end) in enumerate(segments):
        cols = list(range(start, end + 1))
        info_slice = info_values[start : end + 1]
        info_mean = sum(info_slice) / (len(info_slice) or 1)
        merge_log.append(
            {
                "segment": seg_idx,
                "start": start,
                "end": end,
                "count": len(cols),
                "info_mean": info_mean,
                "columns": cols,
            }
        )
    if merge_log:
        merge_log[0]["column_map"] = column_map
        if curve:
            merge_log[0]["curve"] = curve
    return merge_log


def _segments_to_column_map(
    segments: Sequence[Tuple[int, int]],
    width: int,
) -> List[int]:
    column_map = [0 for _ in range(width)]
    for seg_idx, (start, end) in enumerate(segments):
        for w in range(start, end + 1):
            column_map[w] = seg_idx
    return column_map


def _greedy_segments(
    info_values: Sequence[float],
    max_merge: int,
) -> List[Tuple[int, int]]:
    if max_merge <= 0:
        raise ValueError("max_merge must be positive")
    segments: List[Tuple[int, int]] = []
    window = max(1, max_merge * 2)
    for w in range(len(info_values)):
        start = max(0, w - window + 1)
        local_slice = info_values[start : w + 1]
        local_mean = sum(local_slice) / len(local_slice)
        if segments:
            seg_start, seg_end = segments[-1]
            seg_len = seg_end - seg_start + 1
            if info_values[w] <= local_mean and seg_len < max_merge:
                segments[-1] = (seg_start, w)
                continue
        segments.append((w, w))
    return segments


def _dp_optimal_segments(
    info_values: Sequence[float],
    target_width: int,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, float]]]:
    W = len(info_values)
    if target_width <= 0 or target_width > W:
        raise ValueError("target_width must be in [1, W]")
    prefix = [0.0]
    for value in info_values:
        prefix.append(prefix[-1] + value)
    dp = [[math.inf for _ in range(W + 1)] for _ in range(target_width + 1)]
    choice = [[-1 for _ in range(W + 1)] for _ in range(target_width + 1)]
    dp[0][0] = 0.0
    curve: List[Dict[str, float]] = []
    for k in range(1, target_width + 1):
        best_val = math.inf
        best_idx = -1
        min_w = k
        max_w = W - (target_width - k)
        for w in range(min_w, max_w + 1):
            prev_idx = w - 1
            prev_cost = dp[k - 1][prev_idx]
            if prev_cost < math.inf:
                candidate = prev_cost - prefix[prev_idx]
                if candidate < best_val:
                    best_val = candidate
                    best_idx = prev_idx
            if best_idx == -1:
                continue
            dp[k][w] = prefix[w] + best_val
            choice[k][w] = best_idx
        total_cost = dp[k][W]
        if not math.isinf(total_cost):
            curve.append(
                {
                    "segments": float(k),
                    "ratio": float(k) / float(W or 1),
                    "cost": total_cost,
                }
            )
    if math.isinf(dp[target_width][W]):
        raise RuntimeError("Unable to construct optimal segments with the given budget")
    segments: List[Tuple[int, int]] = []
    w = W
    for k in range(target_width, 0, -1):
        prev = choice[k][w]
        if prev < 0:
            raise RuntimeError("Invalid DP backtrack state")
        segments.append((prev, w - 1))
        w = prev
    segments.reverse()
    return segments, curve


def compress_height(
    features: FeatureTensor,
    target_h: int = 1,
    gate: str = "var",
    target_width: int | None = None,
    max_merge: int = 4,
    strategy: str = "auto",
) -> Tuple[SequenceTensor, Tuple[int, int, int], List[Dict[str, float | int | List[int]]]]:
    """Compress features to (T, W', C) by collapsing height and merging columns."""
    if strategy not in ("auto", "dp", "threshold"):
        raise ValueError("strategy must be 'auto', 'dp', or 'threshold'")
    reduced, original_width = _reduce_height(features, target_h=target_h)
    info_values = _column_information(features, gate=gate)
    use_dp = target_width is not None and strategy in ("auto", "dp")
    segments: List[Tuple[int, int]]
    curve: List[Dict[str, float]] | None = None
    if use_dp:
        segments, curve = _dp_optimal_segments(info_values, int(target_width))
        if len(segments) != int(target_width):
            raise RuntimeError("DP segmentation produced inconsistent width")
    else:
        if max_merge <= 0:
            raise ValueError("max_merge must be positive")
        segments = _greedy_segments(info_values, max_merge=max_merge)
    sequence = _segments_to_sequence(reduced, segments)
    column_map = _segments_to_column_map(segments, original_width)
    merge_log = _build_merge_log(info_values, segments, column_map, curve)
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


def column_information(features: FeatureTensor, gate: str = "var") -> List[float]:
    """Expose the per-column information scores used by the OTC compressor."""
    return _column_information(features, gate=gate)


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
    ascii_map = ascii_heatmap(info_values)
    target = max(1, W // 2)
    seq_dp, shape_dp, log_dp = compress_height(
        tensor,
        target_h=1,
        gate="var",
        target_width=target,
        strategy="dp",
    )
    seq_greedy, shape_greedy, log_greedy = compress_height(
        tensor,
        target_h=1,
        gate="var",
        max_merge=2,
        strategy="threshold",
    )
    print("Column information heatmap:")
    print(ascii_map)
    print(f"DP target W'={target}: {W}->{shape_dp[1]} tokens")
    if log_dp:
        curve = log_dp[0].get("curve", [])
        if curve:
            print("Compression curve (segments, ratio, cost):")
            for node in curve:
                print(
                    f"  k={int(node['segments'])} ratio={node['ratio']:.2f} "
                    f"cost={node['cost']:.4f}"
                )
    print("DP merge log:")
    for entry in log_dp:
        print(
            f"  seg {entry['segment']} cols {entry['start']}..{entry['end']} "
            f"count={entry['count']} info≈{entry['info_mean']:.4f}"
        )
    print(f"Greedy baseline: {W}->{shape_greedy[1]} tokens")
    for entry in log_greedy:
        print(
            f"  seg {entry['segment']} cols {entry['start']}..{entry['end']} "
            f"count={entry['count']} info≈{entry['info_mean']:.4f}"
        )
