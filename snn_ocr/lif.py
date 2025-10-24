"""Leaky integrate-and-fire utilities, pure Python convolutions, and residual blocks."""
from __future__ import annotations

from random import Random
from typing import List, Sequence, Tuple

Channel = List[List[float]]
FeatureMap = List[Channel]


class LIF:
    """Leaky integrate-and-fire neuron with soft reset."""

    def __init__(self, alpha: float, v_th: float, v_reset: float = 0.0) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = alpha
        self.v_th = v_th
        self.v_reset = v_reset
        self.v = 0.0

    def step(self, i_t: float) -> Tuple[float, int]:
        """Advance one step given input current i_t."""
        v_t = self.alpha * self.v + i_t
        if v_t >= self.v_th:
            r_t = 1
            v_t = v_t - self.v_th + self.v_reset
        else:
            r_t = 0
        self.v = v_t
        return v_t, r_t


def _pad_channel(channel: Channel, pad_h: int, pad_w: int) -> Channel:
    height = len(channel)
    width = len(channel[0]) if height else 0
    padded: Channel = []
    zero_row = [0.0] * (width + 2 * pad_w)
    for _ in range(pad_h):
        padded.append(zero_row.copy())
    for row in channel:
        padded.append([0.0] * pad_w + row + [0.0] * pad_w)
    for _ in range(pad_h):
        padded.append(zero_row.copy())
    return padded


def conv2d_spike(
    inputs: FeatureMap,
    kernels: FeatureMap,
    bias: Sequence[float] | None = None,
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
) -> FeatureMap:
    """Apply a multi-channel 2D convolution using pure Python lists."""
    if not inputs:
        raise ValueError("inputs must be non-empty")
    in_channels = len(inputs)
    kernel_out = len(kernels)
    if kernel_out == 0:
        raise ValueError("kernels must be non-empty")
    kh = len(kernels[0][0])
    kw = len(kernels[0][0][0])
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("stride must be positive")
    padded_inputs = [_pad_channel(channel, pad_h, pad_w) for channel in inputs]
    padded_height = len(padded_inputs[0])
    padded_width = len(padded_inputs[0][0])
    out_height = (padded_height - kh) // stride_h + 1
    out_width = (padded_width - kw) // stride_w + 1
    result: FeatureMap = []
    for out_c in range(kernel_out):
        kernel_set = kernels[out_c]
        if len(kernel_set) != in_channels:
            raise ValueError("kernel channel mismatch")
        bias_value = bias[out_c] if bias else 0.0
        channel_out: Channel = []
        for oy in range(out_height):
            row_out: List[float] = []
            for ox in range(out_width):
                acc = bias_value
                for in_c in range(in_channels):
                    kernel = kernel_set[in_c]
                    start_y = oy * stride_h
                    start_x = ox * stride_w
                    for ky in range(kh):
                        for kx in range(kw):
                            acc += padded_inputs[in_c][start_y + ky][start_x + kx] * kernel[ky][kx]
                row_out.append(acc)
            channel_out.append(row_out)
        result.append(channel_out)
    return result


def _apply_lif_grid(channel: Channel, lif_params: Tuple[float, float, float]) -> Tuple[Channel, Channel]:
    alpha, v_th, v_reset = lif_params
    potentials: Channel = []
    spikes: Channel = []
    for row in channel:
        pot_row: List[float] = []
        spike_row: List[float] = []
        for current in row:
            neuron = LIF(alpha=alpha, v_th=v_th, v_reset=v_reset)
            v_t, r_t = neuron.step(current)
            pot_row.append(v_t)
            spike_row.append(float(r_t))
        potentials.append(pot_row)
        spikes.append(spike_row)
    return potentials, spikes


def residual_block(
    inputs: FeatureMap,
    kernels1: FeatureMap,
    kernels2: FeatureMap,
    bias1: Sequence[float] | None = None,
    bias2: Sequence[float] | None = None,
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    lif_params: Tuple[float, float, float] = (0.9, 1.0, 0.0),
) -> Tuple[FeatureMap, FeatureMap]:
    """Two-layer residual block with LIF activations."""
    conv1 = conv2d_spike(inputs, kernels1, bias=bias1, stride=stride, padding=padding)
    spikes1: FeatureMap = []
    for channel in conv1:
        _, spike_map = _apply_lif_grid(channel, lif_params)
        spikes1.append(spike_map)
    conv2 = conv2d_spike(spikes1, kernels2, bias=bias2, stride=stride, padding=padding)
    spikes2: FeatureMap = []
    for channel in conv2:
        _, spike_map = _apply_lif_grid(channel, lif_params)
        spikes2.append(spike_map)
    if len(spikes2) != len(inputs):
        raise ValueError("Residual requires equal channel counts")
    residual: FeatureMap = []
    for c in range(len(spikes2)):
        channel_out: Channel = []
        for y in range(len(spikes2[c])):
            row_out: List[float] = []
            for x in range(len(spikes2[c][y])):
                skip = inputs[c][y][x] if y < len(inputs[c]) and x < len(inputs[c][y]) else 0.0
                combined = min(1.0, skip + spikes2[c][y][x])
                row_out.append(combined)
            channel_out.append(row_out)
        residual.append(channel_out)
    return residual, spikes2


def temporal_weighting(loss_per_t: Sequence[float], mode: str = "uniform") -> float:
    """Temporal Efficient Training weighting."""
    if not loss_per_t:
        raise ValueError("loss_per_t must be non-empty")
    if mode == "uniform":
        weight = 1.0 / len(loss_per_t)
        weights = [weight] * len(loss_per_t)
    elif mode == "tail":
        coeffs = [idx + 1 for idx in range(len(loss_per_t))]
        total = sum(coeffs)
        weights = [coeff / total for coeff in coeffs]
    else:
        raise ValueError("Unsupported mode")
    assert abs(sum(weights) - 1.0) < 1e-6
    return sum(w * loss for w, loss in zip(weights, loss_per_t))


def _self_check() -> None:
    neuron = LIF(alpha=0.8, v_th=1.0)
    v0, r0 = neuron.step(0.5)
    assert r0 == 0 and v0 > 0.0
    v1, r1 = neuron.step(1.0)
    assert r1 == 1 and v1 <= neuron.v_th
    losses = [0.1, 0.2, 0.3]
    assert abs(temporal_weighting(losses, "uniform") - sum(losses) / 3) < 1e-8


if __name__ == "__main__":
    _self_check()
    rng = Random(42)
    input_channels: FeatureMap = []
    for _ in range(2):
        channel: Channel = []
        for _ in range(4):
            row = [1.0 if rng.random() > 0.5 else 0.0 for _ in range(4)]
            channel.append(row)
        input_channels.append(channel)
    kernels1: FeatureMap = [
        [
            [[0.2, 0.1], [0.1, 0.2]],
            [[-0.1, 0.0], [0.0, -0.1]],
        ],
        [
            [[0.05, -0.05], [-0.05, 0.05]],
            [[0.1, 0.1], [0.1, 0.1]],
        ],
    ]
    kernels2: FeatureMap = [
        [
            [[0.1, 0.1], [0.1, 0.1]],
            [[0.05, -0.05], [0.05, -0.05]],
        ],
        [
            [[-0.1, 0.05], [0.05, -0.1]],
            [[0.2, 0.0], [0.0, 0.2]],
        ],
    ]
    residual_out, spikes_out = residual_block(
        input_channels,
        kernels1,
        kernels2,
        stride=(1, 1),
        padding=(0, 0),
        lif_params=(0.9, 0.6, 0.0),
    )
    total_spikes = sum(sum(sum(row) for row in channel) for channel in spikes_out)
    area = sum(len(channel) * len(channel[0]) for channel in spikes_out)
    firing_rate = total_spikes / area if area else 0.0
    print(f"Residual firing rate: {firing_rate:.3f}")
