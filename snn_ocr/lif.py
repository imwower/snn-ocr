"""Leaky integrate-and-fire utilities, spike convolutions, and residual blocks."""
from __future__ import annotations

import math
from random import Random
from typing import List, Sequence, Tuple

Channel = List[List[float]]
FeatureMap = List[Channel]
SequenceChannel = List[float]
SequenceMap = List[SequenceChannel]
Kernel2D = List[List[List[List[float]]]]  # out_c x in_c x kh x kw
Kernel1D = List[List[List[float]]]  # out_c x in_c x k


def surrogate_grad(x: float, width: float = 1.0) -> float:
    """Piecewise-linear triangle surrogate for dH/dx around zero."""
    if width <= 0.0:
        raise ValueError("width must be > 0")
    scaled = abs(x) / width
    if scaled >= 1.0:
        return 0.0
    # Triangle function (peak at 0, zero at +-width)
    return (1.0 - scaled) / width


class LIF:
    """Leaky integrate-and-fire neuron with soft reset and surrogate gradient."""

    def __init__(
        self,
        alpha: float,
        v_th: float,
        v_reset: float = 0.0,
        surrogate_width: float = 1.0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if surrogate_width <= 0.0:
            raise ValueError("surrogate_width must be positive")
        self.alpha = alpha
        self.v_th = v_th
        self.v_reset = v_reset
        self.v = v_reset
        self.surrogate_width = surrogate_width
        self._last_u = 0.0

    def reset(self) -> None:
        self.v = self.v_reset
        self._last_u = 0.0

    def step(self, i_t: float) -> Tuple[float, int]:
        """Advance one step given input current i_t."""
        u_t = self.alpha * self.v + i_t
        spike = int(u_t >= self.v_th)
        self._last_u = u_t
        if spike:
            self.v = u_t - self.v_th + self.v_reset
        else:
            self.v = u_t
        return self.v, spike

    def grad(self) -> float:
        """Surrogate gradient evaluated at last membrane potential."""
        return surrogate_grad(self._last_u - self.v_th, self.surrogate_width)


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


def _pad_sequence(channel: SequenceChannel, pad: int) -> SequenceChannel:
    return [0.0] * pad + channel + [0.0] * pad


def _linear_conv2d(
    inputs: FeatureMap,
    kernels: Kernel2D,
    bias: Sequence[float] | None,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> FeatureMap:
    if not inputs:
        raise ValueError("inputs must be non-empty")
    height = len(inputs[0])
    width = len(inputs[0][0]) if height else 0
    for channel in inputs:
        if len(channel) != height or (width and len(channel[0]) != width):
            raise ValueError("input spatial dimensions must match")
    out_channels = len(kernels)
    if out_channels == 0:
        raise ValueError("kernels must be non-empty")
    if not kernels[0] or not kernels[0][0] or not kernels[0][0][0]:
        raise ValueError("kernel dimensions must be non-zero")
    in_channels = len(inputs)
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("stride must be positive")
    kh = len(kernels[0][0])
    kw = len(kernels[0][0][0])
    padded_inputs = [_pad_channel(channel, pad_h, pad_w) for channel in inputs]
    padded_height = len(padded_inputs[0])
    padded_width = len(padded_inputs[0][0])
    out_height = (padded_height - kh) // stride_h + 1
    out_width = (padded_width - kw) // stride_w + 1
    result: FeatureMap = []
    for out_c in range(out_channels):
        kernel_set = kernels[out_c]
        if len(kernel_set) != in_channels:
            raise ValueError("kernel channel mismatch")
        bias_value = bias[out_c] if bias is not None else 0.0
        channel_out: Channel = []
        for oy in range(out_height):
            row_out: List[float] = []
            for ox in range(out_width):
                acc = bias_value
                start_y = oy * stride_h
                start_x = ox * stride_w
                for in_c in range(in_channels):
                    kernel = kernel_set[in_c]
                    for ky in range(kh):
                        for kx in range(kw):
                            acc += (
                                padded_inputs[in_c][start_y + ky][start_x + kx] * kernel[ky][kx]
                            )
                row_out.append(acc)
            channel_out.append(row_out)
        result.append(channel_out)
    return result


def _linear_conv1d(
    inputs: SequenceMap,
    kernels: Kernel1D,
    bias: Sequence[float] | None,
    stride: int,
    padding: int,
) -> SequenceMap:
    if not inputs:
        raise ValueError("inputs must be non-empty")
    length = len(inputs[0])
    for channel in inputs:
        if len(channel) != length:
            raise ValueError("All input channels must share the same length")
    out_channels = len(kernels)
    if out_channels == 0:
        raise ValueError("kernels must be non-empty")
    if not kernels[0] or not kernels[0][0]:
        raise ValueError("kernel dimensions must be non-zero")
    in_channels = len(inputs)
    if stride <= 0:
        raise ValueError("stride must be positive")
    k = len(kernels[0][0])
    padded_inputs = [_pad_sequence(channel, padding) for channel in inputs]
    padded_len = len(padded_inputs[0])
    out_len = (padded_len - k) // stride + 1
    result: SequenceMap = []
    for out_c in range(out_channels):
        kernel_set = kernels[out_c]
        if len(kernel_set) != in_channels:
            raise ValueError("kernel channel mismatch")
        bias_value = bias[out_c] if bias is not None else 0.0
        channel_out: SequenceChannel = []
        for ox in range(out_len):
            start_x = ox * stride
            acc = bias_value
            for in_c in range(in_channels):
                kernel = kernel_set[in_c]
                for kx in range(k):
                    acc += padded_inputs[in_c][start_x + kx] * kernel[kx]
            channel_out.append(acc)
        result.append(channel_out)
    return result


def _apply_lif_grid(
    feature_map: FeatureMap, lif_params: Tuple[float, float, float]
) -> Tuple[FeatureMap, FeatureMap]:
    alpha, v_th, v_reset = lif_params
    potentials: FeatureMap = []
    spikes: FeatureMap = []
    for channel in feature_map:
        pot_channel: Channel = []
        spike_channel: Channel = []
        for row in channel:
            pot_row: List[float] = []
            spike_row: List[float] = []
            for current in row:
                neuron = LIF(alpha=alpha, v_th=v_th, v_reset=v_reset)
                v_t, r_t = neuron.step(current)
                pot_row.append(v_t)
                spike_row.append(float(r_t))
            pot_channel.append(pot_row)
            spike_channel.append(spike_row)
        potentials.append(pot_channel)
        spikes.append(spike_channel)
    return potentials, spikes


def _apply_lif_sequence(
    sequence_map: SequenceMap, lif_params: Tuple[float, float, float]
) -> Tuple[SequenceMap, SequenceMap]:
    alpha, v_th, v_reset = lif_params
    potentials: SequenceMap = []
    spikes: SequenceMap = []
    for channel in sequence_map:
        pot_channel: SequenceChannel = []
        spike_channel: SequenceChannel = []
        neuron = LIF(alpha=alpha, v_th=v_th, v_reset=v_reset)
        neuron.reset()
        for current in channel:
            v_t, r_t = neuron.step(current)
            pot_channel.append(v_t)
            spike_channel.append(float(r_t))
        potentials.append(pot_channel)
        spikes.append(spike_channel)
    return potentials, spikes


def conv2d_spike(
    inputs: FeatureMap,
    kernels: Kernel2D,
    bias: Sequence[float] | None = None,
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    lif_params: Tuple[float, float, float] = (0.9, 1.0, 0.0),
) -> Tuple[FeatureMap, FeatureMap]:
    """Apply a multi-channel 2D convolution followed by element-wise LIF."""
    linear = _linear_conv2d(inputs, kernels, bias, stride, padding)
    return _apply_lif_grid(linear, lif_params)


def conv1d_spike(
    inputs: SequenceMap,
    kernels: Kernel1D,
    bias: Sequence[float] | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
    lif_params: Tuple[float, float, float] = (0.9, 1.0, 0.0),
) -> Tuple[SequenceMap, SequenceMap]:
    """1D spike convolution (channel-first) with shared LIF parameters."""
    linear = _linear_conv1d(inputs, kernels, bias, stride, padding)
    return _apply_lif_sequence(linear, lif_params)


def residual_block(
    inputs: FeatureMap,
    kernels1: Kernel2D,
    kernels2: Kernel2D,
    bias1: Sequence[float] | None = None,
    bias2: Sequence[float] | None = None,
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    lif_params: Tuple[float, float, float] = (0.9, 1.0, 0.0),
) -> Tuple[FeatureMap, FeatureMap]:
    """Two-layer residual block with spike activations."""
    pot1, spikes1 = conv2d_spike(
        inputs,
        kernels1,
        bias=bias1,
        stride=stride,
        padding=padding,
        lif_params=lif_params,
    )
    pot2, spikes2 = conv2d_spike(
        spikes1,
        kernels2,
        bias=bias2,
        stride=stride,
        padding=padding,
        lif_params=lif_params,
    )
    if len(spikes2) != len(inputs):
        raise ValueError("Residual requires equal channel counts")
    residual: FeatureMap = []
    for c in range(len(spikes2)):
        if len(spikes2[c]) != len(inputs[c]):
            raise ValueError("Residual requires equal height")
        channel_out: Channel = []
        for y in range(len(spikes2[c])):
            if len(spikes2[c][y]) != len(inputs[c][y]):
                raise ValueError("Residual requires equal width")
            row_out: List[float] = []
            for x in range(len(spikes2[c][y])):
                combined = min(1.0, max(0.0, inputs[c][y][x] + spikes2[c][y][x]))
                row_out.append(combined)
            channel_out.append(row_out)
        residual.append(channel_out)
    return residual, spikes2


def temporal_weighting(loss_per_t: Sequence[float], mode: str = "uniform") -> float:
    """Temporal Efficient Training weighting."""
    if not loss_per_t:
        raise ValueError("loss_per_t must be non-empty")
    total_steps = len(loss_per_t)
    if mode == "uniform":
        weights = [1.0 / total_steps for _ in loss_per_t]
    elif mode == "tail":
        coeffs = [idx + 1 for idx in range(total_steps)]
        total = sum(coeffs)
        weights = [coeff / total for coeff in coeffs]
    else:
        raise ValueError("Unsupported mode")
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError("TET weights must sum to 1.0")
    return sum(w * loss for w, loss in zip(weights, loss_per_t))


def _self_check() -> None:
    neuron = LIF(alpha=0.8, v_th=1.0, v_reset=0.0)
    v0, r0 = neuron.step(0.5)
    assert r0 == 0 and v0 > 0.0
    v1, r1 = neuron.step(1.0)
    assert r1 == 1 and v1 <= neuron.v_th
    losses = [0.1, 0.2, 0.3]
    assert abs(temporal_weighting(losses, "uniform") - sum(losses) / 3) < 1e-8
    assert surrogate_grad(0.0) > 0.0


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
    kernels1: Kernel2D = [
        [
            [
                [0.05, 0.02, -0.01],
                [0.02, 0.06, 0.02],
                [-0.01, 0.02, 0.05],
            ],
            [
                [0.01, -0.02, 0.01],
                [-0.02, 0.04, -0.02],
                [0.01, -0.02, 0.01],
            ],
        ],
        [
            [
                [0.02, -0.01, 0.02],
                [-0.01, 0.05, -0.01],
                [0.02, -0.01, 0.02],
            ],
            [
                [0.03, 0.03, 0.03],
                [0.0, 0.03, 0.0],
                [-0.03, -0.03, -0.03],
            ],
        ],
    ]
    kernels2: Kernel2D = [
        [
            [
                [0.04, 0.01, 0.04],
                [0.01, 0.05, 0.01],
                [0.04, 0.01, 0.04],
            ],
            [
                [-0.02, 0.0, 0.02],
                [0.0, -0.01, 0.0],
                [0.02, 0.0, -0.02],
            ],
        ],
        [
            [
                [-0.05, 0.02, -0.05],
                [0.02, 0.05, 0.02],
                [-0.05, 0.02, -0.05],
            ],
            [
                [0.03, 0.0, 0.03],
                [0.0, 0.04, 0.0],
                [0.03, 0.0, 0.03],
            ],
        ],
    ]
    block1, spikes1 = residual_block(
        input_channels,
        kernels1,
        kernels2,
        stride=(1, 1),
        padding=(1, 1),
        lif_params=(0.9, 0.6, 0.0),
    )
    block2, spikes2 = residual_block(
        block1,
        kernels1,
        kernels2,
        stride=(1, 1),
        padding=(1, 1),
        lif_params=(0.9, 0.6, 0.0),
    )

    def _firing_rate(spikes: FeatureMap) -> float:
        total_spikes = sum(sum(sum(row) for row in channel) for channel in spikes)
        support = sum(len(channel) * len(channel[0]) for channel in spikes)
        return total_spikes / support if support else 0.0

    print(f"Block1 firing rate: {_firing_rate(spikes1):.3f}")
    print(f"Block2 firing rate: {_firing_rate(spikes2):.3f}")
