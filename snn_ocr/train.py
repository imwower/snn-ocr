"""Unified curriculum training script for the SNN-OCR prototype (standard library only)."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable, Dict, Generator, Iterable, List, Sequence, Tuple, TextIO

import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_ocr import ctc, synth, spikes


# ---------------------------------------------------------------------------
# Vocabulary & helpers
# ---------------------------------------------------------------------------

SYMBOLS: Tuple[str, ...] = ctc.symbol_table()
BLANK_INDEX = 0
CHAR_TO_INDEX: Dict[str, int] = {
    symbol: index for index, symbol in enumerate(SYMBOLS) if symbol
}
NUM_CLASSES = len(SYMBOLS)
VALID_CLASS_INDICES: Tuple[int, ...] = tuple(idx for idx, symbol in enumerate(SYMBOLS) if symbol)
KD_TEMPERATURE = 1.5


def text_to_indices(text: str) -> List[int]:
    indices: List[int] = []
    for ch in text:
        if ch not in CHAR_TO_INDEX:
            raise ValueError(f"Unsupported character: {ch!r}")
        indices.append(CHAR_TO_INDEX[ch])
    return indices


# ---------------------------------------------------------------------------
# Stage configuration & sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageConfig:
    name: str
    tet_mode: str
    loss_type: str  # "ce" or "ctc"
    smoothing_lambda: float
    replay_ratio: float
    previous: Tuple[str, ...]
    train_segments: Tuple[str, ...]


ENCODER_SEGMENTS: Tuple[str, ...] = ("encoder_front", "encoder_back")
SEQUENCE_SEGMENTS: Tuple[str, ...] = ("sequence_front", "sequence_back")


STAGE_CONFIGS: Dict[str, StageConfig] = {
    "S1": StageConfig(
        name="S1",
        tet_mode="uniform",
        loss_type="ce",
        smoothing_lambda=0.0,
        replay_ratio=0.0,
        previous=(),
        train_segments=ENCODER_SEGMENTS,
    ),
    "S2": StageConfig(
        name="S2",
        tet_mode="tail",
        loss_type="ce",
        smoothing_lambda=0.0,
        replay_ratio=0.1,
        previous=("S1",),
        train_segments=ENCODER_SEGMENTS,
    ),
    "S3": StageConfig(
        name="S3",
        tet_mode="tail",
        loss_type="ctc",
        smoothing_lambda=0.01,
        replay_ratio=0.15,
        previous=("S2",),
        train_segments=SEQUENCE_SEGMENTS,
    ),
    "S4": StageConfig(
        name="S4",
        tet_mode="tail",
        loss_type="ctc",
        smoothing_lambda=0.02,
        replay_ratio=0.2,
        previous=("S3",),
        train_segments=SEQUENCE_SEGMENTS,
    ),
}


class StageSampler:
    """Deterministic curriculum sampler built on synth.make_sample."""

    def __init__(self, stage: str) -> None:
        self.stage = stage.upper()
        self.counter = 0

    def next_sample(self) -> Tuple[List[List[int]], str]:
        self.counter += 1
        seed = self.counter
        gray, text = synth.make_sample(self.stage, seed=seed)
        return gray, text

    def state_dict(self) -> Dict[str, int]:
        return {"counter": self.counter}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.counter = int(state.get("counter", 0))


@dataclass
class TrainSample:
    stage: str
    gray: List[List[int]]
    text: str


class ReplayBuffer:
    """Circular buffer that stores a small cache of previous-stage samples."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self.capacity = capacity
        self._items: List[TrainSample] = []

    def add(self, sample: TrainSample) -> None:
        if len(self._items) >= self.capacity:
            self._items.pop(0)
        self._items.append(sample)

    def sample(self, rng: Random) -> TrainSample:
        if not self._items:
            raise ValueError("Replay buffer is empty")
        return rng.choice(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items


@dataclass
class LossBreakdown:
    total: float
    main: float
    smooth: float
    grad_logits: List[List[float]]
    kd: float = 0.0
    cer: float | None = None
    wer: float | None = None


# ---------------------------------------------------------------------------
# Replay & streaming loader
# ---------------------------------------------------------------------------


def mix_replay(stage_cfg: StageConfig, rng: Random, replay_ratio: float) -> str:
    if stage_cfg.previous and rng.random() < replay_ratio:
        return rng.choice(stage_cfg.previous)
    return stage_cfg.name


def build_loader(
    stage_cfg: StageConfig,
    samplers: Dict[str, StageSampler],
    *,
    steps: int,
    batch_size: int,
    rng: Random,
    replay_ratio: float,
    buffers: Dict[str, ReplayBuffer] | None = None,
) -> Generator[List[TrainSample], None, None]:
    def _generator() -> Generator[List[TrainSample], None, None]:
        for _ in range(steps):
            batch: List[TrainSample] = []
            for _ in range(batch_size):
                chosen_stage = mix_replay(stage_cfg, rng, replay_ratio)
                sampler = samplers[chosen_stage]
                if buffers and chosen_stage in buffers and not buffers[chosen_stage].is_empty():
                    batch.append(buffers[chosen_stage].sample(rng))
                    continue
                gray, text = sampler.next_sample()
                sample = TrainSample(stage=chosen_stage, gray=gray, text=text)
                batch.append(sample)
                if buffers and chosen_stage in buffers:
                    buffers[chosen_stage].add(sample)
            yield batch

    return _generator()


def prepare_replay_buffers(
    stage_cfg: StageConfig,
    samplers: Dict[str, StageSampler],
    batch_size: int,
    replay_ratio: float,
) -> Dict[str, ReplayBuffer]:
    if replay_ratio <= 0.0 or not stage_cfg.previous:
        return {}
    capacity = max(4, int(batch_size * max(0.05, replay_ratio) * 2))
    buffers: Dict[str, ReplayBuffer] = {}
    for prev in stage_cfg.previous:
        buffer = ReplayBuffer(capacity)
        for _ in range(capacity):
            gray, text = samplers[prev].next_sample()
            buffer.add(TrainSample(stage=prev, gray=gray, text=text))
        buffers[prev] = buffer
    return buffers


# ---------------------------------------------------------------------------
# Feature extraction utilities
# ---------------------------------------------------------------------------


def image_to_sequence(gray: List[List[int]]) -> List[List[float]]:
    """Convert a grayscale image into a sequence of column descriptors."""
    if not gray or not gray[0]:
        return []
    height = len(gray)
    width = len(gray[0])
    sequence: List[List[float]] = []
    for w in range(width):
        column = [gray[h][w] / 255.0 for h in range(height)]
        mean = sum(column) / height
        mid = max(1, height // 2)
        top = sum(column[:mid]) / mid
        bottom = sum(column[mid:]) / max(1, height - mid)
        variance = sum((value - mean) ** 2 for value in column) / height
        position = w / max(1, width - 1)
        sequence.append([mean, top, bottom, variance, position])
    return merge_low_information(sequence)


def merge_low_information(sequence: List[List[float]], max_merge: int = 4) -> List[List[float]]:
    """Collapse adjacent low-information columns to approximate OTC behaviour."""
    if not sequence:
        return []
    info_scores: List[float] = []
    for column in sequence:
        mean = sum(column[:-1]) / max(1, len(column) - 1)  # exclude position
        variance = sum((value - mean) ** 2 for value in column[:-1]) / max(1, len(column) - 1)
        info_scores.append(variance)
    threshold = sum(info_scores) / max(1, len(info_scores))
    groups: List[List[float]] = []
    counts: List[int] = []
    for idx, column in enumerate(sequence):
        if groups and info_scores[idx] <= threshold and counts[-1] < max_merge:
            prev_count = counts[-1]
            combined = groups[-1]
            new_count = prev_count + 1
            for i in range(len(combined)):
                combined[i] = (combined[i] * prev_count + column[i]) / new_count
            counts[-1] = new_count
        else:
            groups.append(column[:])
            counts.append(1)
    return groups


def classify_single_token(logits: Sequence[Sequence[float]]) -> str:
    if not logits:
        return ""
    accumulator = [0.0 for _ in range(NUM_CLASSES)]
    for step in logits:
        probs = softmax(step)
        for idx, value in enumerate(probs):
            accumulator[idx] += value
    if not VALID_CLASS_INDICES:
        return ""
    best = max(VALID_CLASS_INDICES, key=lambda idx: accumulator[idx])
    return SYMBOLS[best]


def decode_prediction(stage_name: str, logits: Sequence[Sequence[float]]) -> str:
    stage_name = stage_name.upper()
    cfg = STAGE_CONFIGS.get(stage_name)
    if cfg is None:
        return ""
    if cfg.loss_type == "ce":
        return classify_single_token(logits)
    return ctc.greedy_decode(logits, blank=BLANK_INDEX)


def tet_weights(length: int, mode: str) -> List[float]:
    if length <= 0:
        return []
    if mode == "uniform":
        return [1.0 / length for _ in range(length)]
    if mode == "tail":
        coeffs = [i + 1 for i in range(length)]
        total = sum(coeffs)
        return [value / total for value in coeffs]
    raise ValueError(f"Unsupported TET mode: {mode}")


# ---------------------------------------------------------------------------
# Curriculum model (encoder + sequence head)
# ---------------------------------------------------------------------------


Gradients = Dict[str, Dict[str, List[List[float]] | List[float]]]


class LinearLayer:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: str,
        seed: int,
        init: str = "random",
    ) -> None:
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("Layer dimensions must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        if activation not in {"tanh", "linear"}:
            raise ValueError("Unsupported activation")
        if init not in {"random", "zeros"}:
            raise ValueError("init must be 'random' or 'zeros'")
        rng = Random(seed)
        if init == "zeros":
            self.weights = [[0.0 for _ in range(in_dim)] for _ in range(out_dim)]
            self.bias = [0.0 for _ in range(out_dim)]
        else:
            scale = 0.1 / math.sqrt(in_dim)
            self.weights = [
                [rng.uniform(-scale, scale) for _ in range(in_dim)]
                for _ in range(out_dim)
            ]
            self.bias = [0.0 for _ in range(out_dim)]

    def forward(self, inputs: Sequence[float]) -> Tuple[List[float], Dict[str, List[float]]]:
        if len(inputs) != self.in_dim:
            raise ValueError("Input dimension mismatch for layer")
        outputs: List[float] = []
        for row, bias in zip(self.weights, self.bias):
            acc = bias
            for value, weight in zip(inputs, row):
                acc += value * weight
            outputs.append(acc)
        if self.activation == "tanh":
            activated = [math.tanh(value) for value in outputs]
        else:
            activated = outputs[:]
        cache = {"input": list(inputs), "output": activated}
        return activated, cache

    def state_dict(self) -> Dict[str, List[List[float]] | List[float]]:
        return {
            "weights": [[value for value in row] for row in self.weights],
            "bias": [value for value in self.bias],
            "activation": self.activation,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        weights = state.get("weights")
        bias = state.get("bias")
        if not isinstance(weights, list) or not isinstance(bias, list):
            raise ValueError("Invalid layer state")
        self.weights = [[float(value) for value in row] for row in weights]  # type: ignore[arg-type]
        self.bias = [float(value) for value in bias]  # type: ignore[arg-type]


class CurriculumModel:
    """Two-part encoder/sequence head with additive logits for curriculum stages."""

    def __init__(self, feature_dim: int, output_dim: int, seed: int = 0) -> None:
        self.segment_order: Tuple[Tuple[str, int, int, str, str], ...] = (
            ("encoder_front", feature_dim, 32, "tanh", "random"),
            ("encoder_back", 32, output_dim, "linear", "random"),
            ("sequence_front", 32, 32, "tanh", "zeros"),
            ("sequence_back", 32, output_dim, "linear", "zeros"),
        )
        self.layers: Dict[str, LinearLayer] = {}
        for idx, (name, in_dim, out_dim, act, init) in enumerate(self.segment_order):
            layer_seed = seed * 17 + idx * 101
            self.layers[name] = LinearLayer(
                in_dim,
                out_dim,
                activation=act,
                seed=layer_seed,
                init=init,
            )
        self._last_cache: List[Dict[str, Dict[str, List[float]]]] | None = None

    def forward(self, sequence: Sequence[Sequence[float]]) -> List[List[float]]:
        outputs: List[List[float]] = []
        caches: List[Dict[str, Dict[str, List[float]]]] = []
        for column in sequence:
            cache_entry: Dict[str, Dict[str, List[float]]] = {}
            h1, cache_front = self.layers["encoder_front"].forward(column)
            cache_entry["encoder_front"] = cache_front
            base_logits, cache_back = self.layers["encoder_back"].forward(h1)
            cache_entry["encoder_back"] = cache_back
            seq_hidden, cache_seq_front = self.layers["sequence_front"].forward(h1)
            cache_entry["sequence_front"] = cache_seq_front
            seq_logits, cache_seq_back = self.layers["sequence_back"].forward(seq_hidden)
            cache_entry["sequence_back"] = cache_seq_back
            combined = [base + seq for base, seq in zip(base_logits, seq_logits)]
            caches.append(cache_entry)
            outputs.append(combined)
        self._last_cache = caches
        return outputs

    def grad_template(self) -> Gradients:
        template: Gradients = {}
        for name, layer in self.layers.items():
            template[name] = {
                "weights": [[0.0 for _ in range(layer.in_dim)] for _ in range(layer.out_dim)],
                "bias": [0.0 for _ in range(layer.out_dim)],
            }
        return template

    def backward(self, grad_logits: Sequence[Sequence[float]]) -> Gradients:
        if self._last_cache is None:
            raise RuntimeError("forward must be called before backward")
        grads = self.grad_template()
        for cache_entry, grad_out in zip(self._last_cache, grad_logits):
            self._backward_into(grads, cache_entry, grad_out)
        return grads

    def _backward_into(
        self,
        grads: Gradients,
        cache_entry: Dict[str, Dict[str, List[float]]],
        grad_out: Sequence[float],
    ) -> None:
        grad_seq_hidden = self._backward_layer(
            layer=self.layers["sequence_back"],
            cache=cache_entry["sequence_back"],
            grad_output=list(grad_out),
            store=grads["sequence_back"],
        )
        grad_seq_front = self._backward_layer(
            layer=self.layers["sequence_front"],
            cache=cache_entry["sequence_front"],
            grad_output=grad_seq_hidden,
            store=grads["sequence_front"],
        )
        grad_enc_back = self._backward_layer(
            layer=self.layers["encoder_back"],
            cache=cache_entry["encoder_back"],
            grad_output=list(grad_out),
            store=grads["encoder_back"],
        )
        merged = [a + b for a, b in zip(grad_seq_front, grad_enc_back)]
        self._backward_layer(
            layer=self.layers["encoder_front"],
            cache=cache_entry["encoder_front"],
            grad_output=merged,
            store=grads["encoder_front"],
        )

    @staticmethod
    def _backward_layer(
        layer: LinearLayer,
        cache: Dict[str, List[float]],
        grad_output: Sequence[float],
        store: Dict[str, List[List[float]] | List[float]],
    ) -> List[float]:
        grad = list(grad_output)
        if layer.activation == "tanh":
            act = cache["output"]
            grad = [value * (1.0 - act[idx] * act[idx]) for idx, value in enumerate(grad)]
        inputs = cache["input"]
        weights = layer.weights
        store_w = store["weights"]  # type: ignore[index]
        store_b = store["bias"]  # type: ignore[index]
        for out_idx, grad_value in enumerate(grad):
            store_b[out_idx] += grad_value
            row_accum = store_w[out_idx]
            for in_idx, input_value in enumerate(inputs):
                row_accum[in_idx] += grad_value * input_value
        grad_input = [0.0 for _ in range(layer.in_dim)]
        for in_idx in range(layer.in_dim):
            acc = 0.0
            for out_idx, grad_value in enumerate(grad):
                acc += grad_value * weights[out_idx][in_idx]
            grad_input[in_idx] = acc
        return grad_input

    def state_dict(self) -> Dict[str, Dict[str, List[List[float]] | List[float]]]:
        return {name: layer.state_dict() for name, layer in self.layers.items()}

    def load_state_dict(self, state: Dict[str, Dict[str, object]]) -> None:
        for name, params in state.items():
            if name in self.layers:
                self.layers[name].load_state_dict(params)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------


def zero_grad(model: CurriculumModel) -> Gradients:
    return model.grad_template()


def add_grad(accum: Gradients, grad: Gradients) -> None:
    for name in accum:
        weights_accum = accum[name]["weights"]  # type: ignore[index]
        weights_grad = grad[name]["weights"]  # type: ignore[index]
        for row_accum, row_grad in zip(weights_accum, weights_grad):
            for idx in range(len(row_accum)):
                row_accum[idx] += row_grad[idx]
        bias_accum = accum[name]["bias"]  # type: ignore[index]
        bias_grad = grad[name]["bias"]  # type: ignore[index]
        for idx in range(len(bias_accum)):
            bias_accum[idx] += bias_grad[idx]


def scale_grad(grad: Gradients, factor: float) -> None:
    for name in grad:
        weights = grad[name]["weights"]  # type: ignore[index]
        bias = grad[name]["bias"]  # type: ignore[index]
        for row in weights:
            for idx in range(len(row)):
                row[idx] *= factor
        for idx in range(len(bias)):
            bias[idx] *= factor


def grad_norm(grad: Gradients) -> float:
    total = 0.0
    for name in grad:
        weights = grad[name]["weights"]  # type: ignore[index]
        bias = grad[name]["bias"]  # type: ignore[index]
        for row in weights:
            for value in row:
                total += value * value
        for value in bias:
            total += value * value
    return math.sqrt(total)


def clip_gradient(grad: Gradients, max_norm: float) -> float:
    norm = grad_norm(grad)
    if norm > max_norm and max_norm > 0.0:
        scale = max_norm / (norm + 1e-12)
        scale_grad(grad, scale)
        return norm * scale
    return norm


def zero_segment(segment: Dict[str, List[List[float]] | List[float]]) -> None:
    for row in segment["weights"]:  # type: ignore[index]
        for idx in range(len(row)):
            row[idx] = 0.0
    bias = segment["bias"]  # type: ignore[index]
    for idx in range(len(bias)):
        bias[idx] = 0.0


def mask_gradients(grad: Gradients, trainable_segments: Sequence[str]) -> None:
    allowed = set(trainable_segments)
    for name in grad:
        if name not in allowed:
            zero_segment(grad[name])


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class SGDOptimizer:
    name = "sgd"

    def step(
        self,
        model: CurriculumModel,
        grad: Gradients,
        lr: float,
        trainable_segments: Sequence[str],
    ) -> None:
        allowed = set(trainable_segments)
        for name, layer in model.layers.items():
            if name not in allowed:
                continue
            grad_segment = grad[name]
            for row_weights, row_grad in zip(layer.weights, grad_segment["weights"]):  # type: ignore[index]
                for idx in range(len(row_weights)):
                    row_weights[idx] -= lr * row_grad[idx]
            for idx in range(len(layer.bias)):
                layer.bias[idx] -= lr * grad_segment["bias"][idx]  # type: ignore[index]

    def state_dict(self) -> Dict[str, float]:
        return {"type": self.name}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        _ = state  # no-op for vanilla SGD


class AdamOptimizer:
    name = "adam"

    def __init__(self, model: CurriculumModel) -> None:
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        self.m: Gradients = {}
        self.v: Gradients = {}
        for name, layer in model.layers.items():
            self.m[name] = {
                "weights": [[0.0 for _ in range(layer.in_dim)] for _ in range(layer.out_dim)],
                "bias": [0.0 for _ in range(layer.out_dim)],
            }
            self.v[name] = {
                "weights": [[0.0 for _ in range(layer.in_dim)] for _ in range(layer.out_dim)],
                "bias": [0.0 for _ in range(layer.out_dim)],
            }

    def step(
        self,
        model: CurriculumModel,
        grad: Gradients,
        lr: float,
        trainable_segments: Sequence[str],
    ) -> None:
        self.t += 1
        allowed = set(trainable_segments)
        for name, layer in model.layers.items():
            if name not in allowed:
                continue
            grad_segment = grad[name]
            m_segment = self.m[name]
            v_segment = self.v[name]
            for row_idx, row_grad in enumerate(grad_segment["weights"]):  # type: ignore[index]
                m_row = m_segment["weights"][row_idx]  # type: ignore[index]
                v_row = v_segment["weights"][row_idx]  # type: ignore[index]
                weights = layer.weights[row_idx]
                for col_idx, grad_value in enumerate(row_grad):
                    m_row[col_idx] = self.beta1 * m_row[col_idx] + (1 - self.beta1) * grad_value
                    v_row[col_idx] = self.beta2 * v_row[col_idx] + (1 - self.beta2) * (grad_value ** 2)
                    m_hat = m_row[col_idx] / (1 - self.beta1 ** self.t)
                    v_hat = v_row[col_idx] / (1 - self.beta2 ** self.t)
                    weights[col_idx] -= lr * m_hat / (math.sqrt(v_hat) + self.eps)
            bias_grad = grad_segment["bias"]  # type: ignore[index]
            m_bias = m_segment["bias"]  # type: ignore[index]
            v_bias = v_segment["bias"]  # type: ignore[index]
            for idx, grad_value in enumerate(bias_grad):
                m_bias[idx] = self.beta1 * m_bias[idx] + (1 - self.beta1) * grad_value
                v_bias[idx] = self.beta2 * v_bias[idx] + (1 - self.beta2) * (grad_value ** 2)
                m_hat = m_bias[idx] / (1 - self.beta1 ** self.t)
                v_hat = v_bias[idx] / (1 - self.beta2 ** self.t)
                layer.bias[idx] -= lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def state_dict(self) -> Dict[str, object]:
        return {
            "type": self.name,
            "t": self.t,
            "m": self.m,
            "v": self.v,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.t = int(state.get("t", 0))
        m_state = state.get("m")
        v_state = state.get("v")
        if isinstance(m_state, dict):
            for name, segment in self.m.items():
                payload = m_state.get(name)
                if not isinstance(payload, dict):
                    continue
                weights = payload.get("weights")
                bias = payload.get("bias")
                if isinstance(weights, list) and isinstance(bias, list):
                    segment["weights"] = [
                        [float(value) for value in row]
                        for row in weights
                    ]
                    segment["bias"] = [float(value) for value in bias]
        if isinstance(v_state, dict):
            for name, segment in self.v.items():
                payload = v_state.get(name)
                if not isinstance(payload, dict):
                    continue
                weights = payload.get("weights")
                bias = payload.get("bias")
                if isinstance(weights, list) and isinstance(bias, list):
                    segment["weights"] = [
                        [float(value) for value in row]
                        for row in weights
                    ]
                    segment["bias"] = [float(value) for value in bias]


def create_optimizer(name: str, model: CurriculumModel) -> object:
    if name == "sgd":
        return SGDOptimizer()
    if name == "adam":
        return AdamOptimizer(model)
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def softmax(logits: Sequence[float]) -> List[float]:
    shifted = [value - max(logits) for value in logits]
    exps = [math.exp(value) for value in shifted]
    denom = sum(exps) or 1.0
    return [value / denom for value in exps]


def softmax_with_temperature(logits: Sequence[float], temperature: float) -> List[float]:
    if temperature <= 0:
        raise ValueError("Temperature must be positive for softmax.")
    scaled = [value / temperature for value in logits]
    return softmax(scaled)


def knowledge_distillation_loss(
    student: Sequence[Sequence[float]],
    teacher: Sequence[Sequence[float]],
    temperature: float,
) -> Tuple[float, List[List[float]]]:
    if len(student) != len(teacher):
        raise ValueError("Teacher and student logits must align in time dimension")
    if not student:
        return 0.0, []
    grads: List[List[float]] = []
    total = 0.0
    scale = temperature * temperature
    for stud_step, teach_step in zip(student, teacher):
        teacher_probs = softmax_with_temperature(teach_step, temperature)
        student_probs = softmax_with_temperature(stud_step, temperature)
        step_loss = 0.0
        step_grad: List[float] = []
        for sp, tp in zip(student_probs, teacher_probs):
            sp_clamped = max(sp, 1e-12)
            tp_clamped = max(tp, 1e-12)
            step_loss += tp_clamped * math.log(tp_clamped / sp_clamped)
            step_grad.append((sp - tp) * scale)
        grads.append(step_grad)
        total += step_loss
    avg_loss = total / max(1, len(student))
    return avg_loss, grads


def cross_entropy_sequence(
    logits: List[List[float]],
    target_index: int,
    tet_mode: str,
) -> Tuple[float, List[List[float]]]:
    weights = tet_weights(len(logits), tet_mode)
    grad_logits: List[List[float]] = []
    losses: List[float] = []
    for step_logits in logits:
        probs = softmax(step_logits)
        losses.append(-math.log(probs[target_index] + 1e-12))
        grad = [prob for prob in probs]
        grad[target_index] -= 1.0
        grad_logits.append(grad)
    total_loss = sum(weight * loss for weight, loss in zip(weights, losses))
    for weight, grad in zip(weights, grad_logits):
        for idx in range(len(grad)):
            grad[idx] *= weight
    return total_loss, grad_logits


def smoothing_penalty(
    logits: List[List[float]],
    grad: List[List[float]],
    strength: float,
) -> float:
    if len(logits) <= 1 or strength <= 0.0:
        return 0.0
    denom = (len(logits) - 1) * len(logits[0])
    penalty = 0.0
    for t in range(1, len(logits)):
        for k in range(len(logits[t])):
            diff = logits[t][k] - logits[t - 1][k]
            penalty += diff * diff
            coeff = (2.0 * diff * strength) / max(1, denom)
            grad[t][k] += coeff
            grad[t - 1][k] -= coeff
    return (strength * penalty) / max(1, denom)


def ctc_with_regularization(
    logits: List[List[float]],
    target: Sequence[int],
    tet_mode: str,
    smoothing_lambda: float,
) -> Tuple[float, float, float, List[List[float]]]:
    base_loss, grad = ctc.ctc_loss_with_grad(logits, target, blank=BLANK_INDEX)
    if not grad:
        return base_loss, base_loss, 0.0, grad
    weights = tet_weights(len(grad), tet_mode)
    for t, weight in enumerate(weights):
        scale = weight * len(grad)
        for k in range(len(grad[t])):
            grad[t][k] *= scale
    smooth_loss = smoothing_penalty(logits, grad, smoothing_lambda)
    total_loss = base_loss + smooth_loss
    return total_loss, base_loss, smooth_loss, grad


def _edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    rows, cols = len(seq_a) + 1, len(seq_b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def compute_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    distance = _edit_distance(list(ref), list(hyp))
    return distance / len(ref)


def compute_wer(ref: str, hyp: str) -> float:
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else float(len(hyp_tokens))
    distance = _edit_distance(ref_tokens, hyp_tokens)
    return distance / len(ref_tokens)


def compute_loss_for_sample(
    logits: List[List[float]],
    text: str,
    sample_cfg: StageConfig,
    *,
    teacher_logits: List[List[float]] | None = None,
    distill_lambda: float = 0.0,
    kd_temperature: float = KD_TEMPERATURE,
) -> LossBreakdown:
    kd_total = 0.0
    if sample_cfg.loss_type == "ce":
        indices = text_to_indices(text)
        if not indices:
            raise ValueError("Classification samples require at least one target symbol")
        total, grad_logits = cross_entropy_sequence(logits, indices[0], sample_cfg.tet_mode)
        if teacher_logits is not None and distill_lambda > 0.0:
            kd_loss, kd_grad = knowledge_distillation_loss(logits, teacher_logits, kd_temperature)
            kd_total = distill_lambda * kd_loss
            for t in range(len(grad_logits)):
                for k in range(len(grad_logits[t])):
                    grad_logits[t][k] += distill_lambda * kd_grad[t][k]
            total += kd_total
        return LossBreakdown(total=total, main=total - kd_total, smooth=0.0, grad_logits=grad_logits, kd=kd_total)
    target = text_to_indices(text)
    total, main, smooth, grad_logits = ctc_with_regularization(
        logits,
        target,
        sample_cfg.tet_mode,
        sample_cfg.smoothing_lambda,
    )
    if teacher_logits is not None and distill_lambda > 0.0:
        kd_loss, kd_grad = knowledge_distillation_loss(logits, teacher_logits, kd_temperature)
        kd_total = distill_lambda * kd_loss
        for t in range(len(grad_logits)):
            for k in range(len(grad_logits[t])):
                grad_logits[t][k] += distill_lambda * kd_grad[t][k]
        total += kd_total
    decoded = ctc.greedy_decode(logits, blank=BLANK_INDEX)
    return LossBreakdown(
        total=total,
        main=main,
        smooth=smooth,
        grad_logits=grad_logits,
        kd=kd_total,
        cer=compute_cer(text, decoded),
        wer=compute_wer(text, decoded),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainArgs:
    stage: str
    out_dir: Path
    epochs: int
    steps_per_epoch: int
    batch_size: int
    optimizer: str
    lr: float
    min_lr: float
    cosine_anneal: bool
    clip_grad: float
    replay_override: float | None
    resume: bool
    log_every: int
    save_every: int
    seed: int
    distill: bool
    distill_lambda: float
    teacher_ckpt: Path | None


@dataclass
class TrainingResult:
    stage: str
    steps: int
    initial_loss: float | None
    final_loss: float | None


def cosine_anneal(initial: float, minimum: float, step: int, total_steps: int) -> float:
    if not total_steps:
        return initial
    progress = min(step / total_steps, 1.0)
    return minimum + 0.5 * (initial - minimum) * (1.0 + math.cos(math.pi * progress))


def serialize_rng(rng: Random) -> Dict[str, object]:
    version, state, gauss = rng.getstate()
    return {
        "version": version,
        "state": list(state),
        "gauss": gauss,
    }


def deserialize_rng(rng: Random, payload: Dict[str, object]) -> None:
    version = payload.get("version", 3)
    state = payload.get("state")
    gauss = payload.get("gauss", None)
    if not isinstance(state, list):
        raise ValueError("Invalid RNG state")
    rng.setstate((int(version), tuple(int(x) for x in state), gauss))


def run_epoch(
    model: CurriculumModel,
    loader: Iterable[List[TrainSample]],
    optimizer: object,
    stage: str,
    cfg: StageConfig,
    *,
    lr_fn: Callable[[int], float],
    clip_grad: float,
    train_segments: Tuple[str, ...],
    log_every: int,
    metrics_file: TextIO,
    global_step: int,
    save_every: int,
    ckpt_path: Path,
    samplers: Dict[str, StageSampler],
    rng: Random,
    teacher_model: CurriculumModel | None,
    kd_lambda: float,
) -> Tuple[int, float | None, float | None]:
    first_loss: float | None = None
    last_loss: float | None = None
    for batch in loader:
        global_step += 1
        lr = lr_fn(global_step)
        grad_accum = zero_grad(model)
        total_loss = 0.0
        total_main = 0.0
        total_smooth = 0.0
        total_fire = 0.0
        total_width_ratio = 0.0
        cer_total = 0.0
        wer_total = 0.0
        cer_count = 0
        sample_count = 0
        total_kd = 0.0
        old_correct = 0
        old_total = 0
        for sample in batch:
            sequence = image_to_sequence(sample.gray)
            if not sequence:
                continue
            logits = model.forward(sequence)
            apply_kd = teacher_model is not None and sample.stage == stage and kd_lambda > 0.0
            teacher_logits = teacher_model.forward(sequence) if apply_kd and teacher_model else None
            breakdown = compute_loss_for_sample(
                logits,
                sample.text,
                STAGE_CONFIGS[sample.stage],
                teacher_logits=teacher_logits,
                distill_lambda=kd_lambda if apply_kd else 0.0,
            )
            grad_params = model.backward(breakdown.grad_logits)
            add_grad(grad_accum, grad_params)
            total_loss += breakdown.total
            total_main += breakdown.main
            total_smooth += breakdown.smooth
            total_kd += breakdown.kd
            if breakdown.cer is not None and breakdown.wer is not None:
                cer_total += breakdown.cer
                wer_total += breakdown.wer
                cer_count += 1
            total_fire += compute_fire_rate(sample.gray)
            width = len(sample.gray[0]) if sample.gray and sample.gray[0] else 1
            total_width_ratio += len(sequence) / max(1, width)
            sample_count += 1
            if sample.stage != stage:
                prediction = decode_prediction(sample.stage, logits)
                target_text = sample.text[0] if (sample.text and STAGE_CONFIGS[sample.stage].loss_type == "ce") else sample.text
                if prediction == target_text:
                    old_correct += 1
                old_total += 1
        if sample_count == 0:
            continue
        inv_batch = 1.0 / sample_count
        scale_grad(grad_accum, inv_batch)
        mask_gradients(grad_accum, train_segments)
        batch_loss = total_loss * inv_batch
        batch_main = total_main * inv_batch
        batch_smooth = total_smooth * inv_batch
        batch_kd = total_kd * inv_batch
        avg_fire = total_fire * inv_batch
        avg_width_ratio = total_width_ratio * inv_batch
        avg_cer = (cer_total / cer_count) if cer_count else None
        avg_wer = (wer_total / cer_count) if cer_count else None
        old_top1 = (old_correct / old_total) if old_total else None
        grad_norm_before = grad_norm(grad_accum)
        grad_norm_after = clip_gradient(grad_accum, clip_grad)
        if isinstance(optimizer, (SGDOptimizer, AdamOptimizer)):
            optimizer.step(model, grad_accum, lr, train_segments)
        if first_loss is None:
            first_loss = batch_loss
        last_loss = batch_loss
        if global_step % log_every == 0:
            log_entry: Dict[str, object] = {
                "iter": global_step,
                "stage": stage,
                "loss": batch_loss,
                "main_loss": batch_main,
                "smooth_loss": batch_smooth,
                "kd_loss": batch_kd,
                "lr": lr,
                "grad_norm": grad_norm_before,
                "grad_norm_clipped": grad_norm_after,
                "fire_rate": avg_fire,
                "width_ratio": avg_width_ratio,
            }
            if avg_cer is not None:
                log_entry["cer"] = avg_cer
            if avg_wer is not None:
                log_entry["wer"] = avg_wer
            if old_top1 is not None:
                log_entry["old_top1"] = old_top1
            metrics_file.write(json.dumps(log_entry) + "\n")
            metrics_file.flush()
            msg = (
                f"[step {global_step}] loss={batch_loss:.4f} main={batch_main:.4f} smooth={batch_smooth:.4f} "
                f"lr={lr:.5f} fire={avg_fire:.3f} W'={avg_width_ratio:.2f}"
            )
            if batch_kd > 0.0:
                msg += f" kd={batch_kd:.4f}"
            if avg_cer is not None:
                msg += f" cer={avg_cer:.3f}"
            if avg_wer is not None:
                msg += f" wer={avg_wer:.3f}"
            if old_top1 is not None:
                msg += f" old={old_top1:.3f}"
            print(msg)
        if global_step % save_every == 0:
            save_ckpt(ckpt_path, model, optimizer, global_step, samplers, rng)
    return global_step, first_loss, last_loss


def find_teacher_checkpoint(stage_cfg: StageConfig, out_dir: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override
    base = out_dir.parent
    for prev in reversed(stage_cfg.previous):
        candidate = base / prev.lower() / "ckpt.json"
        if candidate.exists():
            return candidate
    return None


def load_model_weights(path: Path | None, model: CurriculumModel) -> bool:
    if path is None or not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state = payload.get("model")
    if isinstance(state, dict):
        model.load_state_dict(state)
        return True
    return False


def train_stage(args: TrainArgs) -> TrainingResult:
    stage_name = args.stage.upper()
    if stage_name not in STAGE_CONFIGS:
        raise ValueError(f"Unsupported stage: {stage_name}")
    stage_cfg = STAGE_CONFIGS[stage_name]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.json"
    metrics_path = out_dir / "metrics.jsonl"
    model = CurriculumModel(feature_dim=5, output_dim=NUM_CLASSES, seed=args.seed)
    optimizer = create_optimizer(args.optimizer, model)
    sampler_names = {stage_cfg.name, *stage_cfg.previous}
    samplers: Dict[str, StageSampler] = {name: StageSampler(name) for name in sampler_names}
    rng = Random(args.seed)
    global_step = 0
    result = TrainingResult(stage=stage_cfg.name, steps=0, initial_loss=None, final_loss=None)

    if args.resume and ckpt_path.exists():
        global_step = load_ckpt(ckpt_path, model, optimizer, samplers, rng)
        if metrics_path.exists():
            try:
                with metrics_path.open("r", encoding="utf-8") as handle:
                    lines = [line.strip() for line in handle if line.strip()]
                if lines:
                    first = json.loads(lines[0])
                    last = json.loads(lines[-1])
                    result.initial_loss = float(first.get("loss"))
                    result.final_loss = float(last.get("loss"))
            except json.JSONDecodeError:
                pass

    total_steps = max(1, args.epochs * max(1, args.steps_per_epoch))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("a", encoding="utf-8")
    replay_ratio = (
        args.replay_override if args.replay_override is not None else stage_cfg.replay_ratio
    )
    buffers = prepare_replay_buffers(stage_cfg, samplers, args.batch_size, replay_ratio)
    teacher_model: CurriculumModel | None = None
    if args.distill and stage_cfg.previous:
        teacher_path = find_teacher_checkpoint(stage_cfg, out_dir, args.teacher_ckpt)
        if teacher_path is None:
            print("[distill] No prior checkpoint found; skipping knowledge distillation.")
        else:
            teacher_model = CurriculumModel(feature_dim=5, output_dim=NUM_CLASSES, seed=args.seed + 97)
            if load_model_weights(teacher_path, teacher_model):
                print(f"[distill] Loaded teacher checkpoint from {teacher_path}")
            else:
                print(f"[distill] Failed to load teacher from {teacher_path}, disabling KD.")
                teacher_model = None

    def lr_fn(step: int) -> float:
        if not args.cosine_anneal:
            return args.lr
        return cosine_anneal(args.lr, args.min_lr, step, total_steps)

    try:
        for epoch in range(args.epochs):
            loader = build_loader(
                stage_cfg,
                samplers,
                steps=args.steps_per_epoch,
                batch_size=args.batch_size,
                rng=rng,
                replay_ratio=replay_ratio,
                buffers=buffers,
            )
            global_step, first_loss, last_loss = run_epoch(
                model,
                loader,
                optimizer,
                stage_cfg.name,
                stage_cfg,
                lr_fn=lr_fn,
                clip_grad=args.clip_grad,
                train_segments=stage_cfg.train_segments,
                log_every=args.log_every,
                metrics_file=metrics_file,
                global_step=global_step,
                save_every=args.save_every,
                ckpt_path=ckpt_path,
                samplers=samplers,
                rng=rng,
                teacher_model=teacher_model,
                kd_lambda=args.distill_lambda if teacher_model is not None else 0.0,
            )
            if result.initial_loss is None and first_loss is not None:
                result.initial_loss = first_loss
            if last_loss is not None:
                result.final_loss = last_loss
            result.steps = global_step
    finally:
        metrics_file.close()
    save_ckpt(ckpt_path, model, optimizer, global_step, samplers, rng)
    return result


def compute_fire_rate(gray: List[List[int]], timesteps: int = 4) -> float:
    spikes_tensor = spikes.encode_ttfs(gray, T=timesteps)
    total_spikes = 0
    for frame in spikes_tensor:
        for row in frame:
            total_spikes += sum(row)
    total_possible = timesteps * len(gray) * len(gray[0]) if gray and gray[0] else 1
    return total_spikes / total_possible


def load_ckpt(
    path: Path,
    model: CurriculumModel,
    optimizer: object,
    samplers: Dict[str, StageSampler],
    rng: Random,
) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    model_state = state.get("model")
    if isinstance(model_state, dict):
        model.load_state_dict(model_state)
    opt_state = state.get("optimizer", {})
    opt_type = opt_state.get("type") if isinstance(opt_state, dict) else None
    opt_name = getattr(optimizer, "name", None)
    if opt_type == opt_name and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(opt_state)  # type: ignore[call-arg]
    sampler_state = state.get("samplers", {})
    if isinstance(sampler_state, dict):
        for name, sampler in samplers.items():
            payload = sampler_state.get(name)
            if isinstance(payload, dict):
                sampler.load_state_dict(payload)
    rng_state = state.get("rng")
    if isinstance(rng_state, dict):
        deserialize_rng(rng, rng_state)
    return int(state.get("step", 0))


def save_ckpt(
    path: Path,
    model: CurriculumModel,
    optimizer: object,
    step: int,
    samplers: Dict[str, StageSampler],
    rng: Random,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": getattr(optimizer, "state_dict", lambda: {"type": "sgd"})(),
        "step": step,
        "samplers": {name: sampler.state_dict() for name, sampler in samplers.items()},
        "rng": serialize_rng(rng),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle)


# ---------------------------------------------------------------------------
# CLI & demo
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SNN-OCR curriculum stages.")
    parser.add_argument("--stage", type=str, default="S1")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=0.001)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--replay", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--distill", action="store_true", help="Enable lightweight LwF knowledge distillation.")
    parser.add_argument(
        "--distill-lambda",
        type=float,
        default=0.3,
        help="Weight for the KD term when --distill is active.",
    )
    parser.add_argument("--teacher", type=Path, default=None, help="Optional explicit teacher checkpoint.")
    parser.add_argument("--demo", action="store_true", help="Run a short S1 demo.")
    return parser.parse_args()


def run_demo() -> None:
    demo_out = Path("runs") / "demo_s1"
    demo_args = TrainArgs(
        stage="S1",
        out_dir=demo_out,
        epochs=1,
        steps_per_epoch=30,
        batch_size=8,
        optimizer="adam",
        lr=0.02,
        min_lr=0.005,
        cosine_anneal=True,
        clip_grad=1.0,
        replay_override=None,
        resume=False,
        log_every=5,
        save_every=1000,
        seed=3,
        distill=False,
        distill_lambda=0.0,
        teacher_ckpt=None,
    )
    result = train_stage(demo_args)
    if result.initial_loss is not None and result.final_loss is not None:
        print(
            f"Demo completed: loss {result.initial_loss:.4f} -> {result.final_loss:.4f} "
            f"in {result.steps} steps."
        )


def main() -> None:
    args_ns = parse_args()
    if args_ns.demo:
        run_demo()
        return
    out_dir = args_ns.out or Path("runs") / args_ns.stage.lower()
    train_args = TrainArgs(
        stage=args_ns.stage,
        out_dir=out_dir,
        epochs=args_ns.epochs,
        steps_per_epoch=args_ns.steps,
        batch_size=args_ns.batch,
        optimizer=args_ns.optimizer,
        lr=args_ns.lr,
        min_lr=args_ns.min_lr,
        cosine_anneal=args_ns.cosine,
        clip_grad=args_ns.clip,
        replay_override=args_ns.replay,
        resume=args_ns.resume,
        log_every=args_ns.log_every,
        save_every=args_ns.save_every,
        seed=args_ns.seed,
        distill=args_ns.distill,
        distill_lambda=args_ns.distill_lambda,
        teacher_ckpt=args_ns.teacher,
    )
    train_stage(train_args)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        run_demo()
    else:
        main()
