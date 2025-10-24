"""Unified curriculum training script for the SNN-OCR prototype (standard library only)."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Sequence, Tuple

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


STAGE_CONFIGS: Dict[str, StageConfig] = {
    "S1": StageConfig(
        name="S1",
        tet_mode="uniform",
        loss_type="ce",
        smoothing_lambda=0.0,
        replay_ratio=0.0,
        previous=(),
    ),
    "S2": StageConfig(
        name="S2",
        tet_mode="tail",
        loss_type="ce",
        smoothing_lambda=0.0,
        replay_ratio=0.1,
        previous=("S1",),
    ),
    "S3": StageConfig(
        name="S3",
        tet_mode="tail",
        loss_type="ctc",
        smoothing_lambda=0.01,
        replay_ratio=0.15,
        previous=("S1", "S2"),
    ),
    "S4": StageConfig(
        name="S4",
        tet_mode="tail",
        loss_type="ctc",
        smoothing_lambda=0.02,
        replay_ratio=0.2,
        previous=("S1", "S2", "S3"),
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
# Simple sequence model (linear layer shared across tokens)
# ---------------------------------------------------------------------------


class SequenceLinear:
    """Per-token linear classifier with shared weights."""

    def __init__(self, feature_dim: int, output_dim: int, seed: int = 0) -> None:
        rng = Random(seed)
        self.weights: List[List[float]] = [
            [rng.uniform(-0.05, 0.05) for _ in range(feature_dim)]
            for _ in range(output_dim)
        ]
        self.bias: List[float] = [0.0 for _ in range(output_dim)]

    @property
    def feature_dim(self) -> int:
        return len(self.weights[0]) if self.weights else 0

    @property
    def output_dim(self) -> int:
        return len(self.weights)

    def forward(self, sequence: Sequence[Sequence[float]]) -> List[List[float]]:
        logits: List[List[float]] = []
        for column in sequence:
            out: List[float] = []
            for row, bias in zip(self.weights, self.bias):
                acc = bias
                for value, weight in zip(column, row):
                    acc += value * weight
                out.append(acc)
            logits.append(out)
        return logits

    def backward(
        self,
        sequence: Sequence[Sequence[float]],
        grad_logits: Sequence[Sequence[float]],
    ) -> Dict[str, List[List[float]] | List[float]]:
        grad_w = [[0.0 for _ in row] for row in self.weights]
        grad_b = [0.0 for _ in self.bias]
        for column, grad in zip(sequence, grad_logits):
            for class_idx, grad_value in enumerate(grad):
                grad_b[class_idx] += grad_value
                weight_grad_row = grad_w[class_idx]
                for feature_idx, feature_value in enumerate(column):
                    weight_grad_row[feature_idx] += grad_value * feature_value
        return {"weights": grad_w, "bias": grad_b}

    def state_dict(self) -> Dict[str, List[List[float]] | List[float]]:
        return {
            "weights": [[value for value in row] for row in self.weights],
            "bias": [value for value in self.bias],
        }

    def load_state_dict(self, state: Dict[str, List[List[float]] | List[float]]) -> None:
        weights = state.get("weights")
        bias = state.get("bias")
        if not isinstance(weights, list) or not isinstance(bias, list):
            raise ValueError("Invalid state dict for SequenceLinear")
        self.weights = [[float(value) for value in row] for row in weights]  # type: ignore[arg-type]
        self.bias = [float(value) for value in bias]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------


def zero_grad(model: SequenceLinear) -> Dict[str, List[List[float]] | List[float]]:
    return {
        "weights": [[0.0 for _ in row] for row in model.weights],
        "bias": [0.0 for _ in model.bias],
    }


def add_grad(
    accum: Dict[str, List[List[float]] | List[float]],
    grad: Dict[str, List[List[float]] | List[float]],
) -> None:
    for row_accum, row_grad in zip(accum["weights"], grad["weights"]):  # type: ignore[index]
        for idx in range(len(row_accum)):
            row_accum[idx] += row_grad[idx]
    bias_accum = accum["bias"]  # type: ignore[index]
    bias_grad = grad["bias"]  # type: ignore[index]
    for idx in range(len(bias_accum)):
        bias_accum[idx] += bias_grad[idx]


def scale_grad(
    grad: Dict[str, List[List[float]] | List[float]],
    factor: float,
) -> None:
    for row in grad["weights"]:  # type: ignore[index]
        for idx in range(len(row)):
            row[idx] *= factor
    bias = grad["bias"]  # type: ignore[index]
    for idx in range(len(bias)):
        bias[idx] *= factor


def grad_norm(grad: Dict[str, List[List[float]] | List[float]]) -> float:
    total = 0.0
    for row in grad["weights"]:  # type: ignore[index]
        for value in row:
            total += value * value
    for value in grad["bias"]:  # type: ignore[index]
        total += value * value
    return math.sqrt(total)


def clip_gradient(
    grad: Dict[str, List[List[float]] | List[float]],
    max_norm: float,
) -> float:
    norm = grad_norm(grad)
    if norm > max_norm and max_norm > 0.0:
        scale = max_norm / (norm + 1e-12)
        scale_grad(grad, scale)
        return norm * scale
    return norm


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class SGDOptimizer:
    name = "sgd"

    def __init__(self) -> None:
        self.momentum = 0.0

    def step(
        self,
        model: SequenceLinear,
        grad: Dict[str, List[List[float]] | List[float]],
        lr: float,
    ) -> None:
        for row_weights, row_grad in zip(model.weights, grad["weights"]):  # type: ignore[index]
            for idx in range(len(row_weights)):
                row_weights[idx] -= lr * row_grad[idx]
        for idx in range(len(model.bias)):
            model.bias[idx] -= lr * grad["bias"][idx]  # type: ignore[index]

    def state_dict(self) -> Dict[str, float]:
        return {"type": self.name}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        _ = state  # no-op for vanilla SGD


class AdamOptimizer:
    name = "adam"

    def __init__(self, feature_dim: int, output_dim: int) -> None:
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        self.m_w = [[0.0 for _ in range(feature_dim)] for _ in range(output_dim)]
        self.v_w = [[0.0 for _ in range(feature_dim)] for _ in range(output_dim)]
        self.m_b = [0.0 for _ in range(output_dim)]
        self.v_b = [0.0 for _ in range(output_dim)]

    def step(
        self,
        model: SequenceLinear,
        grad: Dict[str, List[List[float]] | List[float]],
        lr: float,
    ) -> None:
        self.t += 1
        for row_idx, row_grad in enumerate(grad["weights"]):  # type: ignore[index]
            m_row = self.m_w[row_idx]
            v_row = self.v_w[row_idx]
            weights = model.weights[row_idx]
            for col_idx, grad_value in enumerate(row_grad):
                m_row[col_idx] = self.beta1 * m_row[col_idx] + (1 - self.beta1) * grad_value
                v_row[col_idx] = self.beta2 * v_row[col_idx] + (1 - self.beta2) * (grad_value ** 2)
                m_hat = m_row[col_idx] / (1 - self.beta1 ** self.t)
                v_hat = v_row[col_idx] / (1 - self.beta2 ** self.t)
                weights[col_idx] -= lr * m_hat / (math.sqrt(v_hat) + self.eps)
        for idx, grad_value in enumerate(grad["bias"]):  # type: ignore[index]
            self.m_b[idx] = self.beta1 * self.m_b[idx] + (1 - self.beta1) * grad_value
            self.v_b[idx] = self.beta2 * self.v_b[idx] + (1 - self.beta2) * (grad_value ** 2)
            m_hat = self.m_b[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b[idx] / (1 - self.beta2 ** self.t)
            model.bias[idx] -= lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def state_dict(self) -> Dict[str, object]:
        return {
            "type": self.name,
            "t": self.t,
            "m_w": self.m_w,
            "v_w": self.v_w,
            "m_b": self.m_b,
            "v_b": self.v_b,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.t = int(state.get("t", 0))
        self.m_w = [[float(value) for value in row] for row in state.get("m_w", self.m_w)]  # type: ignore[arg-type]
        self.v_w = [[float(value) for value in row] for row in state.get("v_w", self.v_w)]  # type: ignore[arg-type]
        self.m_b = [float(value) for value in state.get("m_b", self.m_b)]  # type: ignore[arg-type]
        self.v_b = [float(value) for value in state.get("v_b", self.v_b)]  # type: ignore[arg-type]


def create_optimizer(name: str, model: SequenceLinear) -> object:
    if name == "sgd":
        return SGDOptimizer()
    if name == "adam":
        return AdamOptimizer(model.feature_dim, model.output_dim)
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def softmax(logits: Sequence[float]) -> List[float]:
    shifted = [value - max(logits) for value in logits]
    exps = [math.exp(value) for value in shifted]
    denom = sum(exps) or 1.0
    return [value / denom for value in exps]


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


@dataclass
class TrainingResult:
    stage: str
    steps: int
    initial_loss: float | None
    final_loss: float | None


def cosine_anneal_lr(initial: float, minimum: float, step: int, total_steps: int) -> float:
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


def train_stage(args: TrainArgs) -> TrainingResult:
    stage_name = args.stage.upper()
    if stage_name not in STAGE_CONFIGS:
        raise ValueError(f"Unsupported stage: {stage_name}")
    stage_cfg = STAGE_CONFIGS[stage_name]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.json"
    metrics_path = out_dir / "metrics.jsonl"

    model = SequenceLinear(feature_dim=5, output_dim=NUM_CLASSES, seed=args.seed)
    optimizer = create_optimizer(args.optimizer, model)
    samplers: Dict[str, StageSampler] = {
        name: StageSampler(name) for name in (stage_cfg.previous + (stage_cfg.name,))
    }
    rng = Random(args.seed)
    global_step = 0
    result = TrainingResult(stage=stage_cfg.name, steps=0, initial_loss=None, final_loss=None)

    if args.resume and ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        model.load_state_dict(state["model"])
        if state.get("optimizer", {}).get("type") == "adam" and isinstance(optimizer, AdamOptimizer):
            optimizer.load_state_dict(state["optimizer"])
        elif state.get("optimizer", {}).get("type") == "sgd" and isinstance(optimizer, SGDOptimizer):
            optimizer.load_state_dict(state["optimizer"])
        global_step = int(state.get("step", 0))
        sampler_state = state.get("samplers", {})
        for sampler_name, sampler in samplers.items():
            if sampler_name in sampler_state:
                sampler.load_state_dict(sampler_state[sampler_name])
        rng_state = state.get("rng")
        if rng_state:
            deserialize_rng(rng, rng_state)
        if metrics_path.exists() and result.initial_loss is None:
            with metrics_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
            if lines:
                first = json.loads(lines[0])
                last = json.loads(lines[-1])
                result.initial_loss = float(first.get("loss"))
                result.final_loss = float(last.get("loss"))

    total_steps = args.epochs * args.steps_per_epoch
    metrics_file = metrics_path.open("a", encoding="utf-8")

    try:
        for epoch in range(args.epochs):
            for _ in range(args.steps_per_epoch):
                global_step += 1
                lr = args.lr
                if args.cosine_anneal:
                    lr = cosine_anneal_lr(args.lr, args.min_lr, global_step, total_steps)
                grad_accum = zero_grad(model)
                batch_loss = 0.0
                batch_main_loss = 0.0
                batch_smooth = 0.0
                batch_fire = 0.0
                batch_width = 0.0
                replay_ratio = (
                    args.replay_override
                    if args.replay_override is not None
                    else stage_cfg.replay_ratio
                )
                for _ in range(args.batch_size):
                    if stage_cfg.previous and rng.random() < replay_ratio:
                        chosen = rng.choice(stage_cfg.previous)
                    else:
                        chosen = stage_cfg.name
                    sample_cfg = STAGE_CONFIGS[chosen]
                    gray, text = samplers[chosen].next_sample()
                    sequence = image_to_sequence(gray)
                    logits = model.forward(sequence)
                    if not logits:
                        continue
                    if sample_cfg.loss_type == "ce":
                        target_idx = text_to_indices(text)[0]
                        total_sample_loss, grad_logits = cross_entropy_sequence(
                            logits, target_idx, sample_cfg.tet_mode
                        )
                        main_loss = total_sample_loss
                        smooth_loss = 0.0
                    else:
                        target_indices = text_to_indices(text)
                        total_sample_loss, main_loss, smooth_loss, grad_logits = ctc_with_regularization(
                            logits,
                            target_indices,
                            sample_cfg.tet_mode,
                            sample_cfg.smoothing_lambda,
                        )
                    grad_params = model.backward(sequence, grad_logits)
                    add_grad(grad_accum, grad_params)
                    batch_loss += total_sample_loss
                    batch_main_loss += main_loss
                    batch_smooth += smooth_loss
                    fire_rate = compute_fire_rate(gray)
                    batch_fire += fire_rate
                    batch_width += len(sequence)
                if args.batch_size > 0:
                    inv_batch = 1.0 / args.batch_size
                    scale_grad(grad_accum, inv_batch)
                    batch_loss *= inv_batch
                    batch_main_loss *= inv_batch
                    batch_smooth *= inv_batch
                    batch_fire *= inv_batch
                    batch_width *= inv_batch
                grad_norm_before = grad_norm(grad_accum)
                clipped_norm = clip_gradient(grad_accum, args.clip_grad)
                if isinstance(optimizer, (SGDOptimizer, AdamOptimizer)):
                    optimizer.step(model, grad_accum, lr)  # type: ignore[arg-type]
                result.steps = global_step
                if result.initial_loss is None:
                    result.initial_loss = batch_loss
                result.final_loss = batch_loss
                if global_step % args.log_every == 0:
                    log_entry = {
                        "step": global_step,
                        "stage": stage_cfg.name,
                        "loss": batch_loss,
                        "main_loss": batch_main_loss,
                        "smooth_loss": batch_smooth,
                        "lr": lr,
                        "grad_norm": grad_norm_before,
                        "grad_norm_clipped": clipped_norm,
                        "fire_rate": batch_fire,
                        "width_out": batch_width,
                    }
                    metrics_file.write(json.dumps(log_entry) + "\n")
                    metrics_file.flush()
                    print(
                        f"[step {global_step}] loss={batch_loss:.4f} "
                        f"main={batch_main_loss:.4f} smooth={batch_smooth:.4f} "
                        f"lr={lr:.5f} fire={batch_fire:.3f} W'={batch_width:.2f}"
                    )
                if global_step % args.save_every == 0:
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        global_step,
                        samplers,
                        rng,
                    )
    finally:
        metrics_file.close()
    save_checkpoint(ckpt_path, model, optimizer, global_step, samplers, rng)
    return result


def compute_fire_rate(gray: List[List[int]], timesteps: int = 4) -> float:
    spikes_tensor = spikes.encode_ttfs(gray, T=timesteps)
    total_spikes = 0
    for frame in spikes_tensor:
        for row in frame:
            total_spikes += sum(row)
    total_possible = timesteps * len(gray) * len(gray[0]) if gray and gray[0] else 1
    return total_spikes / total_possible


def save_checkpoint(
    path: Path,
    model: SequenceLinear,
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
    )
    train_stage(train_args)


if __name__ == "__main__":
    main()
