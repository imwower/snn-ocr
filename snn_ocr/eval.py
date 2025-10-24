"""Evaluation utilities for SNN-OCR models (metrics, logging, visualization)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_ocr import pgm, render, synth
from snn_ocr.ctc import greedy_decode, symbol_table
from snn_ocr.train import (
    BLANK_INDEX,
    CHAR_TO_INDEX,
    STAGE_CONFIGS,
    SequenceLinear,
    compute_fire_rate,
    image_to_sequence,
    softmax,
    text_to_indices,
)

SYMBOLS: Tuple[str, ...] = symbol_table()


def edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
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


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    distance = edit_distance(list(ref), list(hyp))
    return distance / len(ref)


def wer(ref: str, hyp: str) -> float:
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else float(len(hyp_tokens))
    distance = edit_distance(ref_tokens, hyp_tokens)
    return distance / len(ref_tokens)


def align_strings(ref: str, hyp: str) -> Tuple[str, str]:
    ref_seq = list(ref)
    hyp_seq = list(hyp)
    rows, cols = len(ref_seq) + 1, len(hyp_seq) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = (i - 1, 0)
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = (0, j - 1)
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_seq[i - 1] == hyp_seq[j - 1] else 1
            choices = [
                (dp[i - 1][j] + 1, (i - 1, j)),
                (dp[i][j - 1] + 1, (i, j - 1)),
                (dp[i - 1][j - 1] + cost, (i - 1, j - 1)),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])
    i, j = len(ref_seq), len(hyp_seq)
    aligned_ref: List[str] = []
    aligned_hyp: List[str] = []
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj = prev
        if pi == i - 1 and pj == j - 1:
            aligned_ref.append(ref_seq[i - 1])
            aligned_hyp.append(hyp_seq[j - 1])
        elif pi == i - 1 and pj == j:
            aligned_ref.append(ref_seq[i - 1])
            aligned_hyp.append("-")
        else:
            aligned_ref.append("-")
            aligned_hyp.append(hyp_seq[j - 1])
        i, j = pi, pj
    return "".join(reversed(aligned_ref)), "".join(reversed(aligned_hyp))


@dataclass
class ExampleRecord:
    path: Path
    text: str
    prediction: str
    cer_value: float
    wer_value: float
    ascii_art: str
    alignment_ref: str
    alignment_hyp: str


@dataclass
class EvalMetrics:
    top1: float
    cer: float
    wer: float
    avg_fire_rate: float
    avg_width: float
    samples: int


def dump_examples(examples: Sequence[ExampleRecord], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(examples, start=1):
        target_file = out_dir / f"example_{idx:03d}.txt"
        with target_file.open("w", encoding="utf-8") as handle:
            handle.write(f"File: {record.path}\n")
            handle.write(f"Reference: {record.text}\n")
            handle.write(f"Hypothesis: {record.prediction}\n")
            handle.write(f"CER: {record.cer_value:.3f}, WER: {record.wer_value:.3f}\n")
            handle.write("Alignment (ref/hyp):\n")
            handle.write(record.alignment_ref + "\n")
            handle.write(record.alignment_hyp + "\n")
            handle.write("ASCII preview:\n")
            handle.write(record.ascii_art + "\n")


def load_checkpoint(path: Path) -> SequenceLinear:
    if not path.exists():
        return SequenceLinear(feature_dim=5, output_dim=len(SYMBOLS), seed=11)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model = SequenceLinear(feature_dim=5, output_dim=len(SYMBOLS), seed=11)
    model.load_state_dict(payload["model"])
    return model


def load_dataset(stage: str, data_dir: Path, limit: int | None = None) -> List[Dict[str, object]]:
    labels_path = data_dir / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.jsonl under {data_dir}")
    entries: List[Dict[str, object]] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(entries) >= limit:
                break
            payload = json.loads(line)
            img_path = data_dir / payload["file"]
            gray = pgm.load_pgm(img_path)
            entries.append({"path": img_path, "text": payload["text"], "image": gray})
    return entries


def predict_sequence(stage: str, logits: Sequence[Sequence[float]]) -> str:
    if stage in ("S1", "S2"):
        accumulator = [0.0 for _ in range(len(logits[0]))]
        for step in logits:
            probs = softmax(step)
            for idx, value in enumerate(probs):
                accumulator[idx] += value
        best = max(range(len(accumulator)), key=lambda idx: accumulator[idx])
        return SYMBOLS[best]
    decoded = greedy_decode(logits, blank=BLANK_INDEX)
    return decoded


def evaluate(
    stage: str,
    data_dir: Path,
    checkpoint: Path | None = None,
    *,
    limit: int | None = None,
    sample_count: int = 5,
) -> Tuple[EvalMetrics, List[ExampleRecord]]:
    stage = stage.upper()
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Unsupported stage: {stage}")
    entries = load_dataset(stage, data_dir, limit=limit)
    model = load_checkpoint(checkpoint) if checkpoint else load_checkpoint(Path("__missing.ckpt__"))
    total = len(entries)
    if total == 0:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0), []
    correct = 0
    total_chars = 0
    total_char_err = 0
    total_words = 0
    total_word_err = 0
    total_fire = 0.0
    total_width = 0.0
    rng = Random(0)
    showcase: List[ExampleRecord] = []
    for entry in entries:
        image = entry["image"]  # type: ignore[index]
        text = entry["text"]  # type: ignore[index]
        sequence = image_to_sequence(image)
        logits = model.forward(sequence)
        prediction = predict_sequence(stage, logits)
        if stage in ("S1", "S2"):
            target_indices = text_to_indices(text)
            pred_index = CHAR_TO_INDEX.get(prediction, None)
            if target_indices and pred_index == target_indices[0]:
                correct += 1
        else:
            if prediction == text:
                correct += 1
        char_err = edit_distance(list(text), list(prediction))
        total_char_err += char_err
        total_chars += max(1, len(text))
        word_err = edit_distance(text.split(), prediction.split())
        total_word_err += word_err
        total_words += max(1, len(text.split()))
        total_fire += compute_fire_rate(image)
        total_width += len(sequence)
        if len(showcase) < sample_count and rng.random() < 0.6:
            aligned_ref, aligned_hyp = align_strings(text, prediction)
            ascii_art = render.ascii_preview(image)
            rec = ExampleRecord(
                path=entry["path"],  # type: ignore[arg-type]
                text=text,
                prediction=prediction,
                cer_value=cer(text, prediction),
                wer_value=wer(text, prediction),
                ascii_art=ascii_art,
                alignment_ref=aligned_ref,
                alignment_hyp=aligned_hyp,
            )
            showcase.append(rec)
    metrics = EvalMetrics(
        top1=correct / total,
        cer=total_char_err / total_chars if total_chars else 0.0,
        wer=total_word_err / total_words if total_words else 0.0,
        avg_fire_rate=total_fire / total,
        avg_width=total_width / total,
        samples=total,
    )
    return metrics, showcase


def ensure_dataset(stage: str, data_dir: Path, size: int = 30) -> None:
    labels_path = data_dir / "labels.jsonl"
    if labels_path.exists():
        return
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    synth.generate_dataset(stage, size, data_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SNN-OCR checkpoints.")
    parser.add_argument("--stage", type=str, default="S3")
    parser.add_argument("--data", type=Path, default=None, help="Dataset directory (labels.jsonl).")
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint (.json).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--vis", type=Path, default=Path("runs") / "vis")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def run_demo() -> None:
    stage = "S3"
    data_dir = Path("runs") / "demo_eval_s3"
    ensure_dataset(stage, data_dir, size=20)
    metrics, examples = evaluate(stage, data_dir, checkpoint=None, limit=20, sample_count=5)
    vis_dir = Path("runs") / "vis" / stage.lower()
    dump_examples(examples, vis_dir)
    print(
        f"Demo metrics ({stage}): top1={metrics.top1:.2f}, "
        f"CER={metrics.cer:.3f}, WER={metrics.wer:.3f}, "
        f"fire={metrics.avg_fire_rate:.3f}, W'={metrics.avg_width:.2f}"
    )


def main() -> None:
    args = parse_args()
    if args.demo:
        run_demo()
        return
    stage = args.stage.upper()
    if args.data is None:
        data_dir = Path("runs") / f"eval_{stage.lower()}"
        ensure_dataset(stage, data_dir, size=40)
    else:
        data_dir = args.data
    metrics, examples = evaluate(stage, data_dir, checkpoint=args.ckpt, limit=args.limit, sample_count=args.examples)
    vis_dir = args.vis / stage.lower()
    dump_examples(examples, vis_dir)
    print(
        json.dumps(
            {
                "stage": stage,
                "top1": metrics.top1,
                "cer": metrics.cer,
                "wer": metrics.wer,
                "fire_rate": metrics.avg_fire_rate,
                "width_out": metrics.avg_width,
                "samples": metrics.samples,
            }
        )
    )


if __name__ == "__main__":
    main()
