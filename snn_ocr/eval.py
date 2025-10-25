"""Evaluation utilities for SNN-OCR models (metrics, logging, visualization)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import wrap
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_ocr import otc, pgm, render, synth, spikes
from snn_ocr.ctc import best_alignment_path, greedy_decode, symbol_table
from snn_ocr.train import (
    BLANK_INDEX,
    CHAR_TO_INDEX,
    STAGE_CONFIGS,
    CurriculumModel,
    compute_fire_rate,
    image_to_sequence,
    softmax,
    text_to_indices,
)

SYMBOLS: Tuple[str, ...] = symbol_table()
ENERGY_TIMESTEPS = 6
OTC_GATE = "var"
OTC_TOPK = 5


def _pretty_symbol(index: int) -> str:
    if index == BLANK_INDEX:
        return "<blank>"
    if 0 <= index < len(SYMBOLS):
        symbol = SYMBOLS[index]
        if symbol == " ":
            return "<space>"
        if symbol == "\n":
            return "<lf>"
        return symbol
    return str(index)


def _collapse_path(indices: Sequence[int]) -> str:
    collapsed: List[str] = []
    prev: int | None = None
    for index in indices:
        if index == BLANK_INDEX:
            prev = None
            continue
        if prev == index:
            continue
        symbol = SYMBOLS[index] if 0 <= index < len(SYMBOLS) else str(index)
        collapsed.append(symbol)
        prev = index
    return "".join(collapsed)


def _alignment_with_marks(ref: str, hyp: str) -> Tuple[str, str, str]:
    aligned_ref, aligned_hyp = align_strings(ref, hyp)
    marks: List[str] = []
    for ref_ch, hyp_ch in zip(aligned_ref, aligned_hyp):
        if ref_ch == hyp_ch and ref_ch != "-":
            marks.append("|")
        else:
            marks.append("^")
    return aligned_ref, aligned_hyp, "".join(marks)


def _compute_blank_profile(logits: Sequence[Sequence[float]]) -> Tuple[List[float], float]:
    if not logits:
        return [], 0.0
    series: List[float] = []
    for step in logits:
        probs = softmax(step)
        series.append(probs[BLANK_INDEX] if len(probs) > BLANK_INDEX else 0.0)
    mean = sum(series) / len(series)
    return series, mean


def _compute_ctc_path(stage: str, logits: Sequence[Sequence[float]], target: str) -> Tuple[List[str], str]:
    if stage in ("S1", "S2"):
        return [], ""
    if not logits:
        return [], ""
    try:
        target_indices = text_to_indices(target)
    except ValueError:
        return [], ""
    if not target_indices:
        return [], ""
    path = best_alignment_path(logits, target_indices, blank=BLANK_INDEX)
    tokens = [_pretty_symbol(index) for index in path]
    collapsed = _collapse_path(path)
    return tokens, collapsed


def _gray_to_feature_tensor(image: List[List[int]]) -> List[List[List[List[float]]]]:
    if not image or not image[0]:
        return []
    height = len(image)
    width = len(image[0])
    time_slice: List[List[List[float]]] = []
    for y in range(height):
        row: List[List[float]] = []
        for x in range(width):
            row.append([image[y][x] / 255.0])
        time_slice.append(row)
    return [time_slice]


def _compute_otc_summary(image: List[List[int]]) -> Tuple[str, List[Tuple[int, float]], List[float]]:
    features = _gray_to_feature_tensor(image)
    if not features:
        return "", [], []
    info_values = otc.column_information(features, gate=OTC_GATE)
    heatmap = otc.ascii_heatmap(info_values)
    ranked = sorted(enumerate(info_values), key=lambda item: item[1], reverse=True)
    topk = ranked[:OTC_TOPK]
    return heatmap, topk, info_values


def _format_timeline(tokens: Sequence[str], chunk: int = 10) -> str:
    if not tokens:
        return "<empty>"
    parts: List[str] = []
    current: List[str] = []
    for idx, token in enumerate(tokens):
        current.append(f"{idx:02d}:{token}")
        if len(current) >= chunk:
            parts.append(" | ".join(current))
            current = []
    if current:
        parts.append(" | ".join(current))
    return "\n".join(parts)


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


def energy_stats(spike_tensor: Sequence[Sequence[Sequence[int]]]) -> Dict[str, object]:
    if not spike_tensor:
        raise ValueError("Spike tensor must be non-empty")
    T = len(spike_tensor)
    height = len(spike_tensor[0]) if T else 0
    width = len(spike_tensor[0][0]) if height else 0
    if height == 0 or width == 0:
        raise ValueError("Spike tensor must have positive spatial dimensions")
    total_spikes = 0
    duty_cycle: List[float] = []
    pixels = height * width
    for frame in spike_tensor:
        if len(frame) != height:
            raise ValueError("Inconsistent frame height in spike tensor")
        frame_spikes = 0
        for row in frame:
            if len(row) != width:
                raise ValueError("Inconsistent frame width in spike tensor")
            frame_spikes += sum(row)
        total_spikes += frame_spikes
        duty_cycle.append(frame_spikes / max(1, pixels))
    per_pixel = total_spikes / max(1, pixels)
    return {
        "total": total_spikes,
        "per_pixel": per_pixel,
        "duty_cycle": duty_cycle,
    }


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
    alignment_marks: str = ""
    energy_total: int = 0
    energy_per_pixel: float = 0.0
    duty_cycle: List[float] = field(default_factory=list)
    blank_ratio: float = 0.0
    blank_series: List[float] = field(default_factory=list)
    ctc_path: List[str] = field(default_factory=list)
    ctc_collapsed: str = ""
    otc_heatmap: str = ""
    otc_values: List[float] = field(default_factory=list)
    otc_top_columns: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class EvalMetrics:
    top1: float
    cer: float
    wer: float
    avg_fire_rate: float
    avg_width: float
    samples: int
    energy_total: float
    energy_per_pixel: float
    duty_cycle: List[float]
    blank_ratio: float


def dump_examples(
    examples: Sequence[ExampleRecord],
    out_dir: Path,
    metrics: EvalMetrics | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[str] = []
    for idx, record in enumerate(examples, start=1):
        target_file = out_dir / f"example_{idx:03d}.txt"
        with target_file.open("w", encoding="utf-8") as handle:
            handle.write(f"File: {record.path}\n")
            handle.write(f"Reference: {record.text}\n")
            handle.write(f"Hypothesis: {record.prediction}\n")
            handle.write(f"CER: {record.cer_value:.3f}, WER: {record.wer_value:.3f}\n")
            handle.write("Alignment (ref/hyp/marks):\n")
            handle.write(record.alignment_ref + "\n")
            handle.write(record.alignment_hyp + "\n")
            handle.write(record.alignment_marks + "\n")
            handle.write(f"Blank duty: {record.blank_ratio:.3f}\n")
            if record.blank_series:
                blank_heat = otc.ascii_heatmap(record.blank_series)
                preview = ", ".join(f"{value:.2f}" for value in record.blank_series[:12])
                handle.write(f"Blank timeline: {blank_heat}\n")
                handle.write(f"Blank probs[:12]: {preview}\n")
            if record.ctc_path:
                collapsed = record.ctc_collapsed or "<empty>"
                handle.write(f"CTC collapsed: {collapsed}\n")
                handle.write("CTC path timeline:\n")
                handle.write(_format_timeline(record.ctc_path) + "\n")
            if record.otc_heatmap:
                handle.write("OTC information heatmap:\n")
                for line in wrap(record.otc_heatmap, width=64):
                    handle.write(line + "\n")
                if record.otc_top_columns:
                    tops = ", ".join(
                        f"c{col}={value:.3f}" for col, value in record.otc_top_columns
                    )
                    handle.write(f"OTC top columns: {tops}\n")
            handle.write("ASCII preview:\n")
            handle.write(record.ascii_art + "\n")
            handle.write(
                "Energy: total={:.0f}, per_pixel={:.3f}, duty={}\n".format(
                    record.energy_total,
                    record.energy_per_pixel,
                    ",".join(f"{value:.3f}" for value in record.duty_cycle),
                )
            )
        manifest.append(target_file.name)
    if metrics is not None:
        summary_path = out_dir / "report.json"
        payload = {
            "top1": metrics.top1,
            "cer": metrics.cer,
            "wer": metrics.wer,
            "avg_fire_rate": metrics.avg_fire_rate,
            "avg_width": metrics.avg_width,
            "samples": metrics.samples,
            "energy_total": metrics.energy_total,
            "energy_per_pixel": metrics.energy_per_pixel,
            "duty_cycle": metrics.duty_cycle,
            "blank_ratio": metrics.blank_ratio,
            "examples": manifest,
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_checkpoint(path: Path) -> CurriculumModel:
    model = CurriculumModel(feature_dim=5, output_dim=len(SYMBOLS), seed=11)
    if not path.exists():
        return model
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state = payload.get("model")
    if isinstance(state, dict):
        model.load_state_dict(state)
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
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, [], 0.0), []
    correct = 0
    total_chars = 0
    total_char_err = 0
    total_words = 0
    total_word_err = 0
    total_fire = 0.0
    total_width = 0.0
    energy_total_sum = 0.0
    energy_per_pixel_sum = 0.0
    duty_cycle_sum: List[float] | None = None
    blank_ratio_sum = 0.0
    records: List[ExampleRecord] = []
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
        blank_series, blank_ratio = _compute_blank_profile(logits)
        blank_ratio_sum += blank_ratio
        spike_tensor = spikes.encode_ttfs(image, T=ENERGY_TIMESTEPS)
        energy = energy_stats(spike_tensor)
        energy_total_sum += float(energy["total"])
        energy_per_pixel_sum += float(energy["per_pixel"])
        duty = energy["duty_cycle"]  # type: ignore[assignment]
        if isinstance(duty_cycle_sum, list):
            for idx, value in enumerate(duty):
                duty_cycle_sum[idx] += value
        else:
            duty_cycle_sum = [float(value) for value in duty]
        aligned_ref, aligned_hyp, marks = _alignment_with_marks(text, prediction)
        ascii_art = render.ascii_preview(image)
        ctc_tokens, ctc_collapsed = _compute_ctc_path(stage, logits, text)
        otc_heatmap, otc_top, otc_values = _compute_otc_summary(image)
        rec = ExampleRecord(
            path=entry["path"],  # type: ignore[arg-type]
            text=text,
            prediction=prediction,
            cer_value=cer(text, prediction),
            wer_value=wer(text, prediction),
            ascii_art=ascii_art,
            alignment_ref=aligned_ref,
            alignment_hyp=aligned_hyp,
            alignment_marks=marks,
            energy_total=int(energy["total"]),
            energy_per_pixel=float(energy["per_pixel"]),
            duty_cycle=[float(value) for value in duty],
            blank_ratio=blank_ratio,
            blank_series=[float(value) for value in blank_series],
            ctc_path=ctc_tokens,
            ctc_collapsed=ctc_collapsed,
            otc_heatmap=otc_heatmap,
            otc_values=otc_values,
            otc_top_columns=otc_top,
        )
        records.append(rec)
    error_candidates = [rec for rec in records if rec.prediction != rec.text]
    if not error_candidates:
        error_candidates = records[:]
    sorted_errors = sorted(
        error_candidates,
        key=lambda rec: (rec.cer_value, rec.wer_value, rec.blank_ratio),
        reverse=True,
    )
    showcase = sorted_errors[: sample_count]
    metrics = EvalMetrics(
        top1=correct / total,
        cer=total_char_err / total_chars if total_chars else 0.0,
        wer=total_word_err / total_words if total_words else 0.0,
        avg_fire_rate=total_fire / total,
        avg_width=total_width / total,
        samples=total,
        energy_total=energy_total_sum / total,
        energy_per_pixel=energy_per_pixel_sum / total,
        duty_cycle=[value / total for value in duty_cycle_sum] if duty_cycle_sum else [],
        blank_ratio=blank_ratio_sum / total if total else 0.0,
    )
    return metrics, showcase


def profile_single_example(
    stage: str,
    data_dir: Path,
    checkpoint: Path | None = None,
    *,
    sample_index: int = 0,
) -> Dict[str, object]:
    """Run a single-sample forward pass and collect timing/spike stats."""
    stage = stage.upper()
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Unsupported stage: {stage}")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    limit = sample_index + 1
    entries = load_dataset(stage, data_dir, limit=limit)
    if not entries:
        raise ValueError(f"Dataset at {data_dir} is empty")
    idx = min(sample_index, len(entries) - 1)
    entry = entries[idx]
    model = load_checkpoint(checkpoint) if checkpoint else load_checkpoint(Path("__missing.ckpt__"))
    image = entry["image"]  # type: ignore[index]
    text = entry["text"]  # type: ignore[index]
    sequence = image_to_sequence(image)
    start = perf_counter()
    logits = model.forward(sequence)
    elapsed_ms = (perf_counter() - start) * 1000.0
    prediction = predict_sequence(stage, logits)
    blank_series, blank_ratio = _compute_blank_profile(logits)
    spike_pack = spikes.encode_ttfs(image, T=ENERGY_TIMESTEPS, return_stats=True)
    if isinstance(spike_pack, tuple):
        spike_tensor, spike_summary = spike_pack
    else:  # pragma: no cover - return_stats=True guarantees tuple
        spike_tensor = spike_pack
        histogram = spikes.spike_histogram(spike_tensor)
        spike_summary = {"histogram": histogram, "total": sum(histogram)}
    energy = energy_stats(spike_tensor)
    return {
        "stage": stage,
        "file": str(entry["path"]),  # type: ignore[index]
        "index": idx,
        "text": text,
        "prediction": prediction,
        "match": prediction == text,
        "sequence_length": len(sequence),
        "inference_ms": elapsed_ms,
        "blank_ratio": blank_ratio,
        "blank_series": blank_series,
        "fire_rate": compute_fire_rate(image),
        "spike_total": float(spike_summary.get("total", 0.0)),
        "spike_hist": spike_summary.get("histogram", []),
        "energy_total": float(energy["total"]),
        "energy_per_pixel": float(energy["per_pixel"]),
        "duty_cycle": energy["duty_cycle"],
    }


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
    dump_examples(examples, vis_dir, metrics)
    print(
        f"Demo metrics ({stage}): top1={metrics.top1:.2f}, "
        f"CER={metrics.cer:.3f}, WER={metrics.wer:.3f}, "
        f"fire={metrics.avg_fire_rate:.3f}, W'={metrics.avg_width:.2f}, "
        f"blank={metrics.blank_ratio:.3f}, energy={metrics.energy_total:.1f}, "
        f"duty={','.join(f'{v:.3f}' for v in metrics.duty_cycle)}"
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
    dump_examples(examples, vis_dir, metrics)
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
                "energy_total": metrics.energy_total,
                "energy_per_pixel": metrics.energy_per_pixel,
                "duty_cycle": metrics.duty_cycle,
                "blank_ratio": metrics.blank_ratio,
            }
        )
    )


def preview_examples(count: int = 3) -> None:
    stage = "S3"
    data_dir = Path("runs") / "preview_eval_s3"
    ensure_dataset(stage, data_dir, size=20)
    metrics, examples = evaluate(stage, data_dir, checkpoint=None, limit=20, sample_count=count)
    vis_dir = Path("runs") / "vis" / f"{stage.lower()}_preview"
    dump_examples(examples[:count], vis_dir, metrics)
    print(
        f"Preview ({stage}): saved {min(len(examples), count)} samples to {vis_dir}, "
        f"avg energy={metrics.energy_total:.1f} spikes, blank={metrics.blank_ratio:.3f}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        preview_examples()
    else:
        main()
