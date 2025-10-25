"""Curriculum-aware dataset synthesis utilities for the SNN-OCR project."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Sequence, Tuple

from snn_ocr import pgm, render

GrayGrid = List[List[int]]

DIGITS = "0123456789"
LETTERS_UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS_LO = LETTERS_UP.lower()
PUNCT = ".,!?"
ALLOWED_CHARS = set(DIGITS + LETTERS_UP + LETTERS_LO + PUNCT + " \n")


@dataclass(frozen=True)
class StageSetting:
    width: int
    height: int
    font: str
    jitter: int
    noise: float
    contrast: float
    mode: str  # "digit", "char", "word", "sentence"
    char_pool: str
    min_len: int
    max_len: int
    word_count: Tuple[int, int] | None = None
    scale_range: Tuple[int, int] = (1, 1)
    shear_range: Tuple[int, int] = (0, 0)
    dilate_range: Tuple[int, int] = (0, 0)
    background_range: Tuple[int, int] = (0, 0)
    foreground_range: Tuple[int, int] = (255, 255)
    punctuation: str = ""
    punctuation_prob: float = 0.0
    allow_newline: bool = False
    newline_prob: float = 0.0


@dataclass
class SampleMeta:
    progress: float
    text_length: int
    word_count: int | None
    contrast: float
    noise: float
    scale: int
    shear: int
    dilate: int
    background: int
    foreground: int
    jitter: int


@dataclass
class StageProfile:
    stage: str
    lengths: List[int] = field(default_factory=list)
    word_counts: List[int] = field(default_factory=list)
    contrasts: List[float] = field(default_factory=list)
    noises: List[float] = field(default_factory=list)

    def update(self, meta: SampleMeta) -> None:
        self.lengths.append(meta.text_length)
        if meta.word_count is not None:
            self.word_counts.append(meta.word_count)
        self.contrasts.append(meta.contrast)
        self.noises.append(meta.noise)

    def summary(self) -> Dict[str, object]:
        def _avg(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        length_hist = dict(sorted(Counter(self.lengths).items())) if self.lengths else {}
        word_hist = dict(sorted(Counter(self.word_counts).items())) if self.word_counts else {}

        return {
            "stage": self.stage,
            "count": len(self.lengths),
            "avg_length": _avg([float(x) for x in self.lengths]),
            "min_length": min(self.lengths) if self.lengths else 0,
            "max_length": max(self.lengths) if self.lengths else 0,
            "avg_word_count": _avg([float(x) for x in self.word_counts]) if self.word_counts else None,
            "contrast_range": (
                min(self.contrasts) if self.contrasts else 0.0,
                max(self.contrasts) if self.contrasts else 0.0,
            ),
            "noise_range": (
                min(self.noises) if self.noises else 0.0,
                max(self.noises) if self.noises else 0.0,
            ),
            "length_histogram": length_hist,
            "word_count_histogram": word_hist if word_hist else None,
        }


STAGE_SETTINGS: Dict[str, StageSetting] = {
    "S1": StageSetting(
        width=32,
        height=32,
        font="5x7",
        jitter=0,
        noise=0.0,
        contrast=1.0,
        mode="digit",
        char_pool=DIGITS,
        min_len=1,
        max_len=1,
        scale_range=(2, 3),
        dilate_range=(0, 1),
        background_range=(0, 15),
        foreground_range=(220, 255),
    ),
    "S2": StageSetting(
        width=40,
        height=36,
        font="5x7",
        jitter=1,
        noise=0.01,
        contrast=1.05,
        mode="char",
        char_pool=LETTERS_UP + LETTERS_LO,
        min_len=1,
        max_len=2,
        scale_range=(2, 3),
        shear_range=(-1, 1),
        background_range=(0, 25),
        foreground_range=(210, 255),
    ),
    "S3": StageSetting(
        width=160,
        height=32,
        font="5x7",
        jitter=1,
        noise=0.03,
        contrast=1.1,
        mode="word",
        char_pool=LETTERS_LO,
        min_len=3,
        max_len=8,
        word_count=(1, 1),
        scale_range=(2, 3),
        shear_range=(-1, 2),
        dilate_range=(0, 2),
        background_range=(5, 35),
        foreground_range=(200, 255),
    ),
    "S4": StageSetting(
        width=384,
        height=48,
        font="5x7",
        jitter=2,
        noise=0.05,
        contrast=1.15,
        mode="sentence",
        char_pool=LETTERS_LO,
        min_len=2,
        max_len=8,
        word_count=(3, 9),
        scale_range=(2, 3),
        shear_range=(-2, 2),
        dilate_range=(0, 2),
        background_range=(10, 50),
        foreground_range=(190, 250),
        punctuation=PUNCT,
        punctuation_prob=0.75,
        allow_newline=True,
        newline_prob=0.5,
    ),
}


def _rand_int(rng: Random, bounds: Tuple[int, int]) -> int:
    low, high = bounds
    if low > high:
        low, high = high, low
    return rng.randint(low, high)


def _random_word(rng: Random, length: int, pool: str) -> str:
    return "".join(rng.choice(pool) for _ in range(length))


def _weighted_choice(rng: Random, progress: float, strength: float = 0.7) -> float:
    progress = max(0.0, min(1.0, progress))
    return (1.0 - strength) * rng.random() + strength * progress


def _progressive_int(bounds: Tuple[int, int], rng: Random, progress: float) -> int:
    low, high = bounds
    if low == high:
        return low
    if low > high:
        low, high = high, low
    mix = _weighted_choice(rng, progress)
    value = low + (high - low) * mix
    return int(round(value))


def _sample_text(
    stage: str,
    setting: StageSetting,
    rng: Random,
    *,
    target_len: int | None = None,
    word_target: int | None = None,
) -> Tuple[str, Dict[str, int | None]]:
    mode = setting.mode
    if mode in ("digit", "char"):
        length = target_len if target_len is not None else _rand_int(rng, (setting.min_len, setting.max_len))
        token = "".join(rng.choice(setting.char_pool) for _ in range(length))
        if mode == "char" and rng.random() < 0.3:
            token = token.upper()
        return token, {"word_count": None, "length_target": length}
    if mode == "word":
        length = target_len if target_len is not None else _rand_int(rng, (setting.min_len, setting.max_len))
        word = _random_word(rng, length, setting.char_pool)
        if rng.random() < 0.4:
            word = word.upper()
        return word, {"word_count": 1, "length_target": length}
    if mode == "sentence":
        if not setting.word_count:
            raise ValueError("Sentence mode requires word_count range")
        count = word_target if word_target is not None else _rand_int(rng, setting.word_count)
        words: List[str] = []
        for _ in range(count):
            length = target_len if target_len is not None else _rand_int(rng, (setting.min_len, setting.max_len))
            word = _random_word(rng, length, setting.char_pool)
            if rng.random() < 0.2:
                word = word.upper()
            words.append(word)
        tokens: List[str] = words[:]
        if (
            setting.allow_newline
            and len(tokens) >= 2
            and rng.random() < setting.newline_prob
        ):
            split = rng.randint(1, len(tokens) - 1)
            tokens.insert(split, "\n")
        parts: List[str] = []
        for idx, token in enumerate(tokens):
            if token == "\n":
                parts.append("\n")
                continue
            parts.append(token)
            if idx != len(tokens) - 1 and tokens[idx + 1] != "\n":
                parts.append(" ")
        sentence = "".join(parts)
        lines = sentence.split("\n")
        sentence = "\n".join(line[:1].upper() + line[1:] if line else line for line in lines)
        if setting.punctuation and rng.random() < setting.punctuation_prob:
            sentence = sentence.rstrip(" ")
            sentence += rng.choice(setting.punctuation)
        return sentence, {"word_count": count, "length_target": length}
    raise KeyError(f"Unsupported stage mode {mode!r}")


def _assert_valid_chars(text: str) -> None:
    if not set(text).issubset(ALLOWED_CHARS):
        raise ValueError(f"Text contains unsupported characters: {text}")


def _schedule_parameters(setting: StageSetting, rng: Random, progress: float) -> Dict[str, object]:
    contrast = setting.contrast * (0.9 + 0.2 * _weighted_choice(rng, progress, strength=0.5))
    noise_base = setting.noise
    noise = noise_base * (0.5 + 0.5 * _weighted_choice(rng, progress, strength=0.6)) if noise_base > 0 else 0.0
    scale = max(1, _progressive_int(setting.scale_range, rng, progress))
    shear = _progressive_int(setting.shear_range, rng, progress)
    dilate = max(0, _progressive_int(setting.dilate_range, rng, progress))
    background = _rand_int(rng, setting.background_range)
    foreground = max(background + 1, _rand_int(rng, setting.foreground_range))
    length_target = None
    word_target = None
    if setting.mode in ("digit", "char", "word"):
        length_target = _progressive_int((setting.min_len, setting.max_len), rng, progress)
    if setting.mode == "sentence" and setting.word_count:
        word_target = _progressive_int(setting.word_count, rng, progress)
        length_target = _progressive_int((setting.min_len, setting.max_len), rng, progress)
    return {
        "contrast": contrast,
        "noise": noise,
        "scale": scale,
        "shear": shear,
        "dilate": dilate,
        "background": background,
        "foreground": foreground,
        "length_target": length_target,
        "word_target": word_target,
    }


def make_sample(stage: str, seed: int, progress: float = 0.0) -> Tuple[GrayGrid, str, SampleMeta]:
    """Return a single (image, text) sample for the requested stage."""
    stage = stage.upper()
    if stage not in STAGE_SETTINGS:
        raise KeyError(f"Unknown stage {stage}")
    setting = STAGE_SETTINGS[stage]
    rng = Random((hash(stage) ^ seed) & 0xFFFFFFFF)
    params = _schedule_parameters(setting, rng, progress)
    text, length_info = _sample_text(
        stage,
        setting,
        rng,
        target_len=params["length_target"],
        word_target=params["word_target"],
    )
    _assert_valid_chars(text)
    render_seed = rng.randint(0, 2**31 - 1)
    scale = params["scale"]
    shear = params["shear"]
    dilate = params["dilate"]
    background = params["background"]
    foreground = params["foreground"]
    gray = _render_with_retries(
        text,
        setting,
        scale=scale,
        shear=shear,
        dilate=dilate,
        background=background,
        foreground=foreground,
        contrast=params["contrast"],
        noise=params["noise"],
        seed=render_seed,
    )
    word_count_meta = length_info.get("word_count")
    meta = SampleMeta(
        progress=progress,
        text_length=len(text.replace("\n", "")),
        word_count=word_count_meta,
        contrast=params["contrast"],
        noise=params["noise"],
        scale=scale,
        shear=shear,
        dilate=dilate,
        background=background,
        foreground=foreground,
        jitter=setting.jitter,
    )
    return gray, text, meta


def _render_with_retries(
    text: str,
    setting: StageSetting,
    *,
    scale: int,
    shear: int,
    dilate: int,
    background: int,
    foreground: int,
    contrast: float,
    noise: float,
    seed: int,
) -> GrayGrid:
    attempts = [
        (scale, shear, dilate),
        (max(1, scale - 1), shear // 2, max(0, dilate - 1)),
        (1, 0, 0),
    ]
    for sc, sh, di in attempts:
        try:
            return render.render_text(
                text,
                setting.width,
                setting.height,
                font=setting.font,
                jitter=setting.jitter,
                noise=noise,
                contrast=contrast,
                scale=max(1, sc),
                shear=sh,
                dilate=max(0, di),
                background=background,
                foreground=foreground,
                seed=seed,
            )
        except ValueError:
            continue
    # Final fallback without extra transforms
    return render.render_text(
        text,
        setting.width,
        setting.height,
        font=setting.font,
        jitter=0,
        noise=noise,
        contrast=contrast,
        scale=1,
        shear=0,
        dilate=0,
        background=background,
        foreground=foreground,
        seed=seed,
    )


def generate_dataset(
    stage: str,
    n: int,
    out_dir: Path,
    seed: int = 0,
    *,
    profile: bool = False,
) -> Dict[str, object]:
    """Generate a dataset with images/ and labels.jsonl for the given stage."""
    if n <= 0:
        raise ValueError("n must be positive")
    stage = stage.upper()
    if stage not in STAGE_SETTINGS:
        raise KeyError(f"Unknown stage {stage}")
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"
    master_rng = Random(seed)
    profiler = StageProfile(stage=stage)
    with labels_path.open("w", encoding="utf-8") as labels_file:
        for index in range(n):
            sample_seed = master_rng.randint(0, 2**31 - 1)
            progress = index / max(1, n - 1)
            gray, text, meta = make_sample(stage, seed=sample_seed, progress=progress)
            filename = f"{index + 1:05d}.pgm"
            pgm_path = images_dir / filename
            pgm.save_pgm(pgm_path, gray)
            record = {
                "file": f"images/{filename}",
                "text": text,
                "stage": stage,
                "seed": sample_seed,
                "meta": asdict(meta),
            }
            labels_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            profiler.update(meta)
    _verify_counts(images_dir, labels_path, n)
    summary = profiler.summary()
    setting = STAGE_SETTINGS[stage]
    if stage == "S3":
        avg_len = summary["avg_length"]
        if avg_len < setting.min_len - 0.5 or avg_len > setting.max_len + 0.5:
            raise RuntimeError(f"S3 average length {avg_len:.2f} outside [{setting.min_len}, {setting.max_len}]")
    if stage == "S4" and summary.get("avg_word_count") is not None and setting.word_count:
        avg_wc = summary["avg_word_count"]
        if avg_wc is not None and (avg_wc < setting.word_count[0] - 0.5 or avg_wc > setting.word_count[1] + 0.5):
            raise RuntimeError(
                f"S4 average word count {avg_wc:.2f} outside {setting.word_count}"
            )
    if profile:
        print(json.dumps({"stage": stage, "profile": summary}, ensure_ascii=False, indent=2))
    return summary


def _verify_counts(images_dir: Path, labels_path: Path, expected: int) -> None:
    image_count = sum(1 for _ in images_dir.glob("*.pgm"))
    if image_count != expected:
        raise RuntimeError(f"Expected {expected} images, found {image_count}")
    with labels_path.open("r", encoding="utf-8") as handle:
        label_lines = sum(1 for _ in handle)
    if label_lines != expected:
        raise RuntimeError(f"Expected {expected} labels, found {label_lines}")


def _self_check() -> None:
    for stage, setting in STAGE_SETTINGS.items():
        image, text = make_sample(stage, seed=123)
        assert len(image) == setting.height
        assert all(len(row) == setting.width for row in image)
        _assert_valid_chars(text)


def _demo(base: Path, n: int, seed: int) -> None:
    for stage in STAGE_SETTINGS:
        target = base / stage.lower()
        generate_dataset(stage, n, target, seed=seed)
        print(f"[demo] Generated {n} samples for {stage} at {target}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic curriculum data generator.")
    parser.add_argument("--stage", default="ALL", help="Stage name (S1-S4) or ALL")
    parser.add_argument("--n", type=int, default=10, help="Number of samples per stage")
    parser.add_argument("--out", type=Path, default=Path("data/synth"), help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    stage = args.stage.upper()
    if stage == "ALL":
        _demo(Path(args.out), args.n, args.seed)
        return
    target = Path(args.out)
    generate_dataset(stage, args.n, target, seed=args.seed)
    print(f"Generated {args.n} samples for {stage} at {target}")


if __name__ == "__main__":
    _self_check()
    main()
