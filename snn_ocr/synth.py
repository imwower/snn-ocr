"""Curriculum-aware dataset synthesis utilities for the SNN-OCR project."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Sequence, Tuple

from snn_ocr import pgm, render

GrayGrid = List[List[int]]

DIGITS = "0123456789"
LETTERS_UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS_LO = LETTERS_UP.lower()
PUNCT = ".,!?"
ALLOWED_CHARS = set(DIGITS + LETTERS_UP + LETTERS_LO + PUNCT + " ")


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
    ),
}


def _rand_int(rng: Random, bounds: Tuple[int, int]) -> int:
    low, high = bounds
    if low > high:
        low, high = high, low
    return rng.randint(low, high)


def _random_word(rng: Random, length: int, pool: str) -> str:
    return "".join(rng.choice(pool) for _ in range(length))


def _sample_text(stage: str, setting: StageSetting, rng: Random) -> str:
    mode = setting.mode
    if mode in ("digit", "char"):
        length = _rand_int(rng, (setting.min_len, setting.max_len))
        token = "".join(rng.choice(setting.char_pool) for _ in range(length))
        if mode == "char" and rng.random() < 0.3:
            token = token.upper()
        return token
    if mode == "word":
        length = _rand_int(rng, (setting.min_len, setting.max_len))
        word = _random_word(rng, length, setting.char_pool)
        if rng.random() < 0.4:
            word = word.upper()
        return word
    if mode == "sentence":
        if not setting.word_count:
            raise ValueError("Sentence mode requires word_count range")
        count = _rand_int(rng, setting.word_count)
        words: List[str] = []
        for _ in range(count):
            length = _rand_int(rng, (setting.min_len, setting.max_len))
            word = _random_word(rng, length, setting.char_pool)
            if rng.random() < 0.2:
                word = word.upper()
            words.append(word)
        sentence = " ".join(words)
        sentence = sentence.capitalize()
        if setting.punctuation and rng.random() < setting.punctuation_prob:
            sentence += rng.choice(setting.punctuation)
        return sentence
    raise KeyError(f"Unsupported stage mode {mode!r}")


def _assert_valid_chars(text: str) -> None:
    if not set(text).issubset(ALLOWED_CHARS):
        raise ValueError(f"Text contains unsupported characters: {text}")


def make_sample(stage: str, seed: int) -> Tuple[GrayGrid, str]:
    """Return a single (image, text) sample for the requested stage."""
    stage = stage.upper()
    if stage not in STAGE_SETTINGS:
        raise KeyError(f"Unknown stage {stage}")
    setting = STAGE_SETTINGS[stage]
    rng = Random((hash(stage) ^ seed) & 0xFFFFFFFF)
    text = _sample_text(stage, setting, rng)
    _assert_valid_chars(text)
    render_seed = rng.randint(0, 2**31 - 1)
    scale = _rand_int(rng, setting.scale_range)
    shear = _rand_int(rng, setting.shear_range)
    dilate = max(0, _rand_int(rng, setting.dilate_range))
    background = _rand_int(rng, setting.background_range)
    foreground = max(background + 1, _rand_int(rng, setting.foreground_range))
    gray = _render_with_retries(
        text,
        setting,
        scale=scale,
        shear=shear,
        dilate=dilate,
        background=background,
        foreground=foreground,
        seed=render_seed,
    )
    return gray, text


def _render_with_retries(
    text: str,
    setting: StageSetting,
    *,
    scale: int,
    shear: int,
    dilate: int,
    background: int,
    foreground: int,
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
                noise=setting.noise,
                contrast=setting.contrast,
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
        noise=setting.noise,
        contrast=setting.contrast,
        scale=1,
        shear=0,
        dilate=0,
        background=background,
        foreground=foreground,
        seed=seed,
    )


def generate_dataset(stage: str, n: int, out_dir: Path, seed: int = 0) -> None:
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
    with labels_path.open("w", encoding="utf-8") as labels_file:
        for index in range(n):
            sample_seed = master_rng.randint(0, 2**31 - 1)
            gray, text = make_sample(stage, seed=sample_seed)
            filename = f"{index + 1:05d}.pgm"
            pgm_path = images_dir / filename
            pgm.save_pgm(pgm_path, gray)
            record = {
                "file": f"images/{filename}",
                "text": text,
                "stage": stage,
                "seed": sample_seed,
            }
            labels_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    _verify_counts(images_dir, labels_path, n)


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
