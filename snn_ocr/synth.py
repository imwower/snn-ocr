"""Curriculum-aware dataset synthesis utilities for the SNN-OCR project."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Tuple

from snn_ocr import pgm, render

GrayGrid = List[List[int]]


@dataclass(frozen=True)
class StageSetting:
    width: int
    height: int
    jitter: int
    noise: float
    contrast: float
    font: str


STAGE_SETTINGS: Dict[str, StageSetting] = {
    "S1": StageSetting(width=28, height=28, jitter=0, noise=0.0, contrast=1.0, font="5x7"),
    "S2": StageSetting(width=32, height=32, jitter=1, noise=0.01, contrast=1.0, font="5x7"),
    "S3": StageSetting(width=128, height=32, jitter=1, noise=0.05, contrast=1.1, font="5x7"),
    "S4": StageSetting(width=384, height=32, jitter=2, noise=0.08, contrast=1.1, font="5x7"),
}

DIGITS = "0123456789"
LETTERS_UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS_LO = LETTERS_UP.lower()
PUNCT = ".,!?"
ALLOWED_CHARS = set(DIGITS + LETTERS_UP + LETTERS_LO + PUNCT + " ")


def _random_word(rng: Random, length: int) -> str:
    return "".join(rng.choice(LETTERS_LO) for _ in range(length))


def _sample_text(stage: str, rng: Random) -> str:
    if stage == "S1":
        return rng.choice(DIGITS)
    if stage == "S2":
        pool = LETTERS_UP + LETTERS_LO
        return rng.choice(pool)
    if stage == "S3":
        length = rng.randint(3, 10)
        return _random_word(rng, length)
    if stage == "S4":
        for _ in range(10):
            word_count = rng.randint(3, 8)
            words = [_random_word(rng, rng.randint(2, 8)) for _ in range(word_count)]
            sentence = " ".join(words)
            sentence = sentence.capitalize()
            if rng.random() < 0.7:
                sentence += rng.choice(PUNCT)
            if 15 <= len(sentence) <= 60:
                return sentence
        fallback = "learn spiking networks!"
        return fallback.capitalize()
    raise KeyError(f"Unsupported stage {stage!r}")


def make_sample(stage: str, seed: int) -> Tuple[GrayGrid, str]:
    """Return a single (image, text) sample for the requested stage."""
    stage = stage.upper()
    if stage not in STAGE_SETTINGS:
        raise KeyError(f"Unknown stage {stage}")
    rng = Random((hash(stage) ^ seed) & 0xFFFFFFFF)
    text = _sample_text(stage, rng)
    setting = STAGE_SETTINGS[stage]
    gray = render.render_text(
        text,
        setting.width,
        setting.height,
        font=setting.font,
        jitter=setting.jitter,
        noise=setting.noise,
        contrast=setting.contrast,
    )
    _assert_valid_chars(text)
    return gray, text


def _assert_valid_chars(text: str) -> None:
    if not set(text).issubset(ALLOWED_CHARS):
        raise ValueError("Text contains unsupported characters")


def generate_dataset(stage: str, n: int, out_dir: Path) -> None:
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
    with labels_path.open("w", encoding="utf-8") as labels_file:
        for index in range(n):
            gray, text = make_sample(stage, seed=index + 1)
            filename = f"{index + 1:04d}.pgm"
            pgm_path = images_dir / filename
            pgm.save_pgm(pgm_path, gray)
            record = {"file": f"images/{filename}", "text": text}
            labels_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    image_count = sum(1 for _ in images_dir.glob("*.pgm"))
    assert image_count == n
    with labels_path.open("r", encoding="utf-8") as handle:
        label_lines = sum(1 for _ in handle)
    assert label_lines == n


def _self_check() -> None:
    for stage in STAGE_SETTINGS:
        sample, text = make_sample(stage, 1)
        setting = STAGE_SETTINGS[stage]
        assert len(sample) == setting.height
        assert all(len(row) == setting.width for row in sample)
        _assert_valid_chars(text)


def _demo() -> None:
    base = Path("demo_synth")
    base.mkdir(exist_ok=True)
    for stage in STAGE_SETTINGS:
        target = base / stage.lower()
        generate_dataset(stage, 10, target)
        print(f"Generated {stage} samples at {target}")


if __name__ == "__main__":
    _self_check()
    _demo()
