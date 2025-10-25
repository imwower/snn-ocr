"""Utility helpers for logging, timing, reproducibility, and progress reporting."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random, seed as global_seed
from time import perf_counter
from typing import Dict, Generator, Iterable, TextIO


def progress_bar(current: int, total: int, *, length: int = 30, label: str = "") -> str:
    """Return a textual progress bar string without control characters."""
    total = max(total, 1)
    ratio = max(0.0, min(1.0, current / total))
    filled = int(length * ratio)
    bar = "#" * filled + "-" * (length - filled)
    prefix = f"{label} " if label else ""
    return f"{prefix}[{bar}] {current}/{total}"


def simple_tqdm(
    total: int,
    *,
    label: str = "",
    every: int = 1,
    stream: TextIO | None = None,
) -> Generator[int, None, None]:
    """Yield indices while printing a newline progress bar (never overwriting logs)."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if stream is None:
        stream = sys.stdout
    total = int(total)
    step_interval = max(1, int(every))
    for idx in range(total):
        current = idx + 1
        if current % step_interval == 0 or current == total:
            print(progress_bar(current, total, label=label), file=stream)
        yield idx


@dataclass
class Timer:
    """Context manager for timing code blocks."""

    label: str = ""
    stream: TextIO | None = None

    def __post_init__(self) -> None:
        if self.stream is None:
            self.stream = sys.stdout
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = perf_counter() - self.start
        if self.label:
            print(f"{self.label}: {self.elapsed:.3f}s", file=self.stream)


def timer(label: str = "", stream: TextIO | None = None) -> Timer:
    """Helper so callers can use ``with timer("name")``."""
    return Timer(label=label, stream=stream)


def make_rng(seed: int | None = None) -> Random:
    """Create a seeded Random instance."""
    rng = Random()
    if seed is not None:
        rng.seed(seed)
    return rng


def seed_everything(seed: int) -> Random:
    """Seed the global RNG state and return a dedicated Random instance."""
    global_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return make_rng(seed)


def ensure_dir(path: Path) -> None:
    """Create the directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    """Append a dictionary as one JSON line, creating parent directories when needed."""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
