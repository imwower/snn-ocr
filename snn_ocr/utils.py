"""Utility helpers for logging, timing, and reproducible randomness."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from random import Random
from time import perf_counter
from pathlib import Path


def progress_bar(current: int, total: int, *, length: int = 30, label: str = "") -> str:
    """Return a textual progress bar string."""
    total = max(total, 1)
    ratio = max(0.0, min(1.0, current / total))
    filled = int(length * ratio)
    bar = "#" * filled + "-" * (length - filled)
    prefix = f"{label} " if label else ""
    return f"{prefix}[{bar}] {current}/{total}"


@dataclass
class Timer:
    """Context manager for timing code blocks."""

    label: str = ""
    stream: object = None

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


def make_rng(seed: int | None = None) -> Random:
    """Create a seeded Random instance."""
    rng = Random()
    if seed is not None:
        rng.seed(seed)
    return rng


def ensure_dir(path: Path) -> None:
    """Create the directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
