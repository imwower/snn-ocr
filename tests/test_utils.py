"""Unit tests for utility helpers."""
from __future__ import annotations

import io
import json
import random
import time
import unittest
from pathlib import Path

from snn_ocr import utils


class UtilsTest(unittest.TestCase):
    """Validate progress bars, timers, and RNG helpers."""

    def test_progress_bar_bounds(self) -> None:
        bar = utils.progress_bar(5, 10, length=10, label="demo")
        self.assertIn("[#####-----]", bar)
        self.assertIn("demo", bar)

    def test_simple_tqdm_prints_each_step(self) -> None:
        buffer = io.StringIO()
        steps = list(utils.simple_tqdm(3, label="demo", stream=buffer))
        output = buffer.getvalue().strip().splitlines()
        self.assertEqual(steps, [0, 1, 2])
        self.assertTrue(all("demo" in line for line in output))
        # Ensure no carriage return control characters are present
        self.assertTrue(all("\r" not in line for line in output))

    def test_timer_records_elapsed(self) -> None:
        timer = utils.timer("demo", stream=io.StringIO())
        with timer:
            time.sleep(0.01)
        self.assertGreaterEqual(timer.elapsed, 0.0)

    def test_seed_everything_reproducible(self) -> None:
        utils.seed_everything(7)
        seq1 = [random.random() for _ in range(3)]
        utils.seed_everything(7)
        seq2 = [random.random() for _ in range(3)]
        self.assertEqual(seq1, seq2)

    def test_append_jsonl(self) -> None:
        path = Path("tmp_utils_test.jsonl")
        try:
            utils.append_jsonl(path, {"a": 1})
            utils.append_jsonl(path, {"b": 2})
            with path.open("r", encoding="utf-8") as handle:
                lines = [json.loads(line) for line in handle]
            self.assertEqual(lines[0]["a"], 1)
            self.assertEqual(lines[1]["b"], 2)
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
