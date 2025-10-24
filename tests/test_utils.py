"""Unit tests for utility helpers."""
from __future__ import annotations

import time
import unittest

from snn_ocr import utils


class UtilsTest(unittest.TestCase):
    """Validate progress bars, timers, and RNG helpers."""

    def test_progress_bar_bounds(self) -> None:
        bar = utils.progress_bar(5, 10, length=10, label="demo")
        self.assertIn("[#####-----]", bar)
        self.assertIn("demo", bar)

    def test_timer_records_elapsed(self) -> None:
        timer = utils.Timer()
        with timer:
            time.sleep(0.01)
        self.assertGreaterEqual(timer.elapsed, 0.0)

    def test_make_rng_reproducible(self) -> None:
        rng1 = utils.make_rng(42)
        rng2 = utils.make_rng(42)
        seq1 = [rng1.random() for _ in range(3)]
        seq2 = [rng2.random() for _ in range(3)]
        self.assertEqual(seq1, seq2)


if __name__ == "__main__":
    unittest.main()
