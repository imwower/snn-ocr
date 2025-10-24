"""Unit tests for pure-Python CTC implementation."""
from __future__ import annotations

import unittest

from snn_ocr import ctc


class CtcTest(unittest.TestCase):
    """Validate CTC loss and decoding utilities."""

    def test_ctc_loss_monotonicity(self) -> None:
        logits_low = [
            [1.0, 1.0, 0.0],
            [1.0, 0.5, 0.5],
            [1.0, 0.2, 0.8],
        ]
        logits_high = [
            [0.5, 2.0, 0.0],
            [0.5, 2.0, 0.0],
            [0.5, 0.2, 2.5],
        ]
        target = [1, 2]
        loss_low = ctc.ctc_loss(logits_low, target, blank=0)
        loss_high = ctc.ctc_loss(logits_high, target, blank=0)
        self.assertGreater(loss_low, loss_high)

    def test_greedy_decode_collapses(self) -> None:
        logits = [
            [0.1, 2.0, 0.1],
            [3.0, 0.1, 0.2],
            [3.0, 0.1, 0.2],
            [0.1, 0.2, 2.5],
        ]
        decoded = ctc.greedy_decode(logits, blank=0)
        self.assertEqual(decoded, "AB")

    def test_beam_search_matches_target(self) -> None:
        logits = [
            [0.3, 2.0, 0.2],
            [2.5, 0.4, 0.6],
            [0.2, 0.4, 2.5],
        ]
        greedy = ctc.greedy_decode(logits, blank=0)
        beam = ctc.beam_search(logits, beam=3, blank=0)
        self.assertEqual(greedy, "AB")
        self.assertEqual(beam, "AB")


if __name__ == "__main__":
    unittest.main()
