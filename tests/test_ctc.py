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

    def test_ctc_loss_with_grad_matches_scalar(self) -> None:
        logits = [
            [2.0, 1.0, 0.1],
            [0.5, 3.0, 0.1],
            [0.2, 0.3, 2.5],
        ]
        target = [1, 2]
        loss = ctc.ctc_loss(logits, target, blank=0)
        loss_grad, grad = ctc.ctc_loss_with_grad(logits, target, blank=0)
        self.assertAlmostEqual(loss, loss_grad, places=6)
        self.assertEqual(len(grad), len(logits))
        self.assertEqual(len(grad[0]), len(logits[0]))

    def test_symbol_table_contains_punctuation(self) -> None:
        symbols = ctc.symbol_table()
        self.assertIn(".", symbols)
        self.assertIn("!", symbols)

    def test_ctc_loss_handles_empty_target(self) -> None:
        logits = [
            [5.0, 0.5],
            [4.5, 0.2],
            [5.2, 0.1],
        ]
        loss = ctc.ctc_loss(logits, [], blank=0)
        self.assertLess(loss, 0.05)

    def test_decoders_return_empty_for_all_blank(self) -> None:
        logits = [
            [4.0, 0.5, 0.2],
            [4.3, 0.1, 0.2],
            [4.1, 0.2, 0.1],
        ]
        self.assertEqual(ctc.greedy_decode(logits, blank=0), "")
        self.assertEqual(ctc.beam_search(logits, beam=2, blank=0), "")

    def test_repeated_characters_require_blank_gap(self) -> None:
        logits = [
            [0.2, 3.0, 0.1],
            [3.2, 0.5, 0.1],
            [0.2, 3.1, 0.1],
        ]
        decoded_greedy = ctc.greedy_decode(logits, blank=0)
        decoded_beam = ctc.beam_search(logits, beam=3, blank=0)
        self.assertEqual(decoded_greedy, "AA")
        self.assertEqual(decoded_beam, "AA")


if __name__ == "__main__":
    unittest.main()
