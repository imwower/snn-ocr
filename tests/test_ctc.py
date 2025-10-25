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

    def test_prefix_beam_can_outperform_greedy(self) -> None:
        logits = [
            [3.3776874061001925, 3.03181761176121, 1.68228632332338, 1.0356670011718534],
            [2.045098885474434, 1.6197365498016572, 3.1351943561390905, 1.2132509043157098],
            [1.9063878166094232, 2.333528157820125, 3.6324515407813407, 2.018747423269561],
            [1.1273513775988153, 3.0232168166288957, 2.4734759867013265, 1.0020253654497622],
        ]
        greedy = ctc.greedy_decode(logits, blank=0)
        beam = ctc.beam_search(logits, beam=6, blank=0, len_norm=True)
        self.assertEqual(greedy, "BA")
        self.assertEqual(beam, "ABA")

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

    def test_ctc_loss_improves_with_additional_time(self) -> None:
        base = [
            [2.2992617160098856, 1.4970588555606719, 2.5621124428705824],
            [2.1703830030796274, 0.5028570482860707, 1.7339446661633116],
            [2.669006938731952, 1.1097771922178299, 1.3130109068684752],
        ]
        extended = base + [
            [2.6114136963259638, 0.5732012745071716, 1.7025322218620156],
        ]
        target = [1, 2]
        loss_base = ctc.ctc_loss(base, target, blank=0)
        loss_extended = ctc.ctc_loss(extended, target, blank=0)
        self.assertLessEqual(loss_extended, loss_base)

    def test_symbol_table_contains_punctuation(self) -> None:
        symbols = ctc.symbol_table()
        self.assertIn(".", symbols)
        self.assertIn("!", symbols)
        self.assertIn("\n", symbols)

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

    def test_best_alignment_path_handles_empty_target(self) -> None:
        logits = [[4.0, 0.5], [4.2, 0.4], [3.8, 0.3]]
        path = ctc.best_alignment_path(logits, [], blank=0)
        self.assertEqual(len(path), len(logits))
        self.assertTrue(all(index == 0 for index in path))

    def test_long_sequence_alignment_remains_stable(self) -> None:
        text = "AB" * 8
        logits = []
        for ch in text:
            if ch == "A":
                logits.append([0.1, 5.0, 0.1])
            else:
                logits.append([0.1, 0.1, 5.0])
            logits.append([5.0, 0.1, 0.1])  # encourage blank separation
        greedy = ctc.greedy_decode(logits, blank=0)
        beam = ctc.beam_search(logits, beam=4, blank=0)
        self.assertEqual(greedy, text)
        self.assertEqual(beam, text)
        target = [1 if ch == "A" else 2 for ch in text]
        path = ctc.best_alignment_path(logits, target, blank=0)
        self.assertEqual(len(path), len(logits))
        loss = ctc.ctc_loss(logits, target, blank=0)
        self.assertGreater(loss, 0.0)

    def test_repeated_alignment_contains_blanks(self) -> None:
        logits = [
            [0.3, 4.0, 0.2],
            [4.0, 0.1, 0.1],
            [0.2, 4.1, 0.2],
        ]
        target = [1, 1]
        path = ctc.best_alignment_path(logits, target, blank=0)
        self.assertIn(0, path)
        self.assertEqual(len(path), len(logits))


if __name__ == "__main__":
    unittest.main()
