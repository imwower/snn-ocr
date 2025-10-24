"""Unit tests for Optical Token Compressor."""
from __future__ import annotations

import unittest

from snn_ocr import otc


class OtcTest(unittest.TestCase):
    """Validate height compression and low-information merging."""

    def test_shape_and_order_preserved(self) -> None:
        features = [
            [
                [[float(w)] for w in range(3)],
                [[float(w + 1)] for w in range(3)],
            ]
        ]
        sequence = otc.compress_height(features, max_merge=1)
        self.assertEqual(len(sequence), 1)
        self.assertEqual(len(sequence[0]), 3)
        expected = [
            [(0.0 + 1.0) / 2],
            [(1.0 + 2.0) / 2],
            [(2.0 + 3.0) / 2],
        ]
        for column, vector in zip(sequence[0], expected):
            self.assertAlmostEqual(column[0], vector[0])

    def test_merging_low_information_columns(self) -> None:
        features = []
        for t in range(2):
            time_slice = []
            for h in range(2):
                row = [
                    [1.0 if (h + t) % 2 == 0 else 0.0],
                    [float(h)],
                    [0.05],
                    [0.05],
                ]
                time_slice.append(row)
            features.append(time_slice)
        sequence = otc.compress_height(features, gate="var", max_merge=2)
        self.assertLessEqual(len(sequence[0]), 3)
        merged_tail = sequence[0][-1]
        self.assertTrue(all(abs(value - 0.05) < 1e-6 for value in merged_tail))

    def test_entropy_gate(self) -> None:
        features = [
            [
                [[0.1, 0.2], [0.2, 0.1]],
                [[0.3, 0.4], [0.4, 0.3]],
            ]
        ]
        sequence = otc.compress_height(features, gate="entropy", max_merge=1)
        self.assertEqual(len(sequence), 1)
        self.assertEqual(len(sequence[0]), 2)
        heat = otc.ascii_heatmap([0.1, 0.5, 0.9])
        self.assertEqual(len(heat), 3)


if __name__ == "__main__":
    unittest.main()
