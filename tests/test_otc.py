"""Unit tests for OTC dynamic programming compression."""
from __future__ import annotations

import unittest

from snn_ocr import otc


class OtcTest(unittest.TestCase):
    def test_dp_segments_cover_width(self) -> None:
        info = [0.1, 0.8, 0.2, 0.4, 0.9]
        segments, curve = otc._dp_optimal_segments(info, target_width=3)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], len(info) - 1)
        self.assertTrue(curve)
        self.assertEqual(curve[-1]["segments"], 3.0)

    def test_compress_height_returns_column_map(self) -> None:
        values = [0.0, 0.05, 0.2, 0.9]
        row = [[val, val + 0.01] for val in values]
        features = [[row]]
        sequence, shape, log = otc.compress_height(
            features,
            target_h=1,
            gate="var",
            target_width=2,
            strategy="dp",
        )
        self.assertEqual(shape[1], 2)
        self.assertEqual(len(sequence[0]), 2)
        self.assertTrue(log)
        column_map = log[0].get("column_map")
        self.assertIsInstance(column_map, list)
        self.assertEqual(len(column_map), len(values))
        self.assertEqual(sorted(set(column_map)), [0, 1])


if __name__ == "__main__":
    unittest.main()
