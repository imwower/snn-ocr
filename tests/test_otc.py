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

    def test_width_after_compression_never_increases(self) -> None:
        width = 6
        features: otc.FeatureTensor = []
        row = [[[float(w), float(w + 1)] for w in range(width)]]
        features.append(row)
        sequence, shape, log = otc.compress_height(
            features,
            target_h=1,
            gate="var",
            max_merge=2,
            strategy="threshold",
        )
        self.assertLessEqual(shape[1], width)
        self.assertEqual(len(sequence[0]), shape[1])
        column_map = log[0].get("column_map")
        self.assertIsNotNone(column_map)
        column_map = list(column_map)  # type: ignore[assignment]
        self.assertEqual(len(column_map), width)
        for i in range(width - 1):
            self.assertLessEqual(column_map[i], column_map[i + 1])
        self.assertEqual(max(column_map) + 1, shape[1])

    def test_dp_strategy_preserves_column_order(self) -> None:
        width = 5
        row = [[[float(w)] for w in range(width)]]
        features = [row]
        _, shape, log = otc.compress_height(
            features,
            target_h=1,
            gate="var",
            target_width=3,
            strategy="dp",
        )
        column_map = log[0].get("column_map")
        column_map = list(column_map)  # type: ignore[assignment]
        self.assertEqual(len(column_map), width)
        self.assertEqual(sorted(set(column_map)), list(range(shape[1])))


if __name__ == "__main__":
    unittest.main()
