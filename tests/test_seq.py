"""Unit tests for sequence head layers."""
from __future__ import annotations

import unittest

from snn_ocr import seq


class SequenceHeadTest(unittest.TestCase):
    """Validate depthwise convolution, pointwise mixing, and attention."""

    def setUp(self) -> None:
        self.sample = [
            [[float(w + c) for c in range(3)] for w in range(5)],
            [[float(w - c) for c in range(3)] for w in range(5)],
        ]

    def test_dwconv_preserves_shape(self) -> None:
        smoothed = seq.dwconv1d_spike(self.sample, k=3)
        self.assertEqual(len(smoothed), len(self.sample))
        self.assertEqual(len(smoothed[0]), len(self.sample[0]))
        self.assertEqual(len(smoothed[0][0]), len(self.sample[0][0]))

    def test_pointwise_changes_channels(self) -> None:
        mixed = seq.pointwise_spike(self.sample, out_channels=5)
        self.assertEqual(len(mixed[0][0]), 5)

    def test_linear_attention_respects_shape(self) -> None:
        smoothed = seq.dwconv1d_spike(self.sample, k=3)
        mixed = seq.pointwise_spike(smoothed, out_channels=4)
        attended = seq.linear_attention(mixed, heads=2)
        self.assertEqual(len(attended), len(self.sample))
        self.assertEqual(len(attended[0]), len(self.sample[0]))
        self.assertEqual(len(attended[0][0]), 4)


if __name__ == "__main__":
    unittest.main()
