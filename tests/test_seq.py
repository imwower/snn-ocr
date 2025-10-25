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

    def test_pointwise_preserves_shape_and_range(self) -> None:
        mixed = seq.pointwise_spike(self.sample)
        self.assertEqual(len(mixed[0][0]), len(self.sample[0][0]))
        for t_step in mixed:
            for column in t_step:
                for value in column:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_linear_attention_respects_shape(self) -> None:
        smoothed = seq.dwconv1d_spike(self.sample, k=3)
        mixed = seq.pointwise_spike(smoothed)
        attended = seq.linear_attention(mixed, heads=2, key_dim=2)
        self.assertEqual(len(attended), len(self.sample))
        self.assertEqual(len(attended[0]), len(self.sample[0]))
        self.assertEqual(len(attended[0][0]), len(self.sample[0][0]))

    def test_spiking_head_handles_attention_toggle(self) -> None:
        head_attn = seq.SpikingSeqHead(use_attention=True, heads=2)
        head_no_attn = seq.SpikingSeqHead(use_attention=False)
        out_attn = head_attn.forward(self.sample)
        out_no_attn = head_no_attn.forward(self.sample)
        self.assertEqual(len(out_attn), len(self.sample))
        self.assertEqual(len(out_no_attn[0]), len(self.sample[0]))
        self.assertEqual(len(out_attn[0][0]), len(self.sample[0][0]))
        self.assertEqual(len(out_no_attn[0][0]), len(self.sample[0][0]))

    def test_blockwise_merge_reduces_flat_regions(self) -> None:
        sequence = [[0.1, 0.1, 0.1, 0.1, float(idx)] for idx in range(10)]
        sequence[5] = [0.9, 0.2, 0.8, 0.1, 5.0]
        merged = seq.merge_columns_blockwise(sequence, max_merge=3, window=3)
        self.assertLess(len(merged), len(sequence))
        self.assertEqual(len(merged[0]), len(sequence[0]))


if __name__ == "__main__":
    unittest.main()
