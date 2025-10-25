"""Unit tests for spike encoders and channel splitting."""
from __future__ import annotations

import unittest

from snn_ocr import spikes


class SpikeEncodingTest(unittest.TestCase):
    """Validate TTFS, Poisson, and ON/OFF behaviours."""

    def test_ttfs_single_firing(self) -> None:
        gray = [
            [0, 64, 128, 192, 255],
            [255, 0, 0, 0, 255],
        ]
        tensor = spikes.encode_ttfs(gray, T=5)
        per_pixel_counts = [[0 for _ in row] for row in gray]
        for frame in tensor:
            for y, row in enumerate(frame):
                for x, value in enumerate(row):
                    per_pixel_counts[y][x] += value
        for row in per_pixel_counts:
            for count in row:
                self.assertLessEqual(count, 1)

    def test_poisson_intensity_monotonic(self) -> None:
        low = [[64] * 8 for _ in range(4)]
        high = [[192] * 8 for _ in range(4)]
        tensor_low = spikes.encode_poisson(low, T=64, rate_scale=0.8, seed=7)
        tensor_high = spikes.encode_poisson(high, T=64, rate_scale=0.8, seed=7)
        total_low = sum(sum(sum(row) for row in frame) for frame in tensor_low)
        total_high = sum(sum(sum(row) for row in frame) for frame in tensor_high)
        self.assertGreater(total_high, total_low)

    def test_split_on_off_shapes(self) -> None:
        gray = [[0, 0], [255, 255]]
        tensor = spikes.encode_ttfs(gray, T=4)
        on, off = spikes.split_on_off(tensor)
        self.assertEqual(len(on), len(off))
        self.assertEqual(len(on), len(tensor))
        self.assertEqual(len(on[0]), len(gray))
        self.assertEqual(len(on[0][0]), len(gray[0]))
        on_total = sum(sum(sum(row) for row in frame) for frame in on)
        off_total = sum(sum(sum(row) for row in frame) for frame in off)
        self.assertGreater(on_total, 0)
        self.assertEqual(off_total, on_total)

    def test_micro_saccade_shift(self) -> None:
        gray = [
            [0, 255],
            [0, 0],
        ]
        tensor = spikes.encode_ttfs(gray, T=2)
        shifted = spikes.apply_micro_saccade(tensor, dx=-1, dy=0)
        original_counts = [sum(sum(row) for row in frame) for frame in tensor]
        shifted_counts = [sum(sum(row) for row in frame) for frame in shifted]
        self.assertEqual(original_counts, shifted_counts)
        # Ensure spike moved left into column 0
        self.assertEqual(shifted[0][0][0], 1)


if __name__ == "__main__":
    unittest.main()
