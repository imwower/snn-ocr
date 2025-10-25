"""Unit tests for spike encoding diagnostics."""
from __future__ import annotations

import unittest

from snn_ocr import spikes


class SpikeEncodingTest(unittest.TestCase):
    def test_ttfs_single_spike(self) -> None:
        gray = [[0, 128, 255]]
        tensor, stats = spikes.encode_ttfs(gray, T=6, return_stats=True)
        self.assertEqual(len(stats["histogram"]), 6)
        self.assertEqual(stats["total"], 2)
        spikes.assert_ttfs_single_spike(tensor)

    def test_poisson_intensity_monotonic(self) -> None:
        gray = [[32, 128, 224]]
        totals = [0, 0, 0]
        for seed in range(50):
            tensor = spikes.encode_poisson(gray, T=32, rate_scale=2.0, seed=seed)
            for y, row in enumerate(gray):
                for x, _ in enumerate(row):
                    totals[x] += sum(tensor[t][y][x] for t in range(len(tensor)))
        self.assertLess(totals[0], totals[1])
        self.assertLess(totals[1], totals[2])

    def test_random_micro_saccade_bounds(self) -> None:
        base = spikes.encode_ttfs([[255 for _ in range(4)]], T=4)
        shifted = spikes.apply_random_micro_saccade(base, jitter=1, seed=1)
        T, H, W = spikes._validate_spike_tensor(shifted)
        self.assertEqual((T, H, W), (4, 1, 4))


if __name__ == "__main__":
    unittest.main()
