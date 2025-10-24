"""Unit tests for LIF neurons, convolutions, and residual blocks."""
from __future__ import annotations

import unittest
from random import Random

from snn_ocr import lif


class LifModuleTest(unittest.TestCase):
    """Validate neuron dynamics and helper utilities."""

    def test_lif_threshold(self) -> None:
        neuron = lif.LIF(alpha=0.5, v_th=1.0, v_reset=0.1)
        v0, r0 = neuron.step(0.4)
        self.assertLess(v0, 1.0)
        self.assertEqual(r0, 0)
        v1, r1 = neuron.step(1.0)
        self.assertEqual(r1, 1)
        self.assertGreaterEqual(v1, neuron.v_reset)

    def test_conv2d_spike_dimensions(self) -> None:
        inputs = [[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]
        kernels = [[[[1.0, 0.0], [0.0, -1.0]]]]
        output = lif.conv2d_spike(inputs, kernels, stride=(1, 1), padding=(0, 0))
        self.assertEqual(len(output), 1)
        self.assertEqual(len(output[0]), 2)
        self.assertEqual(len(output[0][0]), 2)

    def test_residual_block_shape(self) -> None:
        rng = Random(7)
        inputs = [
            [[1.0 if rng.random() > 0.5 else 0.0 for _ in range(4)] for _ in range(4)],
            [[1.0 if rng.random() > 0.5 else 0.0 for _ in range(4)] for _ in range(4)],
        ]
        kernels1 = [
            [
                [[0.1, -0.1], [0.1, -0.1]],
                [[0.05, 0.05], [0.05, 0.05]],
            ],
            [
                [[0.0, 0.1], [0.1, 0.0]],
                [[-0.05, 0.05], [-0.05, 0.05]],
            ],
        ]
        kernels2 = [
            [
                [[0.1, 0.0], [0.0, 0.1]],
                [[0.05, -0.05], [0.05, -0.05]],
            ],
            [
                [[-0.1, 0.0], [0.0, -0.1]],
                [[0.05, 0.05], [0.05, 0.05]],
            ],
        ]
        residual, spikes = lif.residual_block(
            inputs,
            kernels1,
            kernels2,
            lif_params=(0.9, 0.6, 0.0),
        )
        self.assertEqual(len(residual), len(inputs))
        self.assertEqual(len(spikes), len(inputs))
        for channel in residual:
            for row in channel:
                self.assertTrue(all(0.0 <= value <= 1.0 for value in row))
        for channel in spikes:
            for row in channel:
                self.assertTrue(all(value in (0.0, 1.0) for value in row))

    def test_temporal_weighting_modes(self) -> None:
        losses = [0.2, 0.3, 0.5]
        uniform = lif.temporal_weighting(losses, mode="uniform")
        tail = lif.temporal_weighting(losses, mode="tail")
        self.assertAlmostEqual(uniform, sum(losses) / len(losses))
        self.assertGreater(tail, uniform)


if __name__ == "__main__":
    unittest.main()
