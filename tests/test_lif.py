"""Unit tests for LIF neurons, spike convolutions, and residual blocks."""
from __future__ import annotations

import unittest
from random import Random

from snn_ocr import lif


class LifModuleTest(unittest.TestCase):
    """Validate neuron dynamics and helper utilities."""

    def test_lif_threshold_and_grad(self) -> None:
        neuron = lif.LIF(alpha=0.5, v_th=1.0, v_reset=0.1, surrogate_width=0.5)
        v0, r0 = neuron.step(0.4)
        self.assertLess(v0, 1.0)
        self.assertEqual(r0, 0)
        v1, r1 = neuron.step(1.0)
        self.assertEqual(r1, 1)
        self.assertGreaterEqual(v1, neuron.v_reset)
        self.assertGreater(neuron.grad(), 0.0)

    def test_conv2d_spike_dimensions(self) -> None:
        inputs = [
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        ]
        kernels = [
            [
                [
                    [0.1, 0.0, -0.1],
                    [0.0, 0.2, 0.0],
                    [-0.1, 0.0, 0.1],
                ]
            ]
        ]
        potentials, spikes = lif.conv2d_spike(
            inputs, kernels, stride=(1, 1), padding=(1, 1), lif_params=(0.9, 0.8, 0.0)
        )
        self.assertEqual(len(spikes), 1)
        self.assertEqual(len(spikes[0]), 3)
        self.assertEqual(len(spikes[0][0]), 3)
        self.assertTrue(all(potentials[0][0][i] >= 0.0 for i in range(3)))

    def test_conv1d_spike_basic(self) -> None:
        inputs = [
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
        ]
        kernels = [
            [
                [0.2, 0.2, 0.2],
                [-0.1, -0.1, -0.1],
            ]
        ]
        potentials, spikes = lif.conv1d_spike(
            inputs, kernels, stride=1, padding=1, lif_params=(0.9, 0.4, 0.0)
        )
        self.assertEqual(len(spikes), 1)
        self.assertEqual(len(spikes[0]), len(inputs[0]))
        self.assertTrue(any(value == 1.0 for value in spikes[0]))
        self.assertEqual(len(potentials[0]), len(spikes[0]))

    def test_residual_block_shape(self) -> None:
        rng = Random(7)
        inputs = [
            [[1.0 if rng.random() > 0.5 else 0.0 for _ in range(4)] for _ in range(4)],
            [[1.0 if rng.random() > 0.5 else 0.0 for _ in range(4)] for _ in range(4)],
        ]
        kernels1 = [
            [
                [
                    [0.05, 0.02, -0.01],
                    [0.02, 0.06, 0.02],
                    [-0.01, 0.02, 0.05],
                ],
                [
                    [0.01, -0.02, 0.01],
                    [-0.02, 0.04, -0.02],
                    [0.01, -0.02, 0.01],
                ],
            ],
            [
                [
                    [0.02, -0.01, 0.02],
                    [-0.01, 0.05, -0.01],
                    [0.02, -0.01, 0.02],
                ],
                [
                    [0.03, 0.03, 0.03],
                    [0.0, 0.03, 0.0],
                    [-0.03, -0.03, -0.03],
                ],
            ],
        ]
        kernels2 = [
            [
                [
                    [0.04, 0.01, 0.04],
                    [0.01, 0.05, 0.01],
                    [0.04, 0.01, 0.04],
                ],
                [
                    [-0.02, 0.0, 0.02],
                    [0.0, -0.01, 0.0],
                    [0.02, 0.0, -0.02],
                ],
            ],
            [
                [
                    [-0.05, 0.02, -0.05],
                    [0.02, 0.05, 0.02],
                    [-0.05, 0.02, -0.05],
                ],
                [
                    [0.03, 0.0, 0.03],
                    [0.0, 0.04, 0.0],
                    [0.03, 0.0, 0.03],
                ],
            ],
        ]
        residual, spikes = lif.residual_block(
            inputs,
            kernels1,
            kernels2,
            padding=(1, 1),
            lif_params=(0.9, 0.6, 0.0),
        )
        self.assertEqual(len(residual), len(inputs))
        self.assertEqual(len(spikes), len(inputs))
        for c in range(len(inputs)):
            self.assertEqual(len(residual[c]), len(inputs[c]))
            self.assertEqual(len(spikes[c]), len(inputs[c]))
            for row in residual[c]:
                self.assertTrue(all(0.0 <= value <= 1.0 for value in row))
            for row in spikes[c]:
                self.assertTrue(all(value in (0.0, 1.0) for value in row))

    def test_temporal_weighting_modes(self) -> None:
        losses = [0.2, 0.3, 0.5]
        uniform = lif.temporal_weighting(losses, mode="uniform")
        tail = lif.temporal_weighting(losses, mode="tail")
        self.assertAlmostEqual(uniform, sum(losses) / len(losses))
        self.assertGreater(tail, uniform)
        with self.assertRaises(ValueError):
            lif.temporal_weighting([], mode="uniform")

    def test_surrogate_grad_triangle(self) -> None:
        center = lif.surrogate_grad(0.0, width=0.5)
        self.assertGreater(center, 0.0)
        self.assertEqual(lif.surrogate_grad(0.5, width=0.5), 0.0)
        self.assertAlmostEqual(center, lif.surrogate_grad(-0.0, width=0.5))
        self.assertLess(lif.surrogate_grad(0.25, width=0.5), center)


if __name__ == "__main__":
    unittest.main()
