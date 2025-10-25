"""Unit tests for replay buffers and KD helpers in train.py."""
from __future__ import annotations

import unittest
from random import Random

from snn_ocr import train


class TrainHelperTest(unittest.TestCase):
    def test_replay_buffer_capacity_and_sampling(self) -> None:
        buffer = train.ReplayBuffer(capacity=2)
        sample1 = train.TrainSample(stage="S1", gray=[[0]], text="0")
        sample2 = train.TrainSample(stage="S1", gray=[[1]], text="1")
        sample3 = train.TrainSample(stage="S1", gray=[[2]], text="2")
        buffer.add(sample1)
        buffer.add(sample2)
        buffer.add(sample3)
        self.assertEqual(len(buffer), 2)
        rng = Random(42)
        seen = {buffer.sample(rng).text for _ in range(10)}
        self.assertTrue("1" in seen and "2" in seen)
        self.assertNotIn("0", seen)

    def test_kd_loss_zero_when_predictions_match(self) -> None:
        student = [[0.1, 0.5, -0.2], [0.0, 0.2, 0.8]]
        teacher = [[0.1, 0.5, -0.2], [0.0, 0.2, 0.8]]
        loss, grads = train.knowledge_distillation_loss(student, teacher, temperature=1.5)
        self.assertAlmostEqual(loss, 0.0, places=6)
        for row in grads:
            for value in row:
                self.assertAlmostEqual(value, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
