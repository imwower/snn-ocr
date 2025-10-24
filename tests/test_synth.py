"""Unit tests for snn_ocr.synth dataset generation."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from snn_ocr import synth


class SynthTest(unittest.TestCase):
    """Validate sample synthesis across curriculum stages."""

    def test_make_sample_dimensions(self) -> None:
        for stage, setting in synth.STAGE_SETTINGS.items():
            image, text = synth.make_sample(stage, seed=123)
            self.assertEqual(len(image), setting.height)
            self.assertTrue(all(len(row) == setting.width for row in image))
            self.assertTrue(set(text).issubset(synth.ALLOWED_CHARS))

    def test_generate_dataset_integrity(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "s1_samples"
            synth.generate_dataset("S1", 3, out)
            images = sorted((out / "images").glob("*.pgm"))
            self.assertEqual(len(images), 3)
            labels_path = out / "labels.jsonl"
            lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            parsed = [json.loads(line) for line in lines]
            for entry in parsed:
                file_path = out / entry["file"]
                self.assertTrue(file_path.exists())
                self.assertTrue(set(entry["text"]).issubset(synth.ALLOWED_CHARS))


if __name__ == "__main__":
    unittest.main()
