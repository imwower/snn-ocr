"""Tests for curriculum data synthesis scheduler."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from snn_ocr import synth


class SynthSchedulerTest(unittest.TestCase):
    def test_labels_include_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)
            synth.generate_dataset("S1", 3, out, seed=1, profile=False)
            labels = out / "labels.jsonl"
            with labels.open("r", encoding="utf-8") as handle:
                line = handle.readline()
            record = json.loads(line)
            self.assertIn("meta", record)
            self.assertIn("text_length", record["meta"])

    def test_profile_summary_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = synth.generate_dataset("S3", 12, Path(tmp_dir), seed=2, profile=False)
        self.assertGreater(summary["avg_length"], 2.0)
        self.assertLess(summary["avg_length"], 9.0)


if __name__ == "__main__":
    unittest.main()
