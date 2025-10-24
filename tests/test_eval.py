"""Unit tests for evaluation metrics and visualization helpers."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from snn_ocr import eval as eval_mod


class EvalMetricsTest(unittest.TestCase):
    """Validate CER/WER and dumping behaviour."""

    def test_cer_and_wer(self) -> None:
        self.assertAlmostEqual(eval_mod.cer("HELLO", "HELXO"), 1 / 5)
        self.assertAlmostEqual(eval_mod.cer("", ""), 0.0)
        self.assertAlmostEqual(eval_mod.cer("", "A"), 1.0)
        self.assertAlmostEqual(eval_mod.wer("HELLO WORLD", "HELLO"), 0.5)
        self.assertAlmostEqual(eval_mod.wer("", ""), 0.0)

    def test_dump_examples_creates_files(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "vis"
            record = eval_mod.ExampleRecord(
                path=Path("dummy.pgm"),
                text="HELLO",
                prediction="HELO",
                cer_value=0.2,
                wer_value=0.25,
                ascii_art="##\n##",
                alignment_ref="HELLO",
                alignment_hyp="HE-LO",
            )
            eval_mod.dump_examples([record], out_dir)
            output_files = list(out_dir.glob("example_*.txt"))
            self.assertEqual(len(output_files), 1)
            content = output_files[0].read_text(encoding="utf-8")
            self.assertIn("Reference: HELLO", content)
            self.assertIn("Hypothesis: HELO", content)


if __name__ == "__main__":
    unittest.main()
