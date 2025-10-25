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

    def test_dump_examples_writes_summary_when_metrics_passed(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "vis"
            record = eval_mod.ExampleRecord(
                path=Path("sample.pgm"),
                text="ABC",
                prediction="ADC",
                cer_value=1 / 3,
                wer_value=0.5,
                ascii_art="@@",
                alignment_ref="ABC",
                alignment_hyp="ADC",
                alignment_marks="^|^",
            )
            metrics = eval_mod.EvalMetrics(
                top1=0.0,
                cer=0.5,
                wer=0.5,
                avg_fire_rate=0.1,
                avg_width=12.0,
                samples=1,
                energy_total=10.0,
                energy_per_pixel=0.5,
                duty_cycle=[0.1, 0.2],
                blank_ratio=0.75,
            )
            eval_mod.dump_examples([record], out_dir, metrics)
            summary_path = out_dir / "report.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["blank_ratio"], 0.75)
            self.assertEqual(payload["examples"], ["example_001.txt"])

    def test_profile_single_example_returns_stats(self) -> None:
        with TemporaryDirectory() as tmp:
            stage = "S1"
            data_dir = Path(tmp) / "profile"
            eval_mod.ensure_dataset(stage, data_dir, size=2)
            stats = eval_mod.profile_single_example(stage, data_dir)
            self.assertEqual(stats["stage"], stage)
            self.assertIn("inference_ms", stats)
            self.assertIn("spike_hist", stats)


if __name__ == "__main__":
    unittest.main()
