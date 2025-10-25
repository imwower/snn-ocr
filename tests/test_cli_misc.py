"""Tests for CLI helper utilities (repro metadata)."""
from __future__ import annotations

import unittest
from pathlib import Path

from snn_ocr import cli


class ReproPayloadTest(unittest.TestCase):
    def test_build_repro_payload_injects_git_info(self) -> None:
        payload = cli.build_repro_payload(
            "S2",
            seed=11,
            data_dir=Path("/tmp/data"),
            ckpt=None,
            command="python -m snn_ocr.cli eval",
            notes="demo",
            git_info={"commit": "deadbeef", "describe": "v1", "dirty": False},
        )
        self.assertEqual(payload["config"]["stage"], "S2")
        self.assertEqual(payload["config"]["data"], "/tmp/data")
        self.assertEqual(payload["git"]["commit"], "deadbeef")
        self.assertFalse(payload["git"]["dirty"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
