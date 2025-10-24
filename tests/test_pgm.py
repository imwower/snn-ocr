"""Unit tests for snn_ocr.pgm Netpbm helpers."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from snn_ocr import pgm


class PgmIoTest(unittest.TestCase):
    """Validate PGM/PPM read/write behaviour."""

    def test_save_and_load_pgm(self) -> None:
        gray = [
            [0, 255, 0, 255],
            [255, 0, 255, 0],
            [0, 255, 0, 255],
        ]
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test.pgm"
            pgm.save_pgm(path, gray)
            raw = path.read_bytes()
            header_parts = raw.split(b"\n", 3)
            self.assertEqual(header_parts[0], b"P5")
            self.assertEqual(header_parts[1], b"4 3")
            self.assertEqual(header_parts[2], b"255")
            self.assertEqual(len(header_parts[3]), 12)
            loaded = pgm.load_pgm(path)
            self.assertEqual(loaded, gray)

    def test_save_ppm_header(self) -> None:
        rgb = [
            [(255, 0, 0), (0, 255, 0)],
            [(0, 0, 255), (255, 255, 255)],
        ]
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test.ppm"
            pgm.save_ppm(path, rgb)
            raw = path.read_bytes()
            header_parts = raw.split(b"\n", 3)
            self.assertEqual(header_parts[0], b"P6")
            self.assertEqual(header_parts[1], b"2 2")
            self.assertEqual(header_parts[2], b"255")
            self.assertEqual(len(header_parts[3]), 12)

    def test_invalid_gray_value_raises(self) -> None:
        gray = [[-1]]
        with self.assertRaises(ValueError):
            pgm.save_pgm(Path("ignored.pgm"), gray)


if __name__ == "__main__":
    unittest.main()
