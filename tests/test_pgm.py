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
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid.pgm"
            with self.assertRaises(ValueError):
                pgm.save_pgm(path, gray)

    def test_invalid_header_raises(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "broken.pgm"
            path.write_bytes(b"P9\n1 1\n255\n\x00")
            with self.assertRaises(ValueError):
                pgm.load_pgm(path)

    def test_load_pgm_tolerates_comments_and_spaces(self) -> None:
        pattern = [
            [0, 255],
            [255, 0],
        ]
        payload = bytes(value for row in pattern for value in row)
        header = b"P5\n# comment about image\n 2 2  \n255\n"
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "comment.pgm"
            path.write_bytes(header + payload)
            loaded = pgm.load_pgm(path)
            self.assertEqual(loaded, pattern)


if __name__ == "__main__":
    unittest.main()
