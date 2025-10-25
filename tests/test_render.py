"""Unit tests for snn_ocr.render text rendering."""
from __future__ import annotations

import unittest

from snn_ocr import render


class RenderTest(unittest.TestCase):
    """Verify rendering dimensions and stochastic features."""

    def test_render_dimensions(self) -> None:
        canvas = render.render_text("HELLO", 128, 32, jitter=0, noise=0.0, contrast=1.0)
        self.assertEqual(len(canvas), 32)
        self.assertTrue(all(len(row) == 128 for row in canvas))
        active = sum(value for row in canvas for value in row)
        self.assertGreater(active, 0)

    def test_noise_changes_pixels(self) -> None:
        clean = render.render_text("HI", 64, 16, jitter=0, noise=0.0, contrast=1.0)
        noisy = render.render_text("HI", 64, 16, jitter=0, noise=0.5, contrast=1.0)
        clean_values = sum(cell for row in clean for cell in row)
        noisy_values = sum(cell for row in noisy for cell in row)
        self.assertNotEqual(clean_values, noisy_values)

    def test_transforms_and_background(self) -> None:
        canvas = render.render_text(
            "OK",
            96,
            32,
            scale=2,
            shear=2,
            dilate=1,
            background=32,
            foreground=220,
            jitter=0,
            noise=0.0,
        )
        self.assertEqual(len(canvas), 32)
        self.assertTrue(all(len(row) == 96 for row in canvas))
        minimum = min(min(row) for row in canvas)
        self.assertEqual(minimum, 32)

    def test_ascii_preview(self) -> None:
        canvas = [[0, 255], [255, 0]]
        preview = render.ascii_preview(canvas)
        self.assertEqual(preview, " @\n@ ")

    def test_multiline_rendering(self) -> None:
        canvas = render.render_text("LINE1\nLINE2", 160, 64, jitter=0, noise=0.0)
        midpoint = max(1, len(canvas) // 2)
        top_active = any(any(value > 0 for value in row) for row in canvas[:midpoint])
        bottom_active = any(any(value > 0 for value in row) for row in canvas[midpoint:])
        self.assertTrue(top_active)
        self.assertTrue(bottom_active)


if __name__ == "__main__":
    unittest.main()
