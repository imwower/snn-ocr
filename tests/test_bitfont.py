"""Unit tests for snn_ocr.bitfont utilities."""
from __future__ import annotations

import unittest

from snn_ocr import bitfont


class BitfontTest(unittest.TestCase):
    """Validate bitmap glyphs and transforms."""

    def test_glyph_dimensions(self) -> None:
        glyph_5x7 = bitfont.get_bitmap("A", "5x7")
        glyph_7x9 = bitfont.get_bitmap("A", "7x9")
        self.assertEqual(len(glyph_5x7), 7)
        self.assertEqual(len(glyph_5x7[0]), 5)
        self.assertEqual(len(glyph_7x9), 9)
        self.assertEqual(len(glyph_7x9[0]), 7)

    def test_render_spacing(self) -> None:
        grid = bitfont.render_text_row("Ab", size="5x7", inner=2, outer=1)
        self.assertEqual(len(grid), 7)
        self.assertTrue(all(len(row) == len(grid[0]) for row in grid))
        left_border = [row[0] for row in grid]
        right_border = [row[-1] for row in grid]
        self.assertTrue(all(pixel == 0 for pixel in left_border))
        self.assertTrue(all(pixel == 0 for pixel in right_border))

    def test_scale_nn(self) -> None:
        glyph = bitfont.get_bitmap("3", "5x7")
        scaled = bitfont.scale_nn(glyph, 3, 2)
        self.assertEqual(len(scaled), 14)
        self.assertEqual(len(scaled[0]), 15)

    def test_dilate_and_shear(self) -> None:
        glyph = bitfont.get_bitmap("W", "5x7")
        dilated = bitfont.dilate3x3(glyph, n=1)
        self.assertEqual(len(dilated), len(glyph))
        self.assertEqual(len(dilated[0]), len(glyph[0]))
        sheared = bitfont.shear_x(glyph, 2)
        self.assertEqual(len(sheared), len(glyph))
        self.assertGreaterEqual(len(sheared[0]), len(glyph[0]))

    def test_to_gray(self) -> None:
        glyph = bitfont.get_bitmap("!", "5x7")
        grayscale = bitfont.to_gray(glyph, fg=200, bg=10)
        self.assertEqual(len(grayscale), len(glyph))
        self.assertEqual(len(grayscale[0]), len(glyph[0]))
        unique_values = {pixel for row in grayscale for pixel in row}
        self.assertSetEqual(unique_values, {10, 200})

    def test_validate_charset(self) -> None:
        self.assertTrue(bitfont.validate_charset("5x7"))
        with self.assertRaises(KeyError):
            bitfont.validate_charset("5x7", required=("~",))


if __name__ == "__main__":
    unittest.main()
