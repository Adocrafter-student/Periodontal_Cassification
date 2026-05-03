import unittest

from utils.fdi_numbering import assign_fdi_numbers


class FdiNumberingTest(unittest.TestCase):
    def test_uses_tooth_row_gap_instead_of_image_midpoint(self):
        xs_upper_left_image = [1350, 1250, 1150, 1050, 950, 850, 750, 650]
        xs_upper_right_image = [1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150]
        xs_lower_left_image = [1390, 1300, 1200, 1100, 1000, 900, 800]
        xs_lower_right_image = [1460, 1560, 1660, 1760, 1860, 1960, 2060, 2160]

        detections = []
        for x in xs_upper_left_image + xs_upper_right_image:
            detections.append({"bbox": [x - 20, 290, x + 20, 330], "tooth_type": "molar"})
        for x in xs_lower_left_image + xs_lower_right_image:
            detections.append({"bbox": [x - 20, 590, x + 20, 630], "tooth_type": "molar"})

        numbered = assign_fdi_numbers(detections, image_width=2800, image_height=1500)
        numbers = {det["tooth_number"] for det in numbered}

        self.assertIn("11", numbers)
        self.assertIn("21", numbers)
        self.assertIn("31", numbers)
        self.assertIn("41", numbers)
        self.assertNotIn("19", numbers)
        self.assertNotIn("29", numbers)
        self.assertNotIn("30", numbers)
        self.assertEqual(len(numbers), len(numbered))


if __name__ == "__main__":
    unittest.main()
