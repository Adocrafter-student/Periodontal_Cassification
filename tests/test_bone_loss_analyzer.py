import unittest

import numpy as np

from utils.bone_loss_analyzer import analyze_tooth_bone_loss


def make_synthetic_tooth(jaw="mandibular", vertical=False):
    crop = np.zeros((120, 80), dtype=np.uint8)
    mask = np.zeros((120, 80), dtype=np.uint8)

    if jaw == "mandibular":
        mask[10:31, 25:55] = 255
        mask[31:101, 34:46] = 255
        cej = 31
        direction_rows = range(cej, 101)
    else:
        mask[89:111, 25:55] = 255
        mask[20:89, 34:46] = 255
        cej = 88
        direction_rows = range(cej, 19, -1)

    crop[:] = 35
    crop[mask > 0] = 190

    left_crest_offset = 6
    right_crest_offset = 32 if vertical else 8

    for offset, row in enumerate(direction_rows):
        if offset >= left_crest_offset:
            crop[row, 5:34] = 170
        if offset >= right_crest_offset:
            crop[row, 46:75] = 170

    return crop, mask


class BoneLossAnalyzerTest(unittest.TestCase):
    def test_invalid_mask_is_not_measurable(self):
        crop = np.zeros((50, 50), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)

        result = analyze_tooth_bone_loss(crop, mask, {"tooth_number": "31"})

        self.assertFalse(result["measurable"])
        self.assertEqual(result["bone_crest_position"], "not_measurable")

    def test_mandibular_vertical_defect_is_detected(self):
        crop, mask = make_synthetic_tooth(jaw="mandibular", vertical=True)
        result = analyze_tooth_bone_loss(
            crop,
            mask,
            {"tooth_number": "31", "tooth_type": "central_incisor", "confidence": 0.9},
        )

        self.assertTrue(result["measurable"])
        self.assertEqual(result["jaw"], "mandibular")
        self.assertEqual(result["bone_loss_pattern"], "vertical")
        self.assertGreater(result["vertical_defect_mm"], 3.0)

    def test_maxillary_horizontal_defect_is_detected(self):
        crop, mask = make_synthetic_tooth(jaw="maxillary", vertical=False)
        result = analyze_tooth_bone_loss(
            crop,
            mask,
            {"tooth_number": "11", "tooth_type": "central_incisor", "confidence": 0.9},
        )

        self.assertTrue(result["measurable"])
        self.assertEqual(result["jaw"], "maxillary")
        self.assertEqual(result["bone_loss_pattern"], "horizontal")
        self.assertGreater(result["root_length_px"], 30)


if __name__ == "__main__":
    unittest.main()
