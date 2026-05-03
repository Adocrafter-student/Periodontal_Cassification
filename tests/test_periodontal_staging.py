import unittest

from utils.periodontal_staging import severity_stage, stage_periodontitis


class PeriodontalStagingRulesTest(unittest.TestCase):
    def test_severity_thresholds(self):
        self.assertIsNone(severity_stage(0.0, 0.0))
        self.assertEqual(severity_stage(10.0, 1.3), 1)
        self.assertEqual(severity_stage(20.0, 3.0), 2)
        self.assertEqual(severity_stage(34.0, 4.4), 3)
        self.assertEqual(severity_stage(25.0, 5.0), 3)

    def test_vertical_complexity_upgrades_to_stage_three(self):
        teeth = [{"tooth_number": "11", "confidence": 0.9}]
        analysis = [{
            "tooth_number": "11",
            "tooth_type": "central_incisor",
            "measurable": True,
            "rbl_percent": 12.0,
            "cal_mm": 1.6,
            "vertical_defect_mm": 3.2,
            "bone_loss_pattern": "vertical",
        }]

        result = stage_periodontitis(teeth, analysis)

        self.assertEqual(result["staging"]["severity_stage"], 1)
        self.assertEqual(result["staging"]["complexity_stage"], 3)
        self.assertEqual(result["staging"]["stage"], 3)

    def test_missing_teeth_do_not_upgrade_stage(self):
        teeth = [{"tooth_number": "11", "confidence": 0.9}]
        analysis = [{
            "tooth_number": "11",
            "tooth_type": "central_incisor",
            "measurable": True,
            "rbl_percent": 10.0,
            "cal_mm": 1.3,
            "vertical_defect_mm": 0.0,
            "bone_loss_pattern": "horizontal",
        }]

        result = stage_periodontitis(teeth, analysis)

        self.assertGreater(result["staging"]["radiographically_missing_teeth_count"], 0)
        self.assertFalse(result["staging"]["tooth_loss_upgrade_applied"])
        self.assertEqual(result["staging"]["stage"], 1)

    def test_explicit_periodontal_tooth_loss_can_upgrade(self):
        teeth = [{"tooth_number": "11", "confidence": 0.9}]
        analysis = [{
            "tooth_number": "11",
            "tooth_type": "central_incisor",
            "measurable": True,
            "rbl_percent": 10.0,
            "cal_mm": 1.3,
            "vertical_defect_mm": 0.0,
            "bone_loss_pattern": "horizontal",
        }]

        result = stage_periodontitis(
            teeth,
            analysis,
            case_metadata={"periodontal_tooth_loss_count": 5},
        )

        self.assertTrue(result["staging"]["tooth_loss_upgrade_applied"])
        self.assertEqual(result["staging"]["stage"], 4)

    def test_extent_thresholds(self):
        teeth = [{"tooth_number": f"1{i}", "confidence": 0.9} for i in range(1, 8)]
        localized = [{
            "tooth_number": "11",
            "tooth_type": "central_incisor",
            "measurable": True,
            "rbl_percent": 5.0,
            "cal_mm": 0.7,
            "vertical_defect_mm": 0.0,
            "bone_loss_pattern": "horizontal",
        }]
        generalized = []
        for number in ["11", "12", "13", "14", "15", "16", "17", "21", "22"]:
            generalized.append({
                "tooth_number": number,
                "tooth_type": "premolar",
                "measurable": True,
                "rbl_percent": 5.0,
                "cal_mm": 0.7,
                "vertical_defect_mm": 0.0,
                "bone_loss_pattern": "horizontal",
            })

        self.assertEqual(stage_periodontitis(teeth, localized)["staging"]["extent"], "localized")
        self.assertEqual(stage_periodontitis(teeth, generalized)["staging"]["extent"], "generalized")

    def test_extent_counts_unique_valid_staging_teeth_only(self):
        teeth = [{"tooth_number": str(number), "confidence": 0.9} for number in range(11, 48)]
        analysis = []
        for idx in range(30):
            number = str(11 + idx)
            analysis.append({
                "tooth_number": number,
                "tooth_type": "premolar",
                "confidence": 0.9,
                "measurement_confidence": 0.9,
                "measurable": True,
                "rbl_percent": 5.0,
                "cal_mm": 0.7,
                "vertical_defect_mm": 0.0,
                "bone_loss_pattern": "horizontal",
            })

        analysis.append(dict(analysis[0]))

        result = stage_periodontitis(teeth, analysis)

        self.assertLessEqual(result["staging"]["percent_teeth_affected"], 100.0)
        self.assertLessEqual(result["staging"]["affected_teeth_count"], 28)


if __name__ == "__main__":
    unittest.main()
