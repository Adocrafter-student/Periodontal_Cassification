"""2017 Tonetti periodontal classification -- rule-based staging.

Implements the three-criterion framework:
  1. Severity  (CAL + RBL%)   --> base stage I-IV
  2. Tooth loss               --> can upgrade to III or IV
  3. Complexity               --> can upgrade to III or IV

Final stage = max(severity, tooth_loss, complexity).
Extent = localized / generalized / molar-incisor pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from utils.bone_loss_analyzer import ToothAnalysis

EXPECTED_FDI = set()
for q in [10, 20, 30, 40]:
    for t in range(1, 8):  # 1-7, excluding 3rd molars (8)
        EXPECTED_FDI.add(str(q + t))

MOLAR_FDI = {"16", "17", "26", "27", "36", "37", "46", "47"}
INCISOR_FDI = {"11", "12", "21", "22", "31", "32", "41", "42"}


@dataclass
class StagingResult:
    stage: int = 0                          # final stage 0-4 (0 = healthy)
    severity_stage: int = 0
    tooth_loss_stage: int = 0
    complexity_stage: int = 0

    max_cal_mm: float = 0.0
    max_rbl_percent: float = 0.0
    rbl_location: str = "coronal_third"
    tooth_loss_count: int = 0

    extent: str = "localized"
    percent_teeth_affected: float = 0.0
    bone_loss_pattern: str = "horizontal"

    teeth_analyzed: int = 0
    teeth_with_bone_loss: int = 0
    missing_teeth: list[str] = field(default_factory=list)
    analysis_notes: list[str] = field(default_factory=list)


def _severity_stage(max_cal: float, max_rbl: float, rbl_loc: str) -> int:
    """Determine stage from severity criteria alone."""
    if max_cal < 1.0 and max_rbl < 5.0:
        return 0
    if max_cal <= 2.0 and max_rbl < 15.0:
        return 1
    if max_cal <= 4.0 and max_rbl <= 33.0:
        return 2
    return 3


def _tooth_loss_stage(missing_count: int) -> int:
    if missing_count == 0:
        return 0
    if missing_count <= 4:
        return 3
    return 4


def _complexity_stage(
    analyses: List[ToothAnalysis],
    missing_count: int,
) -> int:
    """Determine complexity stage from vertical defects and functional needs."""
    max_vertical_loss_mm = 0.0
    has_vertical = False

    for a in analyses:
        if a.bone_loss_pattern == "vertical":
            has_vertical = True
            max_vertical_loss_mm = max(max_vertical_loss_mm, a.cal_mm)

    stage = 0

    if has_vertical and max_vertical_loss_mm >= 3.0:
        stage = max(stage, 3)

    if missing_count >= 5:
        stage = max(stage, 4)

    max_cal = max((a.cal_mm for a in analyses), default=0.0)
    if max_cal <= 4.0:
        stage = max(stage, min(stage, 1))
    elif max_cal <= 5.0:
        stage = max(stage, 2)
    elif max_cal > 5.0 and has_vertical:
        stage = max(stage, 3)

    return stage


def _determine_extent(
    analyses: List[ToothAnalysis],
    detected_fdi: set[str],
) -> tuple[str, float]:
    """Determine localized / generalized / molar-incisor pattern."""
    if not analyses:
        return "localized", 0.0

    bone_loss_threshold_rbl = 10.0
    affected_teeth = {a.tooth_number for a in analyses
                      if a.rbl_percent >= bone_loss_threshold_rbl}

    total_teeth = len(detected_fdi)
    if total_teeth == 0:
        return "localized", 0.0

    affected_teeth = affected_teeth & detected_fdi
    percent = min(100.0, (len(affected_teeth) / total_teeth) * 100.0)

    affected_molars = affected_teeth & MOLAR_FDI
    affected_incisors = affected_teeth & INCISOR_FDI
    affected_other = affected_teeth - MOLAR_FDI - INCISOR_FDI

    molar_incisor_count = len(affected_molars) + len(affected_incisors)
    is_molar_incisor = (
        molar_incisor_count >= 3
        and len(affected_other) <= 2
        and len(affected_teeth) >= 3
    )

    if is_molar_incisor:
        extent = "molar_incisor_pattern"
    elif percent >= 30.0:
        extent = "generalized"
    else:
        extent = "localized"

    return extent, round(percent, 1)


def _overall_pattern(analyses: List[ToothAnalysis]) -> str:
    patterns = [a.bone_loss_pattern for a in analyses if a.analysis_ok]
    if not patterns:
        return "horizontal"

    vert_count = sum(1 for p in patterns if p == "vertical")
    horiz_count = sum(1 for p in patterns if p == "horizontal")

    if vert_count == 0:
        return "horizontal"
    if horiz_count == 0:
        return "vertical"
    return "mixed"


def stage_patient(
    analyses: List[ToothAnalysis],
    detected_fdi_numbers: set[str],
) -> StagingResult:
    """Apply the 2017 Tonetti staging framework to a set of per-tooth analyses.

    Args:
        analyses: list of ToothAnalysis from bone_loss_analyzer
        detected_fdi_numbers: set of FDI numbers detected in the image
    """
    result = StagingResult()

    valid = [a for a in analyses if a.analysis_ok]
    result.teeth_analyzed = len(valid)

    if not valid:
        result.analysis_notes.append("No valid tooth analyses available")
        return result

    third_molars = {"18", "28", "38", "48"}
    detected_non_3m = detected_fdi_numbers - third_molars
    expected_non_3m = EXPECTED_FDI

    missing = expected_non_3m - detected_non_3m
    result.missing_teeth = sorted(missing)
    result.tooth_loss_count = len(missing)
    result.analysis_notes.append("3rd molars excluded from staging")

    max_cal = max(a.cal_mm for a in valid)
    max_rbl = max(a.rbl_percent for a in valid)
    result.max_cal_mm = round(max_cal, 1)
    result.max_rbl_percent = round(max_rbl, 1)

    worst_loc = "coronal_third"
    for a in valid:
        if a.bone_crest_position == "apical_third":
            worst_loc = "apical_third"
            break
        elif a.bone_crest_position == "middle_third":
            worst_loc = "middle_third"
    result.rbl_location = worst_loc

    result.severity_stage = _severity_stage(max_cal, max_rbl, worst_loc)
    result.tooth_loss_stage = _tooth_loss_stage(result.tooth_loss_count)
    result.complexity_stage = _complexity_stage(valid, result.tooth_loss_count)

    result.stage = max(result.severity_stage, result.tooth_loss_stage, result.complexity_stage)

    bone_loss_threshold = 10.0
    result.teeth_with_bone_loss = sum(1 for a in valid if a.rbl_percent >= bone_loss_threshold)

    result.extent, result.percent_teeth_affected = _determine_extent(
        valid, detected_non_3m,
    )

    result.bone_loss_pattern = _overall_pattern(valid)

    return result
