"""Per-tooth bone loss measurement from crop + segmentation mask.

Pipeline per tooth:
  1. Orient (maxillary vs mandibular) -- derived from FDI number, NOT image position
  2. Find CEJ from mask contour (crown-to-root narrowing)
  3. Measure root length (CEJ to apex)
  4. Detect alveolar bone crest from crop intensity outside the mask
  5. Calculate RBL% and estimate CAL (mm)
  6. Detect horizontal vs vertical bone loss pattern
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

ROOT_LENGTH_MM = {
    "central_incisor": 13.0,
    "lateral_incisor": 13.0,
    "canine": 16.0,
    "premolar": 14.0,
    "molar": 13.0,
}


@dataclass
class ToothAnalysis:
    tooth_number: str
    tooth_type: str
    is_maxillary: bool

    root_length_px: int = 0
    cej_row: int = 0
    apex_row: int = 0
    bone_crest_row_mesial: int = 0
    bone_crest_row_distal: int = 0
    bone_loss_px: int = 0

    rbl_percent: float = 0.0
    cal_mm: float = 0.0
    bone_crest_position: str = "coronal_third"
    bone_loss_pattern: str = "horizontal"
    analysis_ok: bool = True
    notes: list[str] = field(default_factory=list)


def _is_maxillary_from_fdi(tooth_number: str) -> bool:
    """FDI quadrants 1 and 2 (11-28) are maxillary, 3 and 4 (31-48) are mandibular."""
    try:
        num = int(tooth_number)
        return num < 30
    except ValueError:
        return True


def _mask_width_profile(mask_binary: np.ndarray) -> np.ndarray:
    return (mask_binary > 127).sum(axis=1).astype(np.float64)


def _find_cej_row(
    width_profile: np.ndarray,
    is_maxillary: bool,
) -> int:
    """Find the CEJ -- the steepest narrowing from crown to root.

    For maxillary teeth in an OPG crop: crown is at the BOTTOM, root at the TOP.
    For mandibular teeth: crown is at the TOP, root at the BOTTOM.
    We scan from crown toward root looking for the biggest width decrease.
    """
    h = len(width_profile)
    if h < 10:
        return h // 2

    kernel_size = max(3, h // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(width_profile, np.ones(kernel_size) / kernel_size, mode="same")

    if is_maxillary:
        # Crown at bottom, scan from bottom upward
        scan = smoothed[::-1].copy()
    else:
        # Crown at top, scan from top downward
        scan = smoothed.copy()

    derivative = np.gradient(scan)

    search_start = int(h * 0.15)
    search_end = int(h * 0.70)
    search_region = derivative[search_start:search_end]

    if len(search_region) == 0:
        return h // 2

    min_idx = int(np.argmin(search_region)) + search_start

    if is_maxillary:
        cej_row = (h - 1) - min_idx
    else:
        cej_row = min_idx

    return int(np.clip(cej_row, 0, h - 1))


def _find_apex_row(
    width_profile: np.ndarray,
    cej_row: int,
    is_maxillary: bool,
) -> int:
    """Find the root apex -- the last row with mask pixels on the root side of CEJ."""
    h = len(width_profile)
    if is_maxillary:
        # Root is above CEJ (toward row 0)
        root_region = width_profile[:cej_row]
        nonzero = np.nonzero(root_region > 0)[0]
        return int(nonzero[0]) if len(nonzero) > 0 else 0
    else:
        # Root is below CEJ (toward row h-1)
        root_region = width_profile[cej_row:]
        nonzero = np.nonzero(root_region > 0)[0]
        return int(cej_row + nonzero[-1]) if len(nonzero) > 0 else h - 1


def _detect_bone_crest(
    crop_gray: np.ndarray,
    mask_binary: np.ndarray,
    cej_row: int,
    apex_row: int,
    is_maxillary: bool,
) -> tuple[int, int]:
    """Detect alveolar bone crest on mesial and distal sides of the root.

    The bone crest is the most coronal (CEJ-side) point of alveolar bone
    alongside the root. On a healthy tooth it sits 1-2 mm below the CEJ.
    With periodontitis it recedes toward the apex.

    Approach: starting from the CROWN SIDE we scan toward the apex and look
    for the first region where the peri-root band shows bright bone.  The
    distance from CEJ to this first-bone row is the bone loss.

    Returns (bone_crest_row_mesial, bone_crest_row_distal).
    """
    h, w = crop_gray.shape[:2]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask_binary, kernel, iterations=5)
    peri_band = cv2.subtract(dilated, mask_binary)

    root_top = max(0, min(cej_row, apex_row))
    root_bottom = min(h - 1, max(cej_row, apex_row))
    root_len = root_bottom - root_top
    if root_len < 10:
        return cej_row, cej_row

    mask_bool = mask_binary > 127
    col_centers = []
    for row in range(root_top, root_bottom + 1):
        cols = np.nonzero(mask_bool[row])[0]
        if len(cols) > 0:
            col_centers.append((cols[0] + cols[-1]) / 2.0)
    midline = np.median(col_centers) if col_centers else w / 2.0

    col_grid = np.arange(w)[np.newaxis, :].repeat(h, axis=0)
    mesial_peri = peri_band.copy()
    distal_peri = peri_band.copy()
    mesial_peri[col_grid <= midline] = 0
    distal_peri[col_grid > midline] = 0

    def _find_crest_in_side(side_mask: np.ndarray) -> int:
        """Scan from CEJ toward apex. The bone crest is where bone first
        appears (first bright region in the peri-root band)."""
        num_slices = max(12, root_len // 5)
        slice_h = max(1, root_len // num_slices)

        slice_means = []
        for i in range(num_slices):
            r0 = root_top + i * slice_h
            r1 = min(r0 + slice_h, root_bottom)
            pixels = crop_gray[r0:r1][side_mask[r0:r1] > 127]
            if len(pixels) > 3:
                slice_means.append((i, float(np.mean(pixels))))
            else:
                slice_means.append((i, -1.0))

        valid = [(i, v) for i, v in slice_means if v >= 0]
        if not valid:
            return cej_row

        vals = np.array([v for _, v in valid])
        max_int = np.max(vals)
        if max_int < 20:
            return cej_row

        # Establish bone reference: median of the brightest 50% of slices
        sorted_vals = np.sort(vals)
        top_half = sorted_vals[len(sorted_vals) // 2:]
        bone_ref = np.median(top_half) if len(top_half) > 0 else max_int

        # Bone-present threshold: a slice has bone if its intensity is
        # at least 70% of the bone reference
        bone_threshold = bone_ref * 0.70

        # Order slices from CEJ toward apex
        if is_maxillary:
            ordered = list(reversed(valid))  # CEJ at bottom -> scan high-to-low index
        else:
            ordered = list(valid)  # CEJ at top -> scan low-to-high index

        # Find the first slice (from CEJ side) that has bone
        crest_slice_idx = None
        for orig_i, intensity in ordered:
            if intensity >= bone_threshold:
                crest_slice_idx = orig_i
                break

        if crest_slice_idx is None:
            # No bone found at all -> complete bone loss, crest at apex
            return apex_row

        crest_row = root_top + crest_slice_idx * slice_h
        return int(np.clip(crest_row, root_top, root_bottom))

    crest_mesial = _find_crest_in_side(mesial_peri)
    crest_distal = _find_crest_in_side(distal_peri)

    return crest_mesial, crest_distal


def _rbl_location(rbl_percent: float) -> str:
    if rbl_percent < 15:
        return "coronal_third"
    elif rbl_percent < 40:
        return "coronal_third"
    elif rbl_percent < 66:
        return "middle_third"
    else:
        return "apical_third"


def _detect_pattern(
    crest_mesial: int,
    crest_distal: int,
    cej_row: int,
    root_length_px: int,
) -> str:
    if root_length_px < 5:
        return "horizontal"

    mesial_loss = abs(crest_mesial - cej_row)
    distal_loss = abs(crest_distal - cej_row)

    diff_ratio = abs(mesial_loss - distal_loss) / root_length_px
    if diff_ratio < 0.12:
        return "horizontal"
    elif diff_ratio > 0.25:
        return "vertical"
    else:
        return "mixed"


def analyze_tooth(
    crop_path: str,
    mask_path: str,
    tooth_number: str,
    tooth_type: str,
    bbox_y_center: float,
    image_height: int,
) -> ToothAnalysis:
    """Full per-tooth bone loss analysis."""
    is_maxillary = _is_maxillary_from_fdi(tooth_number)

    result = ToothAnalysis(
        tooth_number=tooth_number,
        tooth_type=tooth_type,
        is_maxillary=is_maxillary,
    )

    crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if crop is None or mask is None:
        result.analysis_ok = False
        result.notes.append("Could not read crop or mask file")
        return result

    if crop.shape != mask.shape:
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    mask_pixels = np.count_nonzero(mask_binary)
    if mask_pixels < 50:
        result.analysis_ok = False
        result.notes.append("Mask too small or empty")
        return result

    width_profile = _mask_width_profile(mask_binary)
    cej_row = _find_cej_row(width_profile, is_maxillary)
    apex_row = _find_apex_row(width_profile, cej_row, is_maxillary)

    root_length_px = abs(cej_row - apex_row)
    if root_length_px < 10:
        result.analysis_ok = False
        result.notes.append(f"Root length too short ({root_length_px}px)")
        return result

    crest_mesial, crest_distal = _detect_bone_crest(
        crop, mask_binary, cej_row, apex_row, is_maxillary,
    )

    # Bone loss = distance from CEJ to the bone crest.
    # For maxillary: CEJ is at a higher row number (bottom), apex at lower row.
    #   Crest is between them. Loss = cej_row - crest_row.
    # For mandibular: CEJ is at a lower row number (top), apex at higher.
    #   Crest is between them. Loss = crest_row - cej_row.
    if is_maxillary:
        loss_mesial = max(0, cej_row - crest_mesial)
        loss_distal = max(0, cej_row - crest_distal)
    else:
        loss_mesial = max(0, crest_mesial - cej_row)
        loss_distal = max(0, crest_distal - cej_row)

    bone_loss_px = max(loss_mesial, loss_distal)

    # Sanity cap: bone loss cannot exceed root length
    bone_loss_px = min(bone_loss_px, root_length_px)

    rbl_percent = (bone_loss_px / root_length_px) * 100.0
    rbl_percent = min(100.0, max(0.0, rbl_percent))

    avg_root_mm = ROOT_LENGTH_MM.get(tooth_type, 13.0)
    cal_mm = round((rbl_percent / 100.0) * avg_root_mm, 1)

    pattern = _detect_pattern(crest_mesial, crest_distal, cej_row, root_length_px)

    result.root_length_px = root_length_px
    result.cej_row = cej_row
    result.apex_row = apex_row
    result.bone_crest_row_mesial = crest_mesial
    result.bone_crest_row_distal = crest_distal
    result.bone_loss_px = bone_loss_px
    result.rbl_percent = round(rbl_percent, 1)
    result.cal_mm = cal_mm
    result.bone_crest_position = _rbl_location(rbl_percent)
    result.bone_loss_pattern = pattern

    return result
