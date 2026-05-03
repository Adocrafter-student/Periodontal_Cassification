"""
Spatial heuristic to assign FDI tooth numbers from YOLOv8-seg detections.

FDI numbering:
  Quadrant 1 (upper-right from patient POV = left side of image): 18..11
  Quadrant 2 (upper-left  from patient POV = right side of image): 21..28
  Quadrant 3 (lower-left  from patient POV = right side of image): 31..38  (but reversed spatially)
  Quadrant 4 (lower-right from patient POV = left side of image):  48..41

Within each quadrant the expected tooth-type sequence (from midline outward):
  central_incisor, lateral_incisor, canine, premolar, premolar, molar, molar, molar
"""

from typing import Dict, List, Tuple

TOOTH_TYPE_ORDER = [
    "central_incisor",
    "lateral_incisor",
    "canine",
    "premolar",
    "premolar",
    "molar",
    "molar",
    "molar",
]

QUADRANT_FDI_BASE = {
    "upper_right": 10,  # 11-18
    "upper_left": 20,   # 21-28
    "lower_left": 30,   # 31-38
    "lower_right": 40,  # 41-48
}


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _estimate_jaw_split_y(detections: List[Dict], image_height: int) -> float:
    """Estimate the divider between maxillary and mandibular tooth rows.

    OPG teeth are often concentrated above the image midpoint. Splitting on
    ``image_height / 2`` can therefore put both jaws into the upper row. The
    largest gap between detection y-centers is a better divider for this output.
    """
    centers_y = sorted(_bbox_center(det["bbox"])[1] for det in detections)
    if len(centers_y) < 4:
        return image_height / 2.0

    gaps = [
        (centers_y[idx + 1] - centers_y[idx], idx)
        for idx in range(len(centers_y) - 1)
    ]
    largest_gap, split_idx = max(gaps, key=lambda item: item[0])
    if largest_gap < image_height * 0.05:
        return image_height / 2.0

    return (centers_y[split_idx] + centers_y[split_idx + 1]) / 2.0


def assign_fdi_numbers(
    detections: List[Dict],
    image_width: int,
    image_height: int,
) -> List[Dict]:
    """Assign FDI tooth numbers to a list of detections.

    Each detection dict must have at least:
        - "bbox": [x1, y1, x2, y2] in pixel coords
        - "tooth_type": one of the 5 class names

    Returns the same list with an added "tooth_number" key (str, e.g. "11").
    """
    if not detections:
        return detections

    mid_x = image_width / 2.0
    jaw_split_y = _estimate_jaw_split_y(detections, image_height)

    quadrants: Dict[str, List[Dict]] = {
        "upper_right": [],
        "upper_left": [],
        "lower_left": [],
        "lower_right": [],
    }

    for det in detections:
        cx, cy = _bbox_center(det["bbox"])
        if cy < jaw_split_y:
            quadrant = "upper_right" if cx < mid_x else "upper_left"
        else:
            quadrant = "lower_right" if cx < mid_x else "lower_left"
        det["_cx"] = cx
        det["_cy"] = cy
        det["_quadrant"] = quadrant
        quadrants[quadrant].append(det)

    # Sort each quadrant: teeth closest to the midline first (smallest
    # distance from mid_x), which corresponds to incisors → molars.
    for qname, teeth in quadrants.items():
        teeth.sort(key=lambda d: abs(d["_cx"] - mid_x))
        base = QUADRANT_FDI_BASE[qname]
        for idx, det in enumerate(teeth):
            if idx < 8:
                det["tooth_number"] = str(base + idx + 1)
            else:
                det["tooth_number"] = f"extra_{qname}_{idx + 1}"
            det["fdi_quadrant"] = qname

    for det in detections:
        det.pop("_cx", None)
        det.pop("_cy", None)
        det.pop("_quadrant", None)

    return detections
