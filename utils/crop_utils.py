"""Bounding-box expansion and cropping utilities for tooth + bone crops."""

from typing import List, Tuple

import cv2
import numpy as np


def expand_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int,
    expand_ratio: float = 0.40,
) -> List[int]:
    """Expand a bounding box by *expand_ratio* on every side, clamped to image bounds.

    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates.
        image_width: width of the source image.
        image_height: height of the source image.
        expand_ratio: fraction of the box dimension to add on each side.

    Returns:
        [ex1, ey1, ex2, ey2] as ints, clamped to [0, image_width) x [0, image_height).
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    dx = w * expand_ratio
    dy = h * expand_ratio

    ex1 = max(0, int(x1 - dx))
    ey1 = max(0, int(y1 - dy))
    ex2 = min(image_width, int(x2 + dx))
    ey2 = min(image_height, int(y2 + dy))

    return [ex1, ey1, ex2, ey2]


def crop_tooth_region(
    image: np.ndarray,
    bbox: List[float],
    expand_ratio: float = 0.40,
) -> Tuple[np.ndarray, List[int]]:
    """Crop the tooth + surrounding bone area from *image*.

    Returns:
        (cropped_image, expanded_bbox)
    """
    h, w = image.shape[:2]
    expanded = expand_bbox(bbox, w, h, expand_ratio)
    ex1, ey1, ex2, ey2 = expanded
    crop = image[ey1:ey2, ex1:ex2].copy()
    return crop, expanded


def save_mask_crop(
    mask: np.ndarray,
    bbox: List[float],
    image_width: int,
    image_height: int,
    expand_ratio: float = 0.40,
) -> Tuple[np.ndarray, List[int]]:
    """Crop the segmentation mask to the same expanded region as the tooth crop."""
    expanded = expand_bbox(bbox, image_width, image_height, expand_ratio)
    ex1, ey1, ex2, ey2 = expanded
    mask_crop = mask[ey1:ey2, ex1:ex2].copy()
    return mask_crop, expanded
