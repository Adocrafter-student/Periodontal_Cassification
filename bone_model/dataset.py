"""PyTorch Dataset for bone landmark heatmap training.

Expects the directory layout produced by 05_convert_denpar.py:

    data/bone_landmark/
        images/{split}/{id}.jpg
        targets/{split}/{id}_cej.png
        targets/{split}/{id}_apex.png
        targets/{split}/{id}_bone_line.png
        targets/{split}/{id}_tooth.png

The dataset returns:
    image   – (1, H, W) float32 tensor, normalized to [0, 1]
    target  – (C, H, W) float32 tensor, each channel binary {0, 1}
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CHANNEL_SUFFIXES = ["_cej.png", "_apex.png", "_bone_line.png", "_tooth.png"]


class BoneLandmarkDataset(Dataset):
    """Dataset for bone landmark segmentation training."""

    def __init__(
        self,
        images_dir: str | Path,
        targets_dir: str | Path,
        image_size: int = 512,
        augment: bool = False,
        transform: Callable | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.targets_dir = Path(targets_dir)
        self.image_size = image_size
        self.augment = augment
        self.transform = transform

        self.samples: list[str] = []
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            stem = img_path.stem
            cej_path = self.targets_dir / f"{stem}_cej.png"
            if cej_path.exists():
                self.samples.append(stem)

        if not self.samples:
            jpg_paths = sorted(self.images_dir.glob("*.png"))
            for img_path in jpg_paths:
                stem = img_path.stem
                cej_path = self.targets_dir / f"{stem}_cej.png"
                if cej_path.exists():
                    self.samples.append(stem)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.samples[idx]

        # Load image
        img_path = self.images_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = self.images_dir / f"{stem}.png"
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Load target channels
        channels = []
        for suffix in CHANNEL_SUFFIXES:
            mask_path = self.targets_dir / f"{stem}{suffix}"
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    m = np.zeros_like(image)
            else:
                m = np.zeros_like(image)
            channels.append(m)

        target = np.stack(channels, axis=0)  # (C, H, W)

        # Resize
        h, w = image.shape[:2]
        sz = self.image_size
        if h != sz or w != sz:
            image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
            resized_channels = []
            for c in range(target.shape[0]):
                resized_channels.append(
                    cv2.resize(target[c], (sz, sz), interpolation=cv2.INTER_NEAREST)
                )
            target = np.stack(resized_channels, axis=0)

        # Augmentation (simple, compatible with mask targets)
        if self.augment:
            image, target = self._augment(image, target)

        # Normalize
        image = image.astype(np.float32) / 255.0
        target = (target > 127).astype(np.float32)

        image_t = torch.from_numpy(image).unsqueeze(0)   # (1, H, W)
        target_t = torch.from_numpy(target)                # (C, H, W)

        if self.transform is not None:
            image_t, target_t = self.transform(image_t, target_t)

        return image_t, target_t

    @staticmethod
    def _augment(
        image: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple geometric + intensity augmentations."""
        # Horizontal flip (50% chance)
        if np.random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            target = np.flip(target, axis=2).copy()

        # Random brightness/contrast shift
        if np.random.random() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-20, 20)     # brightness
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Random rotation (small angle)
        if np.random.random() < 0.3:
            angle = np.random.uniform(-10, 10)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            for c in range(target.shape[0]):
                target[c] = cv2.warpAffine(
                    target[c], M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )

        return image, target
