"""Loss functions for bone landmark segmentation.

Combines per-channel binary cross-entropy with soft Dice loss for
robust training on sparse heatmap targets (CEJ/apex dots and bone lines
cover only a small fraction of pixels).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss, computed independently per channel then averaged."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        # Flatten spatial dimensions: (B, C, H*W)
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling extreme foreground/background imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """Weighted combination of Focal + Dice loss.

    Focal loss handles the extreme class imbalance (landmarks are tiny),
    Dice loss directly optimizes the overlap metric.

    Per-channel weights allow emphasizing certain landmarks (e.g., bone
    crest line is most important for downstream PBL calculation).
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        channel_weights: list[float] | None = None,
    ):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.channel_weights = channel_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.channel_weights is not None:
            w = torch.tensor(
                self.channel_weights, device=pred.device, dtype=pred.dtype,
            ).view(1, -1, 1, 1)
            pred_w = pred * w
            target_w = target * w
        else:
            pred_w = pred
            target_w = target

        focal = self.focal(pred_w, target_w)
        dice = self.dice(pred_w, target_w)
        return self.focal_weight * focal + self.dice_weight * dice
