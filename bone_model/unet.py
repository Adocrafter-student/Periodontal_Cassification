"""Lightweight U-Net for bone landmark segmentation.

Predicts 4 output channels from a single-channel (grayscale) X-ray:
  0: CEJ heatmap
  1: Apex heatmap
  2: Bone crest line mask
  3: Tooth mask

The architecture follows the classic encoder-decoder pattern with skip
connections.  The depth and base width are configurable for experimenting
on smaller GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            out = base_ch * (2 ** i)
            self.blocks.append(ConvBlock(ch, out))
            self.pools.append(nn.MaxPool2d(2))
            ch = out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, base_ch: int, depth: int):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = base_ch * (2 ** (i + 1)) if i < depth - 1 else base_ch * (2 ** i) * 2
            skip_ch = base_ch * (2 ** i)

            # For the bottleneck->first decoder step, in_ch is the bottleneck channels
            up_in = base_ch * (2 ** (i + 1)) if i < depth - 1 else base_ch * (2 ** depth)
            self.ups.append(nn.ConvTranspose2d(up_in, skip_ch, 2, stride=2))
            self.blocks.append(ConvBlock(skip_ch * 2, skip_ch))

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for up, block, skip in zip(self.ups, self.blocks, reversed(skips)):
            x = up(x)
            # Handle size mismatches from non-power-of-2 inputs
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class BoneLandmarkUNet(nn.Module):
    """U-Net for bone landmark heatmap prediction.

    Args:
        in_channels: number of input channels (1 for grayscale, 3 for RGB)
        out_channels: number of output channels (4: CEJ, apex, bone_line, tooth)
        base_channels: number of filters in the first encoder level
        depth: number of encoder/decoder levels
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, depth)

        bottleneck_ch = base_channels * (2 ** depth)
        encoder_last_ch = base_channels * (2 ** (depth - 1))
        self.bottleneck = ConvBlock(encoder_last_ch, bottleneck_ch)

        self.decoder = Decoder(base_channels, depth)
        self.head = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.head(x)


def build_model(cfg: dict | None = None) -> BoneLandmarkUNet:
    """Build model from config dict, falling back to sensible defaults."""
    if cfg is None:
        cfg = {}
    return BoneLandmarkUNet(
        in_channels=cfg.get("in_channels", 1),
        out_channels=cfg.get("out_channels", 4),
        base_channels=cfg.get("base_channels", 32),
        depth=cfg.get("depth", 4),
    )
