"""Train the U-Net bone landmark segmentation model.

Reads converted data from data/bone_landmark/ (produced by 05_convert_denpar.py)
and trains a multi-channel U-Net to predict CEJ, apex, bone-crest, and tooth masks.

Usage:
    python scripts/06_train_bone_model.py
    python scripts/06_train_bone_model.py --epochs 100 --batch 8
    python scripts/06_train_bone_model.py --resume runs/bone_landmark/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bone_model.dataset import BoneLandmarkDataset
from bone_model.losses import CombinedLoss
from bone_model.unet import build_model


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Per-channel Dice score and pixel accuracy."""
    with torch.no_grad():
        pred_bin = (torch.sigmoid(pred) > threshold).float()
        metrics = {}
        channel_names = ["cej", "apex", "bone_line", "tooth"]
        for c, name in enumerate(channel_names):
            p = pred_bin[:, c]
            t = target[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            metrics[f"dice_{name}"] = dice.item()
        metrics["dice_mean"] = sum(metrics.values()) / len(metrics)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_metrics: dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        batch_metrics = compute_metrics(preds, targets)
        for k, v in batch_metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_metrics['dice_mean']:.3f}")

    avg = {"loss": running_loss / n_batches}
    for k, v in running_metrics.items():
        avg[k] = v / n_batches
    return avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_metrics: dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [val]  ", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = criterion(preds, targets)

        running_loss += loss.item()
        n_batches += 1

        batch_metrics = compute_metrics(preds, targets)
        for k, v in batch_metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v

    avg = {"loss": running_loss / max(n_batches, 1)}
    for k, v in running_metrics.items():
        avg[k] = v / max(n_batches, 1)
    return avg


def main():
    parser = argparse.ArgumentParser(description="Train bone landmark U-Net")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    bone_cfg = cfg.get("bone_model", {})

    epochs = args.epochs or bone_cfg.get("epochs", 80)
    batch_size = args.batch or bone_cfg.get("batch", 4)
    lr = args.lr or bone_cfg.get("lr", 1e-3)
    image_size = bone_cfg.get("image_size", 512)
    patience = bone_cfg.get("patience", 20)
    device_str = args.device or bone_cfg.get("device", "0")

    data_root = ROOT / "data" / "bone_landmark"
    output_dir = ROOT / bone_cfg.get("project", "runs/bone_landmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if device_str in ("cpu", "mps"):
        device = torch.device(device_str)
    else:
        device = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    train_ds = BoneLandmarkDataset(
        images_dir=data_root / "images" / "train",
        targets_dir=data_root / "targets" / "train",
        image_size=image_size,
        augment=True,
    )
    val_ds = BoneLandmarkDataset(
        images_dir=data_root / "images" / "val",
        targets_dir=data_root / "targets" / "val",
        image_size=image_size,
        augment=False,
    )

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training samples found. Run 05_convert_denpar.py first.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Model
    model_cfg = bone_cfg.get("architecture", {})
    model = build_model(model_cfg).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Loss
    channel_weights = bone_cfg.get("channel_weights", [1.0, 1.0, 2.0, 0.5])
    criterion = CombinedLoss(
        focal_weight=1.0,
        dice_weight=1.0,
        channel_weights=channel_weights,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Resume
    start_epoch = 0
    best_val_dice = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val dice: {best_val_dice:.4f}")

    # Training loop
    epochs_no_improve = 0
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Image size: {image_size}")
    print(f"  Patience: {patience}")
    print(f"  Channel weights (CEJ, apex, bone, tooth): {channel_weights}")
    print()

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        val_dice = val_metrics.get("dice_mean", 0.0)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Per-channel dice
        for name in ["cej", "apex", "bone_line", "tooth"]:
            key = f"dice_{name}"
            t_val = train_metrics.get(key, 0.0)
            v_val = val_metrics.get(key, 0.0)
            print(f"  {name:10s}  train={t_val:.3f}  val={v_val:.3f}")

        # Save best
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            epochs_no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "config": bone_cfg,
            }
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  ** New best: val_dice={best_val_dice:.4f} → saved best.pt")
        else:
            epochs_no_improve += 1

        # Save latest every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "config": bone_cfg,
            }
            torch.save(ckpt, output_dir / "last.pt")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")
    print(f"Weights saved to: {output_dir}")


if __name__ == "__main__":
    main()
