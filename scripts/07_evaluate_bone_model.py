"""Evaluate the bone landmark model with visual overlays.

Loads the trained U-Net, runs inference on the test (or val) split,
and produces overlay images showing predicted vs ground-truth landmarks.

Usage:
    python scripts/07_evaluate_bone_model.py
    python scripts/07_evaluate_bone_model.py --split test --max-images 20
    python scripts/07_evaluate_bone_model.py --weights runs/bone_landmark/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bone_model.dataset import BoneLandmarkDataset
from bone_model.unet import build_model


CHANNEL_COLORS = {
    0: (0, 0, 255),    # CEJ: red
    1: (0, 255, 255),  # Apex: yellow
    2: (0, 255, 0),    # Bone crest: green
    3: (255, 128, 0),  # Tooth mask: blue-ish
}
CHANNEL_NAMES = ["CEJ", "Apex", "Bone crest", "Tooth"]


def overlay_predictions(
    image: np.ndarray,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Create a side-by-side or single overlay visualization."""
    if len(image.shape) == 2:
        vis_pred = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_pred = image.copy()

    # Overlay predictions
    for c in range(min(pred_masks.shape[0], 4)):
        mask = (pred_masks[c] > threshold).astype(np.uint8) * 255
        color = CHANNEL_COLORS[c]
        if c == 3:
            # Tooth mask: semi-transparent
            colored = np.zeros_like(vis_pred)
            colored[:] = color
            alpha_mask = (mask > 127).astype(np.float32) * 0.2
            for ch in range(3):
                vis_pred[:, :, ch] = (
                    vis_pred[:, :, ch] * (1 - alpha_mask) +
                    colored[:, :, ch] * alpha_mask
                ).astype(np.uint8)
        else:
            vis_pred[mask > 127] = color

    if gt_masks is None:
        return vis_pred

    # Ground truth overlay
    if len(image.shape) == 2:
        vis_gt = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_gt = image.copy()

    for c in range(min(gt_masks.shape[0], 4)):
        mask = (gt_masks[c] > 0.5).astype(np.uint8) * 255
        color = CHANNEL_COLORS[c]
        if c == 3:
            colored = np.zeros_like(vis_gt)
            colored[:] = color
            alpha_mask = (mask > 127).astype(np.float32) * 0.2
            for ch in range(3):
                vis_gt[:, :, ch] = (
                    vis_gt[:, :, ch] * (1 - alpha_mask) +
                    colored[:, :, ch] * alpha_mask
                ).astype(np.uint8)
        else:
            vis_gt[mask > 127] = color

    # Side-by-side: GT left, Pred right
    label_h = 30
    h, w = vis_gt.shape[:2]

    canvas = np.zeros((h + label_h, w * 2 + 10, 3), dtype=np.uint8)
    canvas[:label_h, :w] = (40, 40, 40)
    canvas[:label_h, w + 10:] = (40, 40, 40)
    cv2.putText(canvas, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, "Prediction", (w + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    canvas[label_h:label_h + h, :w] = vis_gt
    canvas[label_h:label_h + h, w + 10:w + 10 + w] = vis_pred

    # Legend
    y_legend = h + label_h - 5
    for c, name in enumerate(CHANNEL_NAMES):
        x = 10 + c * 150
        color = CHANNEL_COLORS[c]
        cv2.rectangle(canvas, (x, y_legend - 12), (x + 12, y_legend), color, -1)
        cv2.putText(canvas, name, (x + 16, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return canvas


def compute_channel_dice(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)
    results = {}
    for c, name in enumerate(CHANNEL_NAMES):
        p = pred_bin[c].flatten()
        t = target_bin[c].flatten()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        results[name] = float(dice)
    results["mean"] = sum(results.values()) / len(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate bone landmark model")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    bone_cfg = cfg.get("bone_model", {})

    weights_path = args.weights or str(ROOT / bone_cfg.get("project", "runs/bone_landmark") / "best.pt")
    image_size = bone_cfg.get("image_size", 512)
    device_str = args.device or bone_cfg.get("device", "0")

    if device_str in ("cpu", "mps"):
        device = torch.device(device_str)
    else:
        device = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")

    data_root = ROOT / "data" / "bone_landmark"
    output_dir = ROOT / "data" / "bone_eval" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {}).get("architecture", bone_cfg.get("architecture", {}))
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}, val dice: {ckpt.get('best_val_dice', '?')}")

    # Dataset
    dataset = BoneLandmarkDataset(
        images_dir=data_root / "images" / args.split,
        targets_dir=data_root / "targets" / args.split,
        image_size=image_size,
        augment=False,
    )
    print(f"Evaluating on {args.split} split: {len(dataset)} images")

    n_eval = min(args.max_images, len(dataset))
    all_dice: list[dict[str, float]] = []

    for idx in tqdm(range(n_eval), desc="Evaluating"):
        image_t, target_t = dataset[idx]
        stem = dataset.samples[idx]

        with torch.no_grad():
            pred_t = model(image_t.unsqueeze(0).to(device))
            pred_np = torch.sigmoid(pred_t[0]).cpu().numpy()

        target_np = target_t.numpy()
        image_np = (image_t[0].numpy() * 255).astype(np.uint8)

        # Dice scores
        dice = compute_channel_dice(pred_np, target_np, args.threshold)
        all_dice.append(dice)

        # Save overlay
        overlay = overlay_predictions(image_np, pred_np, target_np, args.threshold)
        cv2.imwrite(str(output_dir / f"{stem}_eval.png"), overlay)

    # Print summary
    print(f"\n{'Channel':<15} {'Dice (mean ± std)':>20}")
    print("-" * 37)
    for name in CHANNEL_NAMES + ["mean"]:
        scores = [d[name] for d in all_dice]
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"{name:<15} {mean:.4f} ± {std:.4f}")

    print(f"\nOverlays saved to: {output_dir}")


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
