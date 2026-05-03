"""Convert DenPAR bone-dataset annotations into heatmap / mask targets.

For each image in the bone-dataset, this script reads:
  - Key Points Annotations/{id}.json  (bboxes, CEJ_Points, Apex_Points)
  - Bone Level Annotations/{id}.json  (Bone_Lines polylines)
  - Masks (Radiograph-wise)/{id}.png  (combined tooth mask)

and produces four target masks per image:
  - {id}_cej.png       Gaussian circles at each CEJ point
  - {id}_apex.png      Gaussian circles at each apex point
  - {id}_bone_line.png Polyline rendering of alveolar bone crest
  - {id}_tooth.png     Copy of the radiograph-wise tooth mask

These are saved under  data/bone_targets/{split}/  alongside symlinks (or
copies) of the source images in  data/bone_images/{split}/.

Usage:
    python scripts/05_convert_denpar.py          # uses config.yaml paths
    python scripts/05_convert_denpar.py --vis 5   # also save 5 overlay PNGs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def draw_point_heatmap(
    shape: tuple[int, int],
    points: list[list[float]],
    radius: int = 6,
) -> np.ndarray:
    """Render small filled circles at each point on a blank mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    for pt in points:
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    return mask


def draw_bone_lines(
    shape: tuple[int, int],
    bone_lines: list[list[list[float]]],
    thickness: int = 4,
) -> np.ndarray:
    """Render bone-crest polylines on a blank mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    for line in bone_lines:
        if len(line) < 2:
            continue
        pts = np.array(line, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    return mask


def create_overlay(
    image: np.ndarray,
    cej_mask: np.ndarray,
    apex_mask: np.ndarray,
    bone_mask: np.ndarray,
    tooth_mask: np.ndarray,
) -> np.ndarray:
    """Create a color overlay for visual inspection."""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Tooth mask in semi-transparent blue
    blue_overlay = np.zeros_like(vis)
    blue_overlay[:, :, 0] = tooth_mask
    vis = cv2.addWeighted(vis, 1.0, blue_overlay, 0.25, 0)

    # Bone crest line in green
    vis[bone_mask > 127] = [0, 255, 0]

    # CEJ points in red
    vis[cej_mask > 127] = [0, 0, 255]

    # Apex points in yellow
    vis[apex_mask > 127] = [0, 255, 255]

    return vis


def process_split(
    split_name: str,
    src_dir: Path,
    out_images_dir: Path,
    out_targets_dir: Path,
    point_radius: int,
    line_thickness: int,
    vis_count: int,
    vis_dir: Path | None,
) -> dict:
    """Process one split (Training / Validation / Testing)."""
    img_dir = src_dir / "Images"
    kp_dir = src_dir / "Key Points Annotations"
    bl_dir = src_dir / "Bone Level Annotations"
    mask_dir = src_dir / "Masks (Radiograph-wise)"

    if not img_dir.exists():
        print(f"  [skip] {img_dir} does not exist")
        return {"total": 0, "ok": 0, "skipped": 0}

    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    stats = {"total": len(image_files), "ok": 0, "skipped": 0}

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_targets_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    vis_saved = 0

    for img_path in tqdm(image_files, desc=f"  {split_name}", unit="img"):
        stem = img_path.stem
        kp_path = kp_dir / f"{stem}.json"
        bl_path = bl_dir / f"{stem}.json"
        rmask_path = mask_dir / f"{stem}.png"

        if not kp_path.exists() or not bl_path.exists():
            stats["skipped"] += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            stats["skipped"] += 1
            continue
        h, w = image.shape[:2]

        with open(kp_path) as f:
            kp_data = json.load(f)
        with open(bl_path) as f:
            bl_data = json.load(f)

        cej_points = kp_data.get("CEJ_Points", [])
        apex_points = kp_data.get("Apex_Points", [])
        bone_lines = bl_data.get("Bone_Lines", [])

        cej_mask = draw_point_heatmap((h, w), cej_points, radius=point_radius)
        apex_mask = draw_point_heatmap((h, w), apex_points, radius=point_radius)
        bone_mask = draw_bone_lines((h, w), bone_lines, thickness=line_thickness)

        tooth_mask = np.zeros((h, w), dtype=np.uint8)
        if rmask_path.exists():
            tm = cv2.imread(str(rmask_path), cv2.IMREAD_GRAYSCALE)
            if tm is not None:
                if tm.shape[:2] != (h, w):
                    tm = cv2.resize(tm, (w, h), interpolation=cv2.INTER_NEAREST)
                tooth_mask = tm

        # Save image copy
        out_img = out_images_dir / f"{stem}.jpg"
        if not out_img.exists():
            shutil.copy2(img_path, out_img)

        # Save target masks
        cv2.imwrite(str(out_targets_dir / f"{stem}_cej.png"), cej_mask)
        cv2.imwrite(str(out_targets_dir / f"{stem}_apex.png"), apex_mask)
        cv2.imwrite(str(out_targets_dir / f"{stem}_bone_line.png"), bone_mask)
        cv2.imwrite(str(out_targets_dir / f"{stem}_tooth.png"), tooth_mask)

        stats["ok"] += 1

        # Optional visual overlay
        if vis_dir and vis_saved < vis_count:
            overlay = create_overlay(image, cej_mask, apex_mask, bone_mask, tooth_mask)
            cv2.imwrite(str(vis_dir / f"{stem}_overlay.png"), overlay)
            vis_saved += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert DenPAR annotations to training masks")
    parser.add_argument("--vis", type=int, default=0, help="Number of overlay visualizations to save per split")
    parser.add_argument("--point-radius", type=int, default=6, help="Radius of CEJ/apex circles in pixels")
    parser.add_argument("--line-thickness", type=int, default=4, help="Thickness of bone crest polyline in pixels")
    args = parser.parse_args()

    cfg = load_config()
    bone_cfg = cfg.get("bone_dataset", {})
    bone_root = Path(bone_cfg.get("root", "../bone-dataset")).resolve()

    output_base = ROOT / "data" / "bone_landmark"

    splits = {
        "train": bone_root / "Training",
        "val": bone_root / "Validation",
        "test": bone_root / "Testing",
    }

    print(f"Bone dataset root: {bone_root}")
    print(f"Output directory:  {output_base}")
    print()

    all_stats = {}
    for split_name, src_dir in splits.items():
        print(f"Processing {split_name}...")
        vis_dir = output_base / "visualizations" / split_name if args.vis > 0 else None
        stats = process_split(
            split_name=split_name,
            src_dir=src_dir,
            out_images_dir=output_base / "images" / split_name,
            out_targets_dir=output_base / "targets" / split_name,
            point_radius=args.point_radius,
            line_thickness=args.line_thickness,
            vis_count=args.vis,
            vis_dir=vis_dir,
        )
        all_stats[split_name] = stats
        print(f"  {stats['ok']}/{stats['total']} converted, {stats['skipped']} skipped")

    print()
    print("Done. Summary:")
    for split_name, stats in all_stats.items():
        print(f"  {split_name:6s}: {stats['ok']:4d} images")

    total = sum(s["ok"] for s in all_stats.values())
    print(f"  {'total':6s}: {total:4d} images")


if __name__ == "__main__":
    main()
