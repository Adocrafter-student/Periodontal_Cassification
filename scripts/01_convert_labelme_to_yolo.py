#!/usr/bin/env python3
"""Convert the Niihhaa LabelMe dataset to YOLO instance-segmentation format.

Reads all 3 folders of the Niihhaa dataset, converts polygon annotations to
YOLO-seg label files, copies images, performs a train/val split, and writes
the dataset.yaml consumed by Ultralytics training.

Usage:
    python scripts/01_convert_labelme_to_yolo.py [--config config.yaml]
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

LABEL_TO_INDEX = {
    "molar": 0,
    "premolar": 1,
    "canine": 2,
    "lateral incisor": 3,
    "lateral_incisor": 3,
    "central incisor": 4,
    "central_incisor": 4,
}

CLASS_NAMES = ["molar", "premolar", "canine", "lateral_incisor", "central_incisor"]


def parse_labelme_json(json_path: Path) -> Tuple[List[dict], int, int, str]:
    """Return (shapes, image_height, image_width, image_filename)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    img_h = data["imageHeight"]
    img_w = data["imageWidth"]

    img_path_raw = data.get("imagePath", "")
    img_filename = Path(img_path_raw.replace("\\", "/")).name

    return shapes, img_h, img_w, img_filename


def shapes_to_yolo_seg(
    shapes: List[dict], img_w: int, img_h: int
) -> Tuple[List[str], Dict[str, int]]:
    """Convert LabelMe shapes to YOLO-seg label lines.

    Also returns a mapping of group_id (FDI number) -> class_index for the
    sidecar FDI annotation file.
    """
    lines: List[str] = []
    fdi_map: Dict[str, int] = {}

    for shape in shapes:
        label = shape.get("label", "").strip().lower()
        cls_idx = LABEL_TO_INDEX.get(label)
        if cls_idx is None:
            continue

        if shape.get("shape_type") != "polygon":
            continue

        points = shape.get("points", [])
        if len(points) < 3:
            continue

        norm_coords: List[str] = []
        for x, y in points:
            nx = max(0.0, min(1.0, x / img_w))
            ny = max(0.0, min(1.0, y / img_h))
            norm_coords.append(f"{nx:.6f}")
            norm_coords.append(f"{ny:.6f}")

        line = f"{cls_idx} " + " ".join(norm_coords)
        lines.append(line)

        group_id = shape.get("group_id")
        if group_id is not None:
            fdi_map[str(group_id)] = cls_idx

    return lines, fdi_map


def find_image_file(images_dir: Path, expected_name: str) -> Path | None:
    """Find the actual image file, handling case-insensitive extension matching."""
    candidate = images_dir / expected_name
    if candidate.exists():
        return candidate

    stem = Path(expected_name).stem
    for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate

    return None


def process_folder(
    folder_root: Path,
    annotations_dir_name: str,
    images_dir_name: str,
) -> List[dict]:
    """Process one folder of the Niihhaa dataset.

    Returns a list of dicts:
        {"json_path", "image_path", "yolo_lines", "fdi_map", "stem"}
    """
    ann_dir = folder_root / annotations_dir_name
    img_dir = folder_root / images_dir_name

    if not ann_dir.exists():
        print(f"  [WARN] Annotations dir not found: {ann_dir}")
        return []

    records = []
    json_files = sorted(ann_dir.glob("*.json"))

    for jf in json_files:
        try:
            shapes, img_h, img_w, img_filename = parse_labelme_json(jf)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"  [WARN] Skipping {jf.name}: {e}")
            continue

        image_path = find_image_file(img_dir, img_filename)
        if image_path is None:
            print(f"  [WARN] Image not found for {jf.name} (expected {img_filename})")
            continue

        yolo_lines, fdi_map = shapes_to_yolo_seg(shapes, img_w, img_h)
        if not yolo_lines:
            print(f"  [WARN] No valid annotations in {jf.name}")
            continue

        records.append({
            "json_path": str(jf),
            "image_path": str(image_path),
            "yolo_lines": yolo_lines,
            "fdi_map": fdi_map,
            "stem": jf.stem,
            "folder": folder_root.name,
        })

    return records


def write_split(
    records: List[dict],
    split_name: str,
    output_root: Path,
) -> None:
    """Write images and labels for one split (train or val)."""
    img_out = output_root / split_name / "images"
    lbl_out = output_root / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for rec in records:
        src_img = Path(rec["image_path"])
        unique_stem = f"{rec['folder']}_{rec['stem']}"
        dst_img = img_out / (unique_stem + src_img.suffix.lower())
        dst_lbl = lbl_out / (unique_stem + ".txt")

        shutil.copy2(src_img, dst_img)

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(rec["yolo_lines"]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Niihhaa to YOLO-seg")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    niihhaa_root_raw = cfg["dataset"]["niihhaa_root"]
    niihhaa_root = Path(niihhaa_root_raw)
    if not niihhaa_root.is_absolute():
        niihhaa_root = (project_root / niihhaa_root).resolve()

    output_root_raw = cfg["dataset"]["yolo_output"]
    output_root = Path(output_root_raw)
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()

    val_ratio = cfg["dataset"]["val_ratio"]
    seed = cfg["dataset"]["random_seed"]
    random.seed(seed)

    print(f"Niihhaa root : {niihhaa_root}")
    print(f"YOLO output  : {output_root}")
    print()

    all_records: List[dict] = []

    for folder_cfg in cfg["dataset"]["folders"]:
        folder_name = folder_cfg["name"]
        ann_dir = folder_cfg["annotations_dir"]
        img_dir = folder_cfg["images_dir"]
        folder_path = niihhaa_root / folder_name

        print(f"Processing {folder_name} ...")
        records = process_folder(folder_path, ann_dir, img_dir)
        print(f"  Found {len(records)} valid image-annotation pairs")
        all_records.extend(records)

    print(f"\nTotal records: {len(all_records)}")

    if not all_records:
        print("ERROR: No records found. Check paths in config.yaml.")
        sys.exit(1)

    folder_labels = [r["folder"] for r in all_records]
    train_recs, val_recs = train_test_split(
        all_records,
        test_size=val_ratio,
        random_state=seed,
        stratify=folder_labels,
    )
    print(f"Train: {len(train_recs)}  |  Val: {len(val_recs)}")

    if output_root.exists():
        shutil.rmtree(output_root)

    print("\nWriting train split ...")
    write_split(train_recs, "train", output_root)
    print("Writing val split ...")
    write_split(val_recs, "val", output_root)

    dataset_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"\nDataset YAML written to {yaml_path}")

    fdi_annotations = {}
    for rec in all_records:
        unique_stem = f"{rec['folder']}_{rec['stem']}"
        fdi_annotations[unique_stem] = rec["fdi_map"]

    fdi_path = output_root / "fdi_annotations.json"
    with open(fdi_path, "w", encoding="utf-8") as f:
        json.dump(fdi_annotations, f, indent=2)
    print(f"FDI annotations written to {fdi_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
