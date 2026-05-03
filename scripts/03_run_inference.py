#!/usr/bin/env python3
"""Run the trained YOLOv8-seg tooth model on OPG images.

For each image:
  1. Detect + segment teeth
  2. Assign FDI tooth numbers via spatial heuristics
  3. Produce expanded tooth+bone crops
  4. Save crops, masks, and a structured JSON manifest

Usage:
    python scripts/03_run_inference.py [--config config.yaml]
           [--input INPUT_DIR] [--output OUTPUT_DIR] [--weights WEIGHTS]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.crop_utils import crop_tooth_region, save_mask_crop
from utils.fdi_numbering import assign_fdi_numbers

CLASS_NAMES = ["molar", "premolar", "canine", "lateral_incisor", "central_incisor"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def process_single_image(
    model: YOLO,
    image_path: Path,
    crops_dir: Path,
    masks_dir: Path,
    expand_ratio: float,
    conf: float,
    iou: float,
    imgsz: int,
) -> dict:
    """Run inference on one image and return structured metadata."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [WARN] Could not read {image_path}")
        return {}

    img_h, img_w = image.shape[:2]
    stem = image_path.stem

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return {
            "image_id": image_path.name,
            "image_width": img_w,
            "image_height": img_h,
            "teeth": [],
        }

    result = results[0]
    boxes = result.boxes
    masks_data = result.masks

    detections = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        confidence = float(boxes.conf[i].item())
        bbox = boxes.xyxy[i].tolist()

        tooth_type = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"

        det = {
            "bbox": bbox,
            "tooth_type": tooth_type,
            "confidence": confidence,
            "class_id": cls_id,
            "_mask_idx": i,
            "_det_id": f"{i:03d}",
        }
        detections.append(det)

    detections = assign_fdi_numbers(detections, img_w, img_h)

    teeth_output = []
    for det in detections:
        tooth_num = det.get("tooth_number", "unknown")
        det_id = det.pop("_det_id", "000")
        mask_idx = det.pop("_mask_idx", None)
        det.pop("class_id", None)

        crop_filename = f"{stem}_det_{det_id}_tooth_{tooth_num}.png"
        mask_filename = f"{stem}_det_{det_id}_tooth_{tooth_num}_mask.png"

        crop_img, expanded_bbox = crop_tooth_region(
            image, det["bbox"], expand_ratio
        )
        cv2.imwrite(str(crops_dir / crop_filename), crop_img)

        if masks_data is not None and mask_idx is not None:
            full_mask = masks_data.data[mask_idx].cpu().numpy()
            full_mask_resized = cv2.resize(
                full_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
            )
            mask_binary = (full_mask_resized > 0.5).astype(np.uint8) * 255
            mask_crop, _ = save_mask_crop(
                mask_binary, det["bbox"], img_w, img_h, expand_ratio
            )
            cv2.imwrite(str(masks_dir / mask_filename), mask_crop)
        else:
            mask_filename = None

        teeth_output.append({
            "detection_id": det_id,
            "tooth_number": tooth_num,
            "tooth_type": det["tooth_type"],
            "fdi_quadrant": det.get("fdi_quadrant", "unknown"),
            "bbox": [round(v, 1) for v in det["bbox"]],
            "expanded_bbox": expanded_bbox,
            "crop_path": f"crops/{crop_filename}",
            "mask_path": f"masks/{mask_filename}" if mask_filename else None,
            "confidence": round(det["confidence"], 4),
        })

    return {
        "image_id": image_path.name,
        "image_width": img_w,
        "image_height": img_h,
        "num_teeth_detected": len(teeth_output),
        "teeth": teeth_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tooth detection inference")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    icfg = cfg["inference"]

    def resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (project_root / pp).resolve()

    weights = resolve(args.weights or icfg["weights"])
    input_path = resolve(args.input or icfg["input_dir"])
    output_dir = resolve(args.output or icfg["output_dir"])

    if not weights.exists():
        print(f"ERROR: Weights not found at {weights}")
        print("Train a model first with 02_train_model.py")
        sys.exit(1)

    if not input_path.exists():
        print(f"ERROR: Input path not found at {input_path}")
        sys.exit(1)

    crops_dir = output_dir / "crops"
    masks_dir = output_dir / "masks"
    crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Weights : {weights}")
    print(f"Input   : {input_path}")
    print(f"Output  : {output_dir}")
    print()

    model = YOLO(str(weights))

    if input_path.is_file():
        image_files = [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    else:
        image_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

    if not image_files:
        print(f"No images found at {input_path}")
        sys.exit(0)

    print(f"Found {len(image_files)} images\n")

    all_results = []
    for img_path in tqdm(image_files, desc="Processing"):
        result = process_single_image(
            model=model,
            image_path=img_path,
            crops_dir=crops_dir,
            masks_dir=masks_dir,
            expand_ratio=icfg["bbox_expand_ratio"],
            conf=icfg["confidence"],
            iou=icfg["iou_threshold"],
            imgsz=icfg["imgsz"],
        )
        if result:
            all_results.append(result)

            per_image_json = output_dir / f"{img_path.stem}.json"
            with open(per_image_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    total_teeth = sum(r.get("num_teeth_detected", 0) for r in all_results)
    print(f"\nDone! Processed {len(all_results)} images, detected {total_teeth} teeth total.")
    print(f"Manifest: {manifest_path}")
    print(f"Crops:    {crops_dir}")
    print(f"Masks:    {masks_dir}")


if __name__ == "__main__":
    main()
