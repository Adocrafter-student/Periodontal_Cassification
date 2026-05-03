"""Full pipeline: run bone landmark model on OPG crops → PBL → staging.

Takes inference output from script 03 (crops + masks + JSON), runs the
trained bone-landmark U-Net on each crop, extracts CEJ/apex/bone-crest
positions from predicted heatmaps, computes PBL ratio, and produces
staging results.

Usage:
    python scripts/08_bone_landmark_staging.py --input periodontal_data/output --output periodontal_data/bone_staging
    python scripts/08_bone_landmark_staging.py --input periodontal_data/output --output periodontal_data/bone_staging --vis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bone_model.unet import build_model
from utils.bone_loss_analyzer import ToothAnalysis, ROOT_LENGTH_MM
from utils.periodontal_staging import stage_patient


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_bone_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {}).get("architecture", {})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    dice = ckpt.get("best_val_dice", "?")
    print(f"Loaded bone model from epoch {epoch}, val dice: {dice:.4f}" if isinstance(dice, float) else f"Loaded bone model from epoch {epoch}")
    return model


def predict_landmarks(
    model: torch.nn.Module,
    crop_path: str,
    device: torch.device,
    image_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Run bone model on a single crop, return predicted masks at original resolution.

    Returns (cej_mask, apex_mask, bone_mask, tooth_pred, (orig_h, orig_w))
    """
    image = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {crop_path}")

    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))[0].cpu().numpy()  # (4, H, W)

    # Resize predictions back to original crop size
    masks = []
    for c in range(4):
        m = cv2.resize(pred[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        masks.append(m)

    return masks[0], masks[1], masks[2], masks[3], (orig_h, orig_w)


def extract_points_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 10,
) -> list[tuple[int, int]]:
    """Extract point coordinates from a predicted heatmap via peak detection."""
    binary = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))

    # Filter points too close together (keep the one with higher heatmap value)
    if len(points) > 1:
        filtered = []
        used = set()
        scored = [(p, heatmap[p[1], p[0]]) for p in points]
        scored.sort(key=lambda x: -x[1])
        for pt, score in scored:
            too_close = False
            for fpt in filtered:
                dist = ((pt[0] - fpt[0]) ** 2 + (pt[1] - fpt[1]) ** 2) ** 0.5
                if dist < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered.append(pt)
        points = filtered

    return points


def extract_bone_crest_row(
    bone_mask: np.ndarray,
    col_center: int,
    search_width: int = 40,
    threshold: float = 0.2,
) -> int | None:
    """Find the bone crest row near a given column from the bone-line heatmap."""
    h, w = bone_mask.shape
    col_lo = max(0, col_center - search_width)
    col_hi = min(w, col_center + search_width)

    strip = bone_mask[:, col_lo:col_hi]
    row_max = strip.max(axis=1)

    rows_above_thresh = np.where(row_max > threshold)[0]
    if len(rows_above_thresh) == 0:
        return None
    return int(rows_above_thresh[np.argmax(row_max[rows_above_thresh])])


def analyze_tooth_with_model(
    crop_path: str,
    mask_path: str,
    tooth_number: str,
    tooth_type: str,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 512,
) -> ToothAnalysis:
    """Analyze a single tooth using the bone landmark model."""
    is_maxillary = int(tooth_number) < 30 if tooth_number.isdigit() else True

    result = ToothAnalysis(
        tooth_number=tooth_number,
        tooth_type=tooth_type,
        is_maxillary=is_maxillary,
    )

    try:
        cej_hm, apex_hm, bone_hm, tooth_hm, (orig_h, orig_w) = predict_landmarks(
            model, crop_path, device, image_size,
        )
    except FileNotFoundError as e:
        result.analysis_ok = False
        result.notes.append(str(e))
        return result

    # Load the YOLO segmentation mask for this tooth
    seg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if seg_mask is None:
        result.analysis_ok = False
        result.notes.append("Cannot read mask")
        return result

    # Find tooth centroid from segmentation mask
    mask_binary = (seg_mask > 127).astype(np.uint8)
    mask_cols = np.where(mask_binary.any(axis=0))[0]
    mask_rows = np.where(mask_binary.any(axis=1))[0]

    if len(mask_cols) == 0 or len(mask_rows) == 0:
        result.analysis_ok = False
        result.notes.append("Empty mask")
        return result

    col_center = int(np.mean(mask_cols))

    # Extract CEJ and apex points
    cej_points = extract_points_from_heatmap(cej_hm, threshold=0.25)
    apex_points = extract_points_from_heatmap(apex_hm, threshold=0.20)

    # Find the CEJ and apex closest to this tooth's center column
    def closest_to_col(points: list[tuple[int, int]], target_col: int) -> tuple[int, int] | None:
        if not points:
            return None
        return min(points, key=lambda p: abs(p[0] - target_col))

    cej_pt = closest_to_col(cej_points, col_center)
    apex_pt = closest_to_col(apex_points, col_center)

    if cej_pt is None or apex_pt is None:
        # Fallback: use mask extent
        if is_maxillary:
            cej_row = int(mask_rows[-1])  # crown at bottom
            apex_row = int(mask_rows[0])  # root at top
        else:
            cej_row = int(mask_rows[0])   # crown at top
            apex_row = int(mask_rows[-1]) # root at bottom

        if cej_pt is not None:
            cej_row = cej_pt[1]
        if apex_pt is not None:
            apex_row = apex_pt[1]

        result.notes.append("Partial landmark fallback (missing CEJ or apex)")
    else:
        cej_row = cej_pt[1]
        apex_row = apex_pt[1]

    root_length_px = abs(cej_row - apex_row)
    if root_length_px < 10:
        result.analysis_ok = False
        result.notes.append(f"Root too short ({root_length_px}px)")
        return result

    # Find bone crest from bone-line heatmap
    bone_crest_row = extract_bone_crest_row(bone_hm, col_center, search_width=50, threshold=0.15)

    if bone_crest_row is None:
        # No bone crest detected → assume healthy (crest at CEJ level)
        bone_crest_row = cej_row
        result.notes.append("No bone crest detected, assumed healthy")

    # Compute bone loss
    if is_maxillary:
        bone_loss_px = max(0, cej_row - bone_crest_row)
    else:
        bone_loss_px = max(0, bone_crest_row - cej_row)

    bone_loss_px = min(bone_loss_px, root_length_px)
    rbl_percent = (bone_loss_px / root_length_px) * 100.0
    rbl_percent = min(100.0, max(0.0, rbl_percent))

    avg_root_mm = ROOT_LENGTH_MM.get(tooth_type, 13.0)
    cal_mm = round((rbl_percent / 100.0) * avg_root_mm, 1)

    result.root_length_px = root_length_px
    result.cej_row = cej_row
    result.apex_row = apex_row
    result.bone_crest_row_mesial = bone_crest_row
    result.bone_crest_row_distal = bone_crest_row
    result.bone_loss_px = bone_loss_px
    result.rbl_percent = round(rbl_percent, 1)
    result.cal_mm = cal_mm
    result.bone_crest_position = (
        "coronal_third" if rbl_percent < 33 else
        "middle_third" if rbl_percent < 66 else
        "apical_third"
    )
    # Pattern: with single bone crest point, default to horizontal
    result.bone_loss_pattern = "horizontal"

    return result


def create_tooth_overlay(
    crop_path: str,
    cej_hm: np.ndarray,
    apex_hm: np.ndarray,
    bone_hm: np.ndarray,
    analysis: ToothAnalysis,
) -> np.ndarray:
    """Create a visual overlay for one tooth crop."""
    image = cv2.imread(crop_path)
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    vis = image.copy()

    # Bone crest heatmap in green
    bone_colored = np.zeros_like(vis)
    bone_colored[:, :, 1] = (bone_hm * 255).clip(0, 255).astype(np.uint8)
    vis = cv2.addWeighted(vis, 1.0, bone_colored, 0.5, 0)

    # CEJ points in red
    vis[(cej_hm > 0.3)] = [0, 0, 255]

    # Apex points in yellow
    vis[(apex_hm > 0.2)] = [0, 255, 255]

    # Draw CEJ and apex rows
    h, w = vis.shape[:2]
    cv2.line(vis, (0, analysis.cej_row), (w, analysis.cej_row), (0, 0, 200), 1)
    cv2.line(vis, (0, analysis.apex_row), (w, analysis.apex_row), (0, 200, 200), 1)
    cv2.line(vis, (0, analysis.bone_crest_row_mesial), (w, analysis.bone_crest_row_mesial), (0, 200, 0), 1)

    # Label
    label = f"T{analysis.tooth_number} RBL={analysis.rbl_percent:.0f}%"
    cv2.putText(vis, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def process_image(
    json_path: Path,
    base_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    output_dir: Path,
    save_vis: bool = False,
) -> dict:
    """Process one OPG image's inference output through bone landmark model."""
    with open(json_path) as f:
        data = json.load(f)

    image_id = data["image_id"]
    image_h = data["image_height"]
    image_w = data["image_width"]
    stem = Path(image_id).stem

    analyses = []
    detected_fdi = set()
    per_tooth_details = []

    vis_crops = []

    for tooth in data["teeth"]:
        tooth_num = tooth["tooth_number"]
        tooth_type = tooth["tooth_type"]
        crop_path = str(base_dir / tooth["crop_path"])
        mask_path = str(base_dir / tooth["mask_path"])

        detected_fdi.add(tooth_num)

        analysis = analyze_tooth_with_model(
            crop_path=crop_path,
            mask_path=mask_path,
            tooth_number=tooth_num,
            tooth_type=tooth_type,
            model=model,
            device=device,
            image_size=image_size,
        )
        analyses.append(analysis)

        detail = {
            "tooth_number": tooth_num,
            "tooth_type": tooth_type,
            "rbl_percent": analysis.rbl_percent,
            "cal_mm": analysis.cal_mm,
            "bone_crest_position": analysis.bone_crest_position,
            "analysis_ok": analysis.analysis_ok,
            "notes": analysis.notes,
        }
        per_tooth_details.append(detail)

        # Collect vis if requested
        if save_vis and analysis.analysis_ok:
            try:
                cej_hm, apex_hm, bone_hm, _, _ = predict_landmarks(model, crop_path, device, image_size)
                overlay = create_tooth_overlay(crop_path, cej_hm, apex_hm, bone_hm, analysis)
                vis_crops.append((tooth_num, overlay))
            except Exception:
                pass

    # Run staging
    staging = stage_patient(analyses, detected_fdi)

    # Save results
    result = {
        "image_id": image_id,
        "staging": {
            "stage": staging.stage,
            "severity_stage": staging.severity_stage,
            "tooth_loss_stage": staging.tooth_loss_stage,
            "complexity_stage": staging.complexity_stage,
            "max_cal_mm": staging.max_cal_mm,
            "max_rbl_percent": staging.max_rbl_percent,
            "extent": staging.extent,
            "teeth_analyzed": staging.teeth_analyzed,
            "teeth_with_bone_loss": staging.teeth_with_bone_loss,
            "bone_loss_pattern": staging.bone_loss_pattern,
        },
        "per_tooth": per_tooth_details,
        "missing_teeth": staging.missing_teeth,
    }

    out_json = output_dir / f"{stem}_bone_staging.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # Save vis grid
    if save_vis and vis_crops:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        grid = _make_grid(vis_crops, cols=6)
        cv2.imwrite(str(vis_dir / f"{stem}_bone_overlay.png"), grid)

    return result


def _make_grid(
    labeled_crops: list[tuple[str, np.ndarray]],
    cols: int = 6,
    cell_w: int = 200,
    cell_h: int = 250,
) -> np.ndarray:
    """Arrange tooth overlays in a grid."""
    n = len(labeled_crops)
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, (tooth_num, crop) in enumerate(labeled_crops):
        r, c = divmod(i, cols)
        resized = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        y0, x0 = r * cell_h, c * cell_w
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = resized

    return grid


def main():
    parser = argparse.ArgumentParser(description="Bone landmark model staging pipeline")
    parser.add_argument("--input", type=str, default="periodontal_data/output",
                        help="Directory with script 03 inference output")
    parser.add_argument("--output", type=str, default="periodontal_data/bone_staging",
                        help="Output directory for staging results")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--vis", action="store_true", help="Save per-tooth visual overlays")
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

    input_dir = (ROOT / args.input).resolve()
    output_dir = (ROOT / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Device:  {device}")
    print()

    model = load_bone_model(weights_path, device)

    json_files = sorted([f for f in input_dir.glob("*.json") if f.name != "manifest.json"])
    print(f"Found {len(json_files)} image(s) to process\n")

    for json_path in json_files:
        result = process_image(
            json_path=json_path,
            base_dir=input_dir,
            model=model,
            device=device,
            image_size=image_size,
            output_dir=output_dir,
            save_vis=args.vis,
        )

        stg = result["staging"]
        img = result["image_id"]
        print(
            f"  [{img}] Stage {stg['stage']} | "
            f"RBL={stg['max_rbl_percent']:.1f}% | "
            f"CAL={stg['max_cal_mm']}mm | "
            f"{stg['extent']} | "
            f"{stg['teeth_analyzed']} teeth | "
            f"{stg['teeth_with_bone_loss']} with bone loss"
        )

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
