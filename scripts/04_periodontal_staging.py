#!/usr/bin/env python3
"""Run periodontal staging on tooth-detection output.

Reads the per-image JSON produced by 03_run_inference.py, analyzes each tooth
crop + mask for bone loss, and applies the 2017 Tonetti classification.

Usage:
    # Analyze all images in a directory
    python scripts/04_periodontal_staging.py --input periodontal_data/output --output periodontal_data/staging

    # Analyze a single image
    python scripts/04_periodontal_staging.py --input periodontal_data/output/ds2_train_6.json --output periodontal_data/staging
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.bone_loss_analyzer import analyze_tooth, ToothAnalysis
from utils.periodontal_staging import stage_patient, StagingResult


def process_image_json(
    json_path: Path,
    base_dir: Path,
    output_dir: Path,
) -> dict | None:
    """Run staging on one image's inference JSON."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id = data.get("image_id", json_path.stem)
    image_height = data["image_height"]
    teeth = data.get("teeth", [])

    if not teeth:
        print(f"  [{image_id}] No teeth found, skipping")
        return None

    analyses: list[ToothAnalysis] = []
    detected_fdi: set[str] = set()

    for tooth in teeth:
        tooth_number = tooth["tooth_number"]
        tooth_type = tooth["tooth_type"]
        bbox = tooth["bbox"]
        crop_rel = tooth.get("crop_path", "")
        mask_rel = tooth.get("mask_path", "")

        if not crop_rel or not mask_rel:
            continue

        crop_path = base_dir / crop_rel
        mask_path = base_dir / mask_rel

        if not crop_path.exists() or not mask_path.exists():
            continue

        bbox_y_center = (bbox[1] + bbox[3]) / 2.0
        detected_fdi.add(tooth_number)

        result = analyze_tooth(
            crop_path=str(crop_path),
            mask_path=str(mask_path),
            tooth_number=tooth_number,
            tooth_type=tooth_type,
            bbox_y_center=bbox_y_center,
            image_height=image_height,
        )
        analyses.append(result)

    staging = stage_patient(analyses, detected_fdi)

    per_tooth = []
    for a in analyses:
        entry = {
            "tooth_number": a.tooth_number,
            "tooth_type": a.tooth_type,
            "is_maxillary": a.is_maxillary,
            "rbl_percent": a.rbl_percent,
            "cal_mm": a.cal_mm,
            "bone_crest_position": a.bone_crest_position,
            "bone_loss_pattern": a.bone_loss_pattern,
            "root_length_px": a.root_length_px,
            "bone_loss_px": a.bone_loss_px,
            "analysis_ok": a.analysis_ok,
        }
        if a.notes:
            entry["notes"] = a.notes
        per_tooth.append(entry)

    output = {
        "image_id": image_id,
        "staging": {
            "stage": staging.stage,
            "severity_stage": staging.severity_stage,
            "tooth_loss_stage": staging.tooth_loss_stage,
            "complexity_stage": staging.complexity_stage,
            "max_cal_mm": staging.max_cal_mm,
            "max_rbl_percent": staging.max_rbl_percent,
            "rbl_location": staging.rbl_location,
            "tooth_loss_count": staging.tooth_loss_count,
            "extent": staging.extent,
            "percent_teeth_affected": staging.percent_teeth_affected,
            "bone_loss_pattern": staging.bone_loss_pattern,
            "teeth_analyzed": staging.teeth_analyzed,
            "teeth_with_bone_loss": staging.teeth_with_bone_loss,
        },
        "per_tooth_analysis": per_tooth,
        "missing_teeth": staging.missing_teeth,
        "analysis_notes": staging.analysis_notes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(image_id).stem}_staging.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    stage_str = f"Stage {_roman(staging.stage)}" if staging.stage > 0 else "Healthy"
    print(f"  [{image_id}] {stage_str} | "
          f"CAL={staging.max_cal_mm}mm | RBL={staging.max_rbl_percent}% | "
          f"{staging.extent} | {staging.teeth_analyzed} teeth analyzed | "
          f"Missing: {staging.tooth_loss_count}")

    return output


def _roman(n: int) -> str:
    return {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV"}.get(n, str(n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodontal staging (2017 classification)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to inference output directory or single .json file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to staging output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    if not input_path.exists():
        print(f"ERROR: Input not found at {input_path}")
        sys.exit(1)

    if input_path.is_file() and input_path.suffix == ".json":
        json_files = [input_path]
        base_dir = input_path.parent
    else:
        json_files = sorted(input_path.glob("*.json"))
        json_files = [f for f in json_files if f.name != "manifest.json"]
        base_dir = input_path

    if not json_files:
        print(f"No inference JSON files found at {input_path}")
        sys.exit(0)

    print(f"Input     : {input_path}")
    print(f"Output    : {output_dir}")
    print(f"JSON files: {len(json_files)}")
    print()

    all_results = []
    for jf in json_files:
        result = process_image_json(jf, base_dir, output_dir)
        if result:
            all_results.append(result)

    if all_results:
        summary_path = output_dir / "staging_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary written to {summary_path}")

    print(f"Done! Staged {len(all_results)} images.")


if __name__ == "__main__":
    main()
