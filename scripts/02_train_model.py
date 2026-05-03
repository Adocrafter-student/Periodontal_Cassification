#!/usr/bin/env python3
"""Train YOLOv8-seg on the converted Niihhaa tooth dataset.

Usage:
    python scripts/02_train_model.py [--config config.yaml] [--epochs N]
"""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg tooth model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count from config")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size from config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    dcfg = cfg["dataset"]

    yolo_output = Path(dcfg["yolo_output"])
    if not yolo_output.is_absolute():
        yolo_output = (project_root / yolo_output).resolve()

    dataset_yaml = yolo_output / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {dataset_yaml}. "
            "Run 01_convert_labelme_to_yolo.py first."
        )

    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    batch = args.batch if args.batch is not None else tcfg["batch"]
    device = tcfg.get("device", 0)
    if device == "" or device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    project_dir = Path(tcfg["project"])
    if not project_dir.is_absolute():
        project_dir = project_root / project_dir

    print(f"Dataset YAML : {dataset_yaml}")
    print(f"Base model   : {tcfg['model']}")
    print(f"Epochs       : {epochs}")
    print(f"Image size   : {tcfg['imgsz']}")
    print(f"Batch size   : {batch}")
    print(f"Patience     : {tcfg['patience']}")
    print(f"Device       : {device}" + (f" ({torch.cuda.get_device_name(device)})" if isinstance(device, int) and torch.cuda.is_available() else ""))
    print(f"Output dir   : {project_dir / tcfg['name']}")
    print()

    model = YOLO(tcfg["model"])

    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=tcfg["imgsz"],
        batch=batch,
        patience=tcfg["patience"],
        device=device,
        project=str(project_dir),
        name=tcfg["name"],
        exist_ok=tcfg.get("exist_ok", True),
        # Augmentation defaults are good; override specifics here if needed
        flipud=0.0,         # no vertical flip (jaw orientation matters)
        fliplr=0.5,         # horizontal flip is fine
        mosaic=1.0,
        scale=0.5,
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
    )

    print("\nTraining complete!")
    print(f"Best weights: {project_dir / tcfg['name'] / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
