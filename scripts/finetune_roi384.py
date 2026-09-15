#!/usr/bin/env python3
"""Fine-tune YOLO11n on the final compact ROI without touching the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="yolo11n_roi384_finetune")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model.resolve()))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=15,
        imgsz=384,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        amp=True,
        seed=42,
        deterministic=True,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.03,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.05,
        hsv_v=0.10,
        plots=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
