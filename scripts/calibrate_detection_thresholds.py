#!/usr/bin/env python3
"""Choose per-class confidence thresholds on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate_cell_droplet_yolo import (
    Detection,
    match_detections,
    read_ground_truth,
    resolve_split,
    safe_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate cell and droplet thresholds on validation images."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--max-conf", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def fresh_detection(detection: Detection) -> Detection:
    return Detection(
        class_id=detection.class_id,
        confidence=detection.confidence,
        box=detection.box.copy(),
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_path = args.data.resolve()
    validation_images, validation_labels = resolve_split(data_path, "valid")

    model = YOLO(str(args.model.resolve()))
    names = {int(index): str(name) for index, name in model.names.items()}
    results = model.predict(
        source=str(validation_images),
        imgsz=args.imgsz,
        conf=max(0.001, args.min_conf / 2.0),
        iou=args.iou,
        max_det=300,
        device=args.device,
        batch=args.batch,
        verbose=False,
    )

    samples: list[tuple[list[Detection], list[Detection]]] = []
    for result in results:
        image = result.orig_img
        height, width = image.shape[:2]
        ground_truth = read_ground_truth(
            validation_labels / f"{Path(result.path).stem}.txt", width, height
        )
        predictions: list[Detection] = []
        if result.boxes is not None:
            for class_id, confidence, box in zip(
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
            ):
                predictions.append(Detection(int(class_id), float(confidence), box.copy()))
        samples.append((ground_truth, predictions))

    thresholds = np.arange(
        args.min_conf, args.max_conf + args.step / 2.0, args.step, dtype=float
    )
    sweep_rows: list[dict[str, float | int | str]] = []
    best_by_class: dict[int, dict[str, float | int | str]] = {}

    for class_id, class_name in names.items():
        class_rows: list[dict[str, float | int | str]] = []
        for threshold in thresholds:
            true_positive = 0
            false_positive = 0
            false_negative = 0
            for ground_truth, predictions in samples:
                truth_subset = [
                    fresh_detection(item)
                    for item in ground_truth
                    if item.class_id == class_id
                ]
                prediction_subset = [
                    fresh_detection(item)
                    for item in predictions
                    if item.class_id == class_id and item.confidence >= threshold
                ]
                tp_count, fp_count, fn_count = match_detections(
                    truth_subset, prediction_subset, args.iou
                )
                true_positive += tp_count
                false_positive += fp_count
                false_negative += fn_count

            precision = safe_ratio(true_positive, true_positive + false_positive)
            recall = safe_ratio(true_positive, true_positive + false_negative)
            f1 = safe_ratio(2.0 * precision * recall, precision + recall)
            row: dict[str, float | int | str] = {
                "class_id": class_id,
                "class": class_name,
                "threshold": round(float(threshold), 4),
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            class_rows.append(row)
            sweep_rows.append(row)

        # Prefer the higher threshold when F1 values tie to reduce false alarms.
        best_by_class[class_id] = max(
            class_rows,
            key=lambda row: (float(row["f1"]), float(row["threshold"])),
        )

    sweep_path = output / "confidence_threshold_sweep.csv"
    with sweep_path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    configuration = {
        "model": str(args.model.resolve()),
        "data": str(data_path),
        "image_size": args.imgsz,
        "nms_iou": args.iou,
        "confidence": {
            names[class_id]: round(float(row["threshold"]), 4)
            for class_id, row in best_by_class.items()
        },
        "validation_metrics_at_threshold": {
            names[class_id]: {
                key: row[key]
                for key in ("tp", "fp", "fn", "precision", "recall", "f1")
            }
            for class_id, row in best_by_class.items()
        },
        "roi": {
            "reference_width": 1280,
            "reference_height": 800,
            "x": 560,
            "y": 342,
            "size": 256,
            "count_line_fraction": 0.65,
        },
    }
    config_path = output / "deployment_config.json"
    config_path.write_text(json.dumps(configuration, indent=2), encoding="ascii")

    fig, axes = plt.subplots(1, len(names), figsize=(12, 4.8), squeeze=False)
    for plot_index, (class_id, class_name) in enumerate(names.items()):
        axis = axes[0, plot_index]
        class_rows = [row for row in sweep_rows if int(row["class_id"]) == class_id]
        x_values = [float(row["threshold"]) for row in class_rows]
        axis.plot(x_values, [float(row["precision"]) for row in class_rows], label="Precision")
        axis.plot(x_values, [float(row["recall"]) for row in class_rows], label="Recall")
        axis.plot(x_values, [float(row["f1"]) for row in class_rows], label="F1", linewidth=2.5)
        best = best_by_class[class_id]
        axis.axvline(float(best["threshold"]), color="#c92a2a", linestyle="--")
        axis.set_title(
            f"{class_name}: best conf={float(best['threshold']):.2f}, "
            f"F1={float(best['f1']):.3f}"
        )
        axis.set_xlabel("Confidence threshold")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Score")
    axes[0, -1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "confidence_thresholds.png", dpi=180)
    plt.close(fig)

    print(f"Validation images: {len(samples)}")
    for class_id, row in best_by_class.items():
        print(
            f"{names[class_id]}: conf={float(row['threshold']):.2f}, "
            f"P={float(row['precision']):.4f}, R={float(row['recall']):.4f}, "
            f"F1={float(row['f1']):.4f}, TP={row['tp']} FP={row['fp']} FN={row['fn']}"
        )
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
