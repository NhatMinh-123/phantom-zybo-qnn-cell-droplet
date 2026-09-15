#!/usr/bin/env python3
"""Evaluate the grouped cell/droplet detector and build report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Detection:
    class_id: int
    confidence: float
    box: np.ndarray
    matched: bool = False


@dataclass
class ImageRecord:
    image: str
    annotated_path: Path
    ground_truth: int
    predictions: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best.pt on the grouped test split."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--threshold-config",
        type=Path,
        default=None,
        help="Optional deployment_config.json with per-class confidence values.",
    )
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--sample-count", type=int, default=6)
    return parser.parse_args()


def resolve_split(data_path: Path, split: str) -> tuple[Path, Path]:
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    key = "val" if split == "valid" else split
    relative = Path(data[key])
    image_dir = relative if relative.is_absolute() else root / relative
    label_dir = image_dir.parent / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {split} split: images={image_dir}, labels={label_dir}"
        )
    return image_dir, label_dir


def read_ground_truth(path: Path, width: int, height: int) -> list[Detection]:
    detections: list[Detection] = []
    if not path.exists():
        return detections
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        center_x, center_y, box_width, box_height = map(float, parts[1:])
        x1 = (center_x - box_width / 2.0) * width
        y1 = (center_y - box_height / 2.0) * height
        x2 = (center_x + box_width / 2.0) * width
        y2 = (center_y + box_height / 2.0) * height
        detections.append(
            Detection(class_id, 1.0, np.array([x1, y1, x2, y2], dtype=float))
        )
    return detections


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection_x1 = max(first[0], second[0])
    intersection_y1 = max(first[1], second[1])
    intersection_x2 = min(first[2], second[2])
    intersection_y2 = min(first[3], second[3])
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_detections(
    ground_truth: list[Detection], predictions: list[Detection], iou_threshold: float
) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(ground_truth):
            if prediction.class_id != truth.class_id:
                continue
            overlap = box_iou(prediction.box, truth.box)
            if overlap >= iou_threshold:
                candidates.append((overlap, prediction_index, truth_index))

    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    for _, prediction_index, truth_index in sorted(candidates, reverse=True):
        if prediction_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions.add(prediction_index)
        matched_truth.add(truth_index)
        predictions[prediction_index].matched = True
        ground_truth[truth_index].matched = True

    true_positive = len(matched_predictions)
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    return true_positive, false_positive, false_negative


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def draw_label(
    image: np.ndarray,
    text: str,
    box: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = min(width - 1, max(0, x1))
    y1 = min(height - 1, max(0, y1))
    x2 = min(width - 1, max(0, x2))
    y2 = min(height - 1, max(0, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    )
    label_y1 = max(0, y1 - text_height - baseline - 4)
    label_y2 = label_y1 + text_height + baseline + 4
    cv2.rectangle(
        image,
        (x1, label_y1),
        (min(width - 1, x1 + text_width + 6), label_y2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 3, label_y2 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def annotate_image(
    image: np.ndarray,
    ground_truth: list[Detection],
    predictions: list[Detection],
    names: dict[int, str],
    summary: str,
) -> np.ndarray:
    annotated = image.copy()
    for truth in ground_truth:
        color = (0, 165, 255) if not truth.matched else (210, 140, 20)
        prefix = "MISS" if not truth.matched else "GT"
        draw_label(
            annotated,
            f"{prefix} {names[truth.class_id]}",
            truth.box,
            color,
            1,
        )
    for prediction in predictions:
        color = (30, 190, 40) if prediction.matched else (30, 30, 230)
        prefix = "TP" if prediction.matched else "FP"
        draw_label(
            annotated,
            f"{prefix} {names[prediction.class_id]} {prediction.confidence:.2f}",
            prediction.box,
            color,
            2,
        )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        summary,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return annotated


def write_contact_sheet(records: list[ImageRecord], output: Path) -> None:
    if not records:
        return
    tile_width = 480
    tile_height = 480
    columns = 3
    tiles: list[np.ndarray] = []
    for record in records:
        image = cv2.imread(str(record.annotated_path))
        if image is None:
            continue
        image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        caption = np.full((44, tile_width, 3), 245, dtype=np.uint8)
        text = (
            f"{record.image}  TP={record.true_positive} FP={record.false_positive} "
            f"FN={record.false_negative} F1={record.f1:.3f}"
        )
        cv2.putText(
            caption,
            text,
            (7, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        tiles.append(np.vstack([caption, image]))
    if not tiles:
        return
    blank = np.full_like(tiles[0], 245)
    rows = (len(tiles) + columns - 1) // columns
    tiles.extend([blank] * (rows * columns - len(tiles)))
    sheet = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    cv2.imwrite(str(output), sheet)


def write_metric_plot(rows: list[dict[str, float | str]], output: Path) -> None:
    labels = [str(row["scope"]) for row in rows]
    metric_names = ["precision", "recall", "f1", "map50", "map50_95"]
    display_names = ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95"]
    x_positions = np.arange(len(labels))
    bar_width = 0.15
    colors = ["#1570a6", "#2f9e44", "#e67700", "#7048e8", "#c92a2a"]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for metric_index, (metric_name, display_name, color) in enumerate(
        zip(metric_names, display_names, colors)
    ):
        values = [float(row[metric_name]) * 100.0 for row in rows]
        positions = x_positions + (metric_index - 2) * bar_width
        bars = axis.bar(positions, values, bar_width, label=display_name, color=color)
        axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axis.set_title("Cell and droplet detector performance on grouped test split")
    axis.set_ylabel("Percent")
    axis.set_ylim(0, 110)
    axis.set_xticks(x_positions, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    model = YOLO(str(model_path))
    names = {int(index): str(name) for index, name in model.names.items()}
    class_thresholds = {class_id: args.conf for class_id in names}
    if args.threshold_config is not None:
        threshold_config = json.loads(
            args.threshold_config.resolve().read_text(encoding="ascii")
        )
        configured = threshold_config.get("confidence", {})
        class_thresholds = {
            class_id: float(configured.get(class_name, args.conf))
            for class_id, class_name in names.items()
        }
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        project=str(output),
        name="ultralytics",
        exist_ok=True,
        verbose=True,
    )

    mean_precision, mean_recall, map50, map50_95 = metrics.box.mean_results()
    metric_rows: list[dict[str, float | str]] = [
        {
            "scope": "overall",
            "precision": float(mean_precision),
            "recall": float(mean_recall),
            "f1": safe_ratio(
                2.0 * mean_precision * mean_recall, mean_precision + mean_recall
            ),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }
    ]
    for result_index, class_id in enumerate(metrics.box.ap_class_index.astype(int)):
        precision, recall, class_map50, class_map50_95 = metrics.box.class_result(
            result_index
        )
        metric_rows.append(
            {
                "scope": names[class_id],
                "precision": float(precision),
                "recall": float(recall),
                "f1": safe_ratio(2.0 * precision * recall, precision + recall),
                "map50": float(class_map50),
                "map50_95": float(class_map50_95),
            }
        )

    metric_csv = output / "metrics_summary.csv"
    with metric_csv.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    write_metric_plot(metric_rows, output / "performance_metrics.png")

    test_image_dir, test_label_dir = resolve_split(data_path, "test")
    test_images = sorted(
        path for path in test_image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not test_images:
        raise RuntimeError(f"No test images found in {test_image_dir}")

    results = model.predict(
        source=str(test_image_dir),
        imgsz=args.imgsz,
        conf=min(class_thresholds.values()),
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        batch=args.batch,
        verbose=False,
    )
    image_records: list[ImageRecord] = []
    class_errors = {
        class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in names
    }
    speed_rows: list[dict[str, float]] = []

    for result in results:
        image_path = Path(result.path)
        image = result.orig_img
        height, width = image.shape[:2]
        ground_truth = read_ground_truth(
            test_label_dir / f"{image_path.stem}.txt", width, height
        )
        predictions: list[Detection] = []
        if result.boxes is not None:
            for class_id, confidence, box in zip(
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
            ):
                class_id = int(class_id)
                confidence = float(confidence)
                if confidence >= class_thresholds[class_id]:
                    predictions.append(Detection(class_id, confidence, box.copy()))

        true_positive, false_positive, false_negative = match_detections(
            ground_truth, predictions, args.iou
        )
        precision = safe_ratio(true_positive, true_positive + false_positive)
        recall = safe_ratio(true_positive, true_positive + false_negative)
        f1 = safe_ratio(2.0 * precision * recall, precision + recall)
        summary = (
            f"TP {true_positive} | FP {false_positive} | FN {false_negative} | "
            f"P {precision:.2f} R {recall:.2f} F1 {f1:.2f}"
        )
        annotated = annotate_image(image, ground_truth, predictions, names, summary)
        annotated_path = predictions_dir / image_path.name
        cv2.imwrite(str(annotated_path), annotated)
        image_records.append(
            ImageRecord(
                image=image_path.name,
                annotated_path=annotated_path,
                ground_truth=len(ground_truth),
                predictions=len(predictions),
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
        for prediction in predictions:
            class_errors[prediction.class_id]["tp" if prediction.matched else "fp"] += 1
        for truth in ground_truth:
            if not truth.matched:
                class_errors[truth.class_id]["fn"] += 1
        speed_rows.append(
            {
                "preprocess_ms": float(result.speed.get("preprocess", 0.0)),
                "inference_ms": float(result.speed.get("inference", 0.0)),
                "postprocess_ms": float(result.speed.get("postprocess", 0.0)),
            }
        )

    image_csv = output / "image_metrics.csv"
    with image_csv.open("w", newline="", encoding="ascii") as csv_file:
        fieldnames = [field for field in ImageRecord.__dataclass_fields__ if field != "annotated_path"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in image_records:
            writer.writerow(
                {
                    field: getattr(record, field)
                    for field in fieldnames
                }
            )

    best_records = sorted(
        image_records,
        key=lambda item: (-item.f1, item.false_positive + item.false_negative, -item.true_positive),
    )[: args.sample_count]
    worst_records = sorted(
        image_records,
        key=lambda item: (item.f1, -(item.false_positive + item.false_negative), -item.ground_truth),
    )[: args.sample_count]
    write_contact_sheet(best_records, output / "best_predictions.jpg")
    write_contact_sheet(worst_records, output / "difficult_predictions.jpg")

    error_csv = output / "threshold_error_summary.csv"
    with error_csv.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["class", "tp", "fp", "fn"])
        writer.writeheader()
        for class_id, counts in class_errors.items():
            writer.writerow({"class": names[class_id], **counts})

    average_speed = {
        key: float(np.mean([row[key] for row in speed_rows]))
        for key in ("preprocess_ms", "inference_ms", "postprocess_ms")
    }
    average_speed["total_ms"] = sum(average_speed.values())
    average_speed["estimated_fps"] = safe_ratio(1000.0, average_speed["total_ms"])
    (output / "runtime_summary.json").write_text(
        json.dumps(average_speed, indent=2), encoding="ascii"
    )
    (output / "applied_thresholds.json").write_text(
        json.dumps(
            {names[class_id]: threshold for class_id, threshold in class_thresholds.items()},
            indent=2,
        ),
        encoding="ascii",
    )

    print(f"Model: {model_path}")
    print(f"Test images: {len(image_records)}")
    print(
        f"Overall: P={mean_precision:.4f} R={mean_recall:.4f} "
        f"mAP50={map50:.4f} mAP50-95={map50_95:.4f}"
    )
    print(
        f"Runtime: {average_speed['total_ms']:.2f} ms/image, "
        f"estimated {average_speed['estimated_fps']:.2f} FPS"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
