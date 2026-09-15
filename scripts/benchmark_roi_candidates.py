#!/usr/bin/env python3
"""Benchmark rectangular ROI candidates for the cell/droplet YOLO model.

The grouped dataset contains 640x640 crops made from a 256x256 camera ROI.
Each candidate below removes unused context without rescaling the objects. Class
thresholds are selected only on validation data and are then frozen for test.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
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


@dataclass(frozen=True)
class Candidate:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class Detection:
    class_id: int
    confidence: float
    box: np.ndarray


CANDIDATES = (
    Candidate("full_640x640", 0, 0, 640, 640),
    Candidate("band_640x384", 0, 160, 640, 544),
    Candidate("compact_512x384", 64, 160, 576, 544),
    Candidate("compact_448x352", 96, 176, 544, 528),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--benchmark-frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_data(data_path: Path) -> tuple[Path, dict[int, str]]:
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    raw_names = data["names"]
    names = (
        {int(index): str(name) for index, name in raw_names.items()}
        if isinstance(raw_names, dict)
        else {index: str(name) for index, name in enumerate(raw_names)}
    )
    return root, names


def image_paths(root: Path, split: str) -> list[Path]:
    return sorted(
        path
        for path in (root / split / "images").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_labels(path: Path, width: int, height: int) -> list[Detection]:
    labels: list[Detection] = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="ascii").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        class_id = int(values[0])
        cx, cy, box_width, box_height = map(float, values[1:])
        labels.append(
            Detection(
                class_id,
                1.0,
                np.array(
                    [
                        (cx - box_width / 2.0) * width,
                        (cy - box_height / 2.0) * height,
                        (cx + box_width / 2.0) * width,
                        (cy + box_height / 2.0) * height,
                    ],
                    dtype=float,
                ),
            )
        )
    return labels


def crop_labels(labels: list[Detection], candidate: Candidate) -> list[Detection]:
    cropped: list[Detection] = []
    for label in labels:
        center_x = (label.box[0] + label.box[2]) / 2.0
        center_y = (label.box[1] + label.box[3]) / 2.0
        if not (
            candidate.x1 <= center_x < candidate.x2
            and candidate.y1 <= center_y < candidate.y2
        ):
            continue
        box = label.box.copy()
        box[[0, 2]] = np.clip(box[[0, 2]], candidate.x1, candidate.x2) - candidate.x1
        box[[1, 3]] = np.clip(box[[1, 3]], candidate.y1, candidate.y2) - candidate.y1
        if box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0:
            cropped.append(Detection(label.class_id, label.confidence, box))
    return cropped


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_counts(
    truths: list[Detection], predictions: list[Detection], iou_threshold: float
) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            if prediction.class_id != truth.class_id:
                continue
            overlap = box_iou(prediction.box, truth.box)
            if overlap >= iou_threshold:
                candidates.append((overlap, prediction_index, truth_index))
    matched_predictions: set[int] = set()
    matched_truths: set[int] = set()
    for _, prediction_index, truth_index in sorted(candidates, reverse=True):
        if prediction_index in matched_predictions or truth_index in matched_truths:
            continue
        matched_predictions.add(prediction_index)
        matched_truths.add(truth_index)
    true_positive = len(matched_predictions)
    return true_positive, len(predictions) - true_positive, len(truths) - true_positive


def score_samples(
    samples: list[tuple[list[Detection], list[Detection]]],
    class_ids: list[int],
    thresholds: dict[int, float],
    iou_threshold: float,
) -> dict[str, object]:
    by_class: dict[int, dict[str, int]] = {
        class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in class_ids
    }
    for truths, predictions in samples:
        for class_id in class_ids:
            truth_subset = [item for item in truths if item.class_id == class_id]
            prediction_subset = [
                item
                for item in predictions
                if item.class_id == class_id
                and item.confidence >= thresholds[class_id]
            ]
            tp, fp, fn = match_counts(truth_subset, prediction_subset, iou_threshold)
            by_class[class_id]["tp"] += tp
            by_class[class_id]["fp"] += fp
            by_class[class_id]["fn"] += fn

    def metrics(counts: dict[str, int]) -> dict[str, float | int]:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**counts, "precision": precision, "recall": recall, "f1": f1}

    overall_counts = {
        key: sum(values[key] for values in by_class.values())
        for key in ("tp", "fp", "fn")
    }
    return {
        "overall": metrics(overall_counts),
        "classes": {class_id: metrics(counts) for class_id, counts in by_class.items()},
    }


def choose_thresholds(
    samples: list[tuple[list[Detection], list[Detection]]],
    class_ids: list[int],
    iou_threshold: float,
    step: float,
) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    values = np.arange(0.05, 0.951, step)
    for class_id in class_ids:
        best = (-1.0, -1.0, 0.5)
        for value in values:
            result = score_samples(samples, [class_id], {class_id: float(value)}, iou_threshold)
            metrics = result["overall"]
            candidate = (float(metrics["f1"]), float(metrics["precision"]), float(value))
            if candidate > best:
                best = candidate
        thresholds[class_id] = best[2]
    return thresholds


def predict_split(
    model: YOLO,
    root: Path,
    split: str,
    candidate: Candidate,
    device: str,
    half: bool,
) -> tuple[list[tuple[list[Detection], list[Detection]]], dict[int, int]]:
    samples: list[tuple[list[Detection], list[Detection]]] = []
    retained: dict[int, int] = {}
    for image_path in image_paths(root, split):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        labels = read_labels(
            root / split / "labels" / f"{image_path.stem}.txt", width, height
        )
        truths = crop_labels(labels, candidate)
        for truth in truths:
            retained[truth.class_id] = retained.get(truth.class_id, 0) + 1
        crop = image[candidate.y1 : candidate.y2, candidate.x1 : candidate.x2]
        result = model.predict(
            crop,
            imgsz=(candidate.height, candidate.width),
            conf=0.01,
            iou=0.5,
            max_det=300,
            device=device,
            quantize=16 if half else None,
            verbose=False,
        )[0]
        predictions: list[Detection] = []
        if result.boxes is not None:
            for class_id, confidence, box in zip(
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
            ):
                predictions.append(Detection(int(class_id), float(confidence), box.copy()))
        samples.append((truths, predictions))
    return samples, retained


def source_geometry(candidate: Candidate) -> tuple[int, int, int, int]:
    base_x, base_y = 560, 342
    scale = 256.0 / 640.0
    return (
        int(round(base_x + candidate.x1 * scale)),
        int(round(base_y + candidate.y1 * scale)),
        int(round(candidate.width * scale)),
        int(round(candidate.height * scale)),
    )


def benchmark_video(
    model: YOLO,
    video_path: Path,
    candidate: Candidate,
    device: str,
    half: bool,
    frame_count: int,
    warmup_count: int,
) -> dict[str, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    roi_x, roi_y, roi_width, roi_height = source_geometry(candidate)
    times: list[float] = []
    processed = 0
    while processed < frame_count + warmup_count:
        ok, frame = capture.read()
        if not ok:
            break
        roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
        model_input = cv2.resize(
            roi, (candidate.width, candidate.height), interpolation=cv2.INTER_CUBIC
        )
        started = time.perf_counter()
        model.predict(
            model_input,
            imgsz=(candidate.height, candidate.width),
            conf=0.05,
            iou=0.5,
            max_det=300,
            device=device,
            quantize=16 if half else None,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if processed >= warmup_count:
            times.append(elapsed_ms)
        processed += 1
    capture.release()
    if not times:
        raise RuntimeError("No video frames were benchmarked")
    sorted_times = sorted(times)
    p95_index = min(len(sorted_times) - 1, int(round(0.95 * (len(sorted_times) - 1))))
    mean_ms = statistics.fmean(times)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times),
        "p95_ms": sorted_times[p95_index],
        "fps": 1000.0 / mean_ms,
    }


def flatten_metrics(
    candidate: Candidate,
    names: dict[int, str],
    thresholds: dict[int, float],
    result: dict[str, object],
    retained: dict[int, int],
    speed: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate": candidate.name,
        "input_width": candidate.width,
        "input_height": candidate.height,
        "pixel_ratio": candidate.width * candidate.height / (640.0 * 640.0),
        "mean_ms": speed["mean_ms"],
        "p95_ms": speed["p95_ms"],
        "fps": speed["fps"],
    }
    overall = result["overall"]
    for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
        row[key] = overall[key]
    for class_id, class_name in names.items():
        class_metrics = result["classes"][class_id]
        row[f"{class_name}_threshold"] = thresholds[class_id]
        row[f"{class_name}_retained"] = retained.get(class_id, 0)
        for key in ("precision", "recall", "f1"):
            row[f"{class_name}_{key}"] = class_metrics[key]
    return row


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root, names = load_data(args.data.resolve())
    class_ids = list(names)
    model = YOLO(str(args.model.resolve()))
    rows: list[dict[str, object]] = []
    detail: dict[str, object] = {}

    for candidate in CANDIDATES:
        print(f"Benchmarking {candidate.name} ...", flush=True)
        validation_samples, validation_retained = predict_split(
            model, root, "valid", candidate, args.device, args.half
        )
        thresholds = choose_thresholds(
            validation_samples, class_ids, args.iou, args.threshold_step
        )
        validation_result = score_samples(
            validation_samples, class_ids, thresholds, args.iou
        )
        test_samples, test_retained = predict_split(
            model, root, "test", candidate, args.device, args.half
        )
        test_result = score_samples(test_samples, class_ids, thresholds, args.iou)
        speed = benchmark_video(
            model,
            args.video.resolve(),
            candidate,
            args.device,
            args.half,
            args.benchmark_frames,
            args.warmup_frames,
        )
        rows.append(
            flatten_metrics(
                candidate, names, thresholds, test_result, test_retained, speed
            )
        )
        detail[candidate.name] = {
            "crop_in_640_image": [candidate.x1, candidate.y1, candidate.x2, candidate.y2],
            "source_roi_1280x800": list(source_geometry(candidate)),
            "thresholds": {names[key]: value for key, value in thresholds.items()},
            "validation_retained": {
                names[key]: validation_retained.get(key, 0) for key in class_ids
            },
            "test_retained": {names[key]: test_retained.get(key, 0) for key in class_ids},
            "validation": validation_result,
            "test": test_result,
            "speed": speed,
        }

    full_f1 = float(rows[0]["f1"])
    eligible = [
        row
        for row in rows
        if float(row["f1"]) >= full_f1 - 0.03
        and float(row["cell_recall"]) >= 0.75
        and float(row["droplet_recall"]) >= 0.95
    ]
    recommended = max(eligible or rows, key=lambda row: float(row["fps"]))
    detail["recommended"] = recommended["candidate"]

    csv_path = output / "roi_benchmark.csv"
    with csv_path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "roi_benchmark.json").write_text(
        json.dumps(detail, indent=2), encoding="ascii"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = [str(row["candidate"]).replace("_", "\n") for row in rows]
    axes[0].bar(labels, [float(row["f1"]) * 100.0 for row in rows], color="#1971c2")
    axes[0].plot(
        labels,
        [float(row["cell_recall"]) * 100.0 for row in rows],
        color="#e8590c",
        marker="o",
        label="Cell recall",
    )
    axes[0].plot(
        labels,
        [float(row["droplet_recall"]) * 100.0 for row in rows],
        color="#2b8a3e",
        marker="o",
        label="Droplet recall",
    )
    axes[0].set_ylabel("Test score (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Accuracy after validation-only threshold tuning")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [float(row["fps"]) for row in rows], color="#0b7285")
    axes[1].set_ylabel("Inference FPS on MX330")
    axes[1].set_title("Single-frame ROI inference, FP16")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Recommended ROI: {recommended['candidate']}")
    fig.tight_layout()
    fig.savefig(output / "roi_accuracy_speed.png", dpi=180)
    plt.close(fig)

    print(f"Recommended: {recommended['candidate']}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
