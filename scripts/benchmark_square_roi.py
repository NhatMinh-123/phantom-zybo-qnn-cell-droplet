#!/usr/bin/env python3
"""Benchmark GPU-friendly square inputs and compact camera ROIs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmark_roi_candidates import (
    Candidate,
    Detection,
    choose_thresholds,
    crop_labels,
    image_paths,
    load_data,
    read_labels,
    score_samples,
    source_geometry,
)


@dataclass(frozen=True)
class Mode:
    name: str
    crop: Candidate
    canvas_size: int


MODES = (
    Mode("full_640", Candidate("full", 0, 0, 640, 640), 640),
    Mode("full_512", Candidate("full", 0, 0, 640, 640), 512),
    Mode("band_pad512", Candidate("band", 0, 160, 640, 544), 512),
    Mode("compact_pad512", Candidate("compact", 64, 160, 576, 544), 512),
    Mode("compact_pad448", Candidate("compact", 96, 176, 544, 528), 448),
    Mode("compact_pad384", Candidate("compact", 128, 192, 512, 480), 384),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--benchmark-frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--modes",
        default="",
        help="Optional comma-separated subset, for example full_640,compact_pad384.",
    )
    return parser.parse_args()


def prepare_canvas(
    image: np.ndarray, labels: list[Detection], mode: Mode
) -> tuple[np.ndarray, list[Detection]]:
    crop = image[mode.crop.y1 : mode.crop.y2, mode.crop.x1 : mode.crop.x2]
    cropped_labels = crop_labels(labels, mode.crop)
    scale = min(mode.canvas_size / mode.crop.width, mode.canvas_size / mode.crop.height)
    resized_width = max(1, int(round(mode.crop.width * scale)))
    resized_height = max(1, int(round(mode.crop.height * scale)))
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    median_color = np.median(crop.reshape(-1, crop.shape[2]), axis=0).astype(np.uint8)
    canvas = np.empty((mode.canvas_size, mode.canvas_size, 3), dtype=np.uint8)
    canvas[:] = median_color
    offset_x = (mode.canvas_size - resized_width) // 2
    offset_y = (mode.canvas_size - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    transformed: list[Detection] = []
    for label in cropped_labels:
        box = label.box.copy() * scale
        box[[0, 2]] += offset_x
        box[[1, 3]] += offset_y
        transformed.append(Detection(label.class_id, label.confidence, box))
    return canvas, transformed


def predict_split(
    model: YOLO,
    root: Path,
    split: str,
    mode: Mode,
    device: str,
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
        canvas, truths = prepare_canvas(image, labels, mode)
        for truth in truths:
            retained[truth.class_id] = retained.get(truth.class_id, 0) + 1
        result = model.predict(
            canvas,
            imgsz=mode.canvas_size,
            conf=0.01,
            iou=0.5,
            max_det=300,
            device=device,
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


def prepare_video_input(frame: np.ndarray, mode: Mode) -> np.ndarray:
    roi_x, roi_y, roi_width, roi_height = source_geometry(mode.crop)
    roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    training_scale_crop = cv2.resize(
        roi, (mode.crop.width, mode.crop.height), interpolation=cv2.INTER_CUBIC
    )
    canvas, _ = prepare_canvas(
        training_scale_crop,
        [],
        Mode(mode.name, Candidate("local", 0, 0, mode.crop.width, mode.crop.height), mode.canvas_size),
    )
    return canvas


def benchmark_video(
    model: YOLO,
    video_path: Path,
    mode: Mode,
    device: str,
    frame_count: int,
    warmup_count: int,
) -> dict[str, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    times: list[float] = []
    processed = 0
    while processed < frame_count + warmup_count:
        ok, frame = capture.read()
        if not ok:
            break
        started = time.perf_counter()
        model_input = prepare_video_input(frame, mode)
        model.predict(
            model_input,
            imgsz=mode.canvas_size,
            conf=0.05,
            iou=0.5,
            max_det=300,
            device=device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if processed >= warmup_count:
            times.append(elapsed_ms)
        processed += 1
    capture.release()
    sorted_times = sorted(times)
    p95_index = min(len(sorted_times) - 1, int(round(0.95 * (len(sorted_times) - 1))))
    mean_ms = statistics.fmean(times)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times),
        "p95_ms": sorted_times[p95_index],
        "fps": 1000.0 / mean_ms,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root, names = load_data(args.data.resolve())
    class_ids = list(names)
    requested = {value.strip() for value in args.modes.split(",") if value.strip()}
    modes = tuple(mode for mode in MODES if not requested or mode.name in requested)
    if not modes:
        raise SystemExit(f"No modes matched: {sorted(requested)}")
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}

    for mode in modes:
        print(f"Benchmarking {mode.name} ...", flush=True)
        model = YOLO(str(args.model.resolve()))
        validation_samples, validation_retained = predict_split(
            model, root, "valid", mode, args.device
        )
        thresholds = choose_thresholds(
            validation_samples, class_ids, args.iou, args.threshold_step
        )
        validation_metrics = score_samples(
            validation_samples, class_ids, thresholds, args.iou
        )
        test_samples, test_retained = predict_split(model, root, "test", mode, args.device)
        test_metrics = score_samples(test_samples, class_ids, thresholds, args.iou)
        speed = benchmark_video(
            model,
            args.video.resolve(),
            mode,
            args.device,
            args.benchmark_frames,
            args.warmup_frames,
        )
        overall = test_metrics["overall"]
        row: dict[str, object] = {
            "mode": mode.name,
            "canvas": mode.canvas_size,
            "source_roi_x": source_geometry(mode.crop)[0],
            "source_roi_y": source_geometry(mode.crop)[1],
            "source_roi_width": source_geometry(mode.crop)[2],
            "source_roi_height": source_geometry(mode.crop)[3],
            "mean_ms": speed["mean_ms"],
            "p95_ms": speed["p95_ms"],
            "fps": speed["fps"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
        }
        for class_id, class_name in names.items():
            metrics = test_metrics["classes"][class_id]
            row[f"{class_name}_threshold"] = thresholds[class_id]
            row[f"{class_name}_retained"] = test_retained.get(class_id, 0)
            row[f"{class_name}_precision"] = metrics["precision"]
            row[f"{class_name}_recall"] = metrics["recall"]
            row[f"{class_name}_f1"] = metrics["f1"]
        rows.append(row)
        details[mode.name] = {
            "crop_in_640_image": [mode.crop.x1, mode.crop.y1, mode.crop.x2, mode.crop.y2],
            "source_roi_1280x800": list(source_geometry(mode.crop)),
            "canvas_size": mode.canvas_size,
            "thresholds": {names[key]: value for key, value in thresholds.items()},
            "validation_retained": {
                names[key]: validation_retained.get(key, 0) for key in class_ids
            },
            "test_retained": {names[key]: test_retained.get(key, 0) for key in class_ids},
            "validation": validation_metrics,
            "test": test_metrics,
            "speed": speed,
        }
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_f1 = float(rows[0]["f1"])
    eligible = [
        row
        for row in rows
        if float(row["f1"]) >= full_f1 - 0.03
        and float(row["cell_recall"]) >= 0.75
        and float(row["droplet_recall"]) >= 0.95
    ]
    recommended = max(eligible or rows, key=lambda row: float(row["fps"]))
    details["recommended"] = recommended["mode"]

    with (output / "square_roi_benchmark.csv").open(
        "w", newline="", encoding="ascii"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "square_roi_benchmark.json").write_text(
        json.dumps(details, indent=2), encoding="ascii"
    )

    labels = [str(row["mode"]).replace("_", "\n") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(labels, [float(row["f1"]) * 100.0 for row in rows], color="#1971c2")
    axes[0].plot(
        labels,
        [float(row["cell_recall"]) * 100.0 for row in rows],
        marker="o",
        color="#e8590c",
        label="Cell recall",
    )
    axes[0].plot(
        labels,
        [float(row["droplet_recall"]) * 100.0 for row in rows],
        marker="o",
        color="#2b8a3e",
        label="Droplet recall",
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Test score (%)")
    axes[0].set_title("Detection quality")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [float(row["fps"]) for row in rows], color="#0b7285")
    axes[1].set_ylabel("End-to-end ROI FPS on MX330")
    axes[1].set_title("Preprocess + YOLO + NMS")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Recommended PC mode: {recommended['mode']}")
    fig.tight_layout()
    fig.savefig(output / "square_roi_accuracy_speed.png", dpi=180)
    plt.close(fig)

    print(f"Recommended: {recommended['mode']}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
