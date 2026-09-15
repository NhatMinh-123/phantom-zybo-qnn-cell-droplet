#!/usr/bin/env python3
"""Calibrate a hardware-oriented classical cell branch and QNN fusion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_qnn_video_roi import infer, load_runtime, preprocess_roi
from scripts.dual_branch_classical import (
    DetectorConfig,
    detect_small_particles,
    fuse_cell_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "dataset/15micro_roi120_center_v1"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/15micro/qnn_w4a6_96_roi120_v1/last.pt",
    )
    parser.add_argument(
        "--postprocess",
        type=Path,
        default=ROOT / "reports/15micro_qnn_roi120_v1/postprocess_balanced_epoch29.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/15micro_dual_branch_roi120_v1",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def read_labels(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    output: dict[int, list[tuple[float, float, float, float]]] = {0: [], 1: []}
    for row in path.read_text(encoding="utf-8").splitlines():
        fields = row.split()
        if not fields:
            continue
        class_id = int(fields[0])
        cx, cy, width, height = map(float, fields[1:])
        output[class_id].append(
            (
                cx - width / 2.0,
                cy - height / 2.0,
                cx + width / 2.0,
                cy + height / 2.0,
            )
        )
    return output


def match_counts(
    predictions: Sequence[Sequence[float]],
    truth: Sequence[Sequence[float]],
    *,
    maximum_center_distance: float = 0.065,
) -> tuple[int, int, int]:
    remaining = set(range(len(truth)))
    true_positive = 0
    for prediction in predictions:
        px = (float(prediction[0]) + float(prediction[2])) / 2.0
        py = (float(prediction[1]) + float(prediction[3])) / 2.0
        choices = []
        for index in remaining:
            item = truth[index]
            tx = (float(item[0]) + float(item[2])) / 2.0
            ty = (float(item[1]) + float(item[3])) / 2.0
            distance = float(np.hypot(px - tx, py - ty))
            if distance <= maximum_center_distance:
                choices.append((distance, index))
        if choices:
            _, index = min(choices)
            remaining.remove(index)
            true_positive += 1
    return true_positive, len(predictions) - true_positive, len(remaining)


def metrics(counts: tuple[int, int, int]) -> dict[str, float | int]:
    true_positive, false_positive, false_negative = counts
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def add_counts(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def collect_split(split: str, args: argparse.Namespace, runtime, device):
    records = []
    image_dir = args.data / split / "images"
    for image_path in sorted(image_dir.iterdir()):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")
        gray = cv2.cvtColor(
            cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2GRAY,
        )
        tensor = preprocess_roi(
            image,
            width=runtime.input_width,
            height=runtime.input_height,
            device=device,
        )
        detections, _, _ = infer(runtime, tensor, device)
        qnn_cells = [item.box for item in detections if item.class_id == 0]
        qnn_droplets = [item.box for item in detections if item.class_id == 1]
        truth = read_labels(
            args.data / split / "labels" / f"{image_path.stem}.txt"
        )
        records.append((image_path.name, gray, qnn_cells, qnn_droplets, truth[0]))
    return records


def evaluate(records, config: DetectorConfig) -> dict[str, object]:
    totals = {
        "qnn": (0, 0, 0),
        "classical": (0, 0, 0),
        "fusion": (0, 0, 0),
    }
    candidate_counts = []
    for _, gray, qnn_cells, qnn_droplets, truth_cells in records:
        classical = detect_small_particles(
            gray,
            config,
            droplet_boxes=qnn_droplets,
        )
        fused = fuse_cell_boxes(qnn_cells, classical)
        candidate_counts.append(len(classical))
        totals["qnn"] = add_counts(
            totals["qnn"], match_counts(qnn_cells, truth_cells)
        )
        totals["classical"] = add_counts(
            totals["classical"],
            match_counts([item.box for item in classical], truth_cells),
        )
        totals["fusion"] = add_counts(
            totals["fusion"], match_counts(fused, truth_cells)
        )
    return {
        "config": asdict(config),
        "qnn": metrics(totals["qnn"]),
        "classical": metrics(totals["classical"]),
        "fusion": metrics(totals["fusion"]),
        "classical_candidates_per_image": {
            "mean": float(np.mean(candidate_counts)),
            "maximum": int(max(candidate_counts, default=0)),
        },
    }


def selection_key(result: dict[str, object]) -> tuple[float, float, float]:
    fusion = result["fusion"]
    assert isinstance(fusion, dict)
    # Precision guard prevents an apparent recall gain caused by accepting noise.
    precision = float(fusion["precision"])
    recall = float(fusion["recall"])
    f1 = float(fusion["f1"])
    guarded_f1 = f1 if precision >= 0.70 else f1 * precision / 0.70
    return guarded_f1, recall, precision


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runtime = load_runtime("15micro W4A6", args.checkpoint, args.postprocess, device)
    validation = collect_split("valid", args, runtime, device)
    sweep = []
    for response_threshold in range(5, 25, 2):
        for contrast_threshold in (12, 16, 20, 24, 28, 32):
            for minimum_distance in (4, 5, 6, 7):
                config = DetectorConfig(
                    response_threshold=response_threshold,
                    local_contrast_threshold=contrast_threshold,
                    minimum_distance=minimum_distance,
                )
                sweep.append(evaluate(validation, config))
    selected = max(sweep, key=selection_key)
    selected_config = DetectorConfig(**selected["config"])
    test = collect_split("test", args, runtime, device)
    test_result = evaluate(test, selected_config)
    report = {
        "selection_policy": (
            "maximize validation fusion F1 with precision >= 0.70 guard; "
            "test is evaluated once after selection"
        ),
        "validation_images": len(validation),
        "test_images": len(test),
        "selected_validation": selected,
        "held_out_test": test_result,
        "implementation_contract": {
            "input": "96x96 UINT8 grayscale",
            "operators": [
                "3x3 box mean",
                "9x9 box mean",
                "integer subtract",
                "7x7 local min/max",
                "3x3 local maximum",
                "threshold and distance suppression",
            ],
            "droplet_source": "QNN droplet detections",
            "fusion_scope": "cell detections only",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
