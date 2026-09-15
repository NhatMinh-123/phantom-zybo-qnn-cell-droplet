#!/usr/bin/env python3
"""Calibrate QNN-cell proposals guarded by an integer radial image response."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.calibrate_15micro_dual_branch as base
from scripts.compare_qnn_video_roi import infer, load_runtime, preprocess_roi
from scripts.dual_branch_radial_v2 import radial_response


@dataclass(frozen=True)
class GuardConfig:
    low_confidence: float
    high_confidence: float
    response_threshold: int
    contrast_threshold: int
    support_radius: int


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
        default=ROOT / "reports/15micro_dual_branch_guarded_v3",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


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
        response, contrast = radial_response(gray)
        tensor = preprocess_roi(
            image,
            width=runtime.input_width,
            height=runtime.input_height,
            device=device,
        )
        detections, _, _ = infer(runtime, tensor, device)
        qnn_cells = [item for item in detections if item.class_id == 0]
        truth = base.read_labels(
            args.data / split / "labels" / f"{image_path.stem}.txt"
        )
        records.append((image_path.name, response, contrast, qnn_cells, truth[0]))
    return records


def local_support(
    response: np.ndarray,
    contrast: np.ndarray,
    box: tuple[float, float, float, float],
    radius: int,
) -> tuple[int, int]:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    x = int(round(cx * (response.shape[1] - 1)))
    y = int(round(cy * (response.shape[0] - 1)))
    x1 = max(0, x - radius)
    x2 = min(response.shape[1], x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(response.shape[0], y + radius + 1)
    local_response = response[y1:y2, x1:x2]
    if local_response.size == 0:
        return -32768, 0
    flat_index = int(np.argmax(local_response))
    local_y, local_x = np.unravel_index(flat_index, local_response.shape)
    return (
        int(local_response[local_y, local_x]),
        int(contrast[y1 + local_y, x1 + local_x]),
    )


def select_cells(record, config: GuardConfig):
    _, response, contrast, qnn_cells, _ = record
    accepted = []
    guarded = 0
    rejected = 0
    for item in qnn_cells:
        if item.confidence >= config.high_confidence:
            accepted.append(item.box)
            continue
        if item.confidence < config.low_confidence:
            continue
        radial, local_contrast = local_support(
            response, contrast, item.box, config.support_radius
        )
        if (
            radial >= config.response_threshold
            and local_contrast >= config.contrast_threshold
        ):
            accepted.append(item.box)
            guarded += 1
        else:
            rejected += 1
    return accepted, guarded, rejected


def evaluate(records, config: GuardConfig) -> dict[str, object]:
    baseline_counts = (0, 0, 0)
    guarded_counts = (0, 0, 0)
    accepted_low = 0
    rejected_low = 0
    for record in records:
        truth = record[4]
        baseline = [
            item.box for item in record[3]
            if item.confidence >= config.high_confidence
        ]
        guarded, accepted, rejected = select_cells(record, config)
        baseline_counts = base.add_counts(
            baseline_counts, base.match_counts(baseline, truth)
        )
        guarded_counts = base.add_counts(
            guarded_counts, base.match_counts(guarded, truth)
        )
        accepted_low += accepted
        rejected_low += rejected
    return {
        "config": asdict(config),
        "baseline": base.metrics(baseline_counts),
        "guarded": base.metrics(guarded_counts),
        "low_confidence_proposals": {
            "accepted": accepted_low,
            "rejected": rejected_low,
        },
    }


def selection_key(result: dict[str, object]) -> tuple[float, float, float, float]:
    guarded = result["guarded"]
    baseline = result["baseline"]
    assert isinstance(guarded, dict) and isinstance(baseline, dict)
    precision = float(guarded["precision"])
    recall = float(guarded["recall"])
    f1 = float(guarded["f1"])
    baseline_precision = float(baseline["precision"])
    precision_floor = max(0.75, baseline_precision - 0.08)
    penalty = min(1.0, precision / max(precision_floor, 1e-12))
    return f1 * penalty, f1, recall, precision


def object_code(confidence: float, output_scale: float) -> int:
    logit = math.log(confidence / (1.0 - confidence))
    return int(round(logit / output_scale))


def main() -> None:
    args = parse_args()
    device = base.resolve_device(args.device)
    runtime = load_runtime("15micro W4A6", args.checkpoint, args.postprocess, device)
    high_confidence = float(runtime.thresholds[0])
    # Decode once at the lowest proposed cell threshold; each sweep point then
    # filters the same quantized QNN candidates without rerunning the network.
    runtime.thresholds = (0.30, runtime.thresholds[1])
    validation = collect_split("valid", args, runtime, device)

    sweep = []
    for low_confidence in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.88, 0.92):
        for response_threshold in (20, 24, 28, 32, 36, 40, 44, 48):
            for contrast_threshold in (8, 12, 16, 20, 24):
                for support_radius in (0, 1, 2, 3, 4):
                    sweep.append(
                        evaluate(
                            validation,
                            GuardConfig(
                                low_confidence=low_confidence,
                                high_confidence=high_confidence,
                                response_threshold=response_threshold,
                                contrast_threshold=contrast_threshold,
                                support_radius=support_radius,
                            ),
                        )
                    )
    selected = max(sweep, key=selection_key)
    selected_config = GuardConfig(**selected["config"])

    test = collect_split("test", args, runtime, device)
    held_out = evaluate(test, selected_config)
    output_scale = 0.030286800116300583
    report = {
        "selection_policy": (
            "maximize validation cell F1 while penalizing precision below "
            "max(0.75, baseline precision - 0.08); test evaluated once"
        ),
        "validation_images": len(validation),
        "test_images": len(test),
        "selected_validation": selected,
        "held_out_test": held_out,
        "quantized_thresholds": {
            "output_scale": output_scale,
            "cell_low_object_code": object_code(
                selected_config.low_confidence, output_scale
            ),
            "cell_high_object_code": object_code(
                selected_config.high_confidence, output_scale
            ),
        },
        "implementation_contract": {
            "input": "96x96 UINT8 grayscale already consumed by QNN",
            "fusion": (
                "cell QNN proposals at high threshold pass; lower proposals pass "
                "only when integer radius-3 radial response confirms them"
            ),
            "droplet": "unchanged QNN path and threshold",
            "classical_only_boxes": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
