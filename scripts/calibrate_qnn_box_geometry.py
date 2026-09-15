#!/usr/bin/env python3
"""Calibrate droplet box geometry on validation predictions only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.dataset import YoloDetectionDataset
from qnn.detection import Detection, count_matches, decode_predictions
from qnn.evaluate_qat import collect_predictions
from qnn.model import TinyQuantDetector, config_from_dict


DEFAULT_CHECKPOINT = (
    ROOT / "models" / "qnn_cell_droplet_v2_w4a6_square192_grouped" / "best.pt"
)
DEFAULT_POSTPROCESS = (
    ROOT
    / "reports"
    / "qnn_droplet_postprocess"
    / "w4a6_square192"
    / "postprocess_config.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "qnn_box_geometry_w4a6_square192"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--postprocess", type=Path, default=DEFAULT_POSTPROCESS)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "dataset" / "cell_droplet_roi384_grouped"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def apply_calibration(
    all_detections: list[list[Detection]], calibration: dict[str, float]
) -> list[list[Detection]]:
    calibrated: list[list[Detection]] = []
    for detections in all_detections:
        image_items = []
        for item in detections:
            if item.class_id != 1:
                image_items.append(item)
                continue
            x1, y1, x2, y2 = item.box
            center_x = (x1 + x2) / 2 + calibration["center_x_offset"]
            center_y = (y1 + y2) / 2 + calibration["center_y_offset"]
            width = (x2 - x1) * calibration["width_scale"]
            height = (y2 - y1) * calibration["height_scale"]
            image_items.append(
                Detection(
                    class_id=item.class_id,
                    confidence=item.confidence,
                    box=(
                        max(0.0, center_x - width / 2),
                        max(0.0, center_y - height / 2),
                        min(1.0, center_x + width / 2),
                        min(1.0, center_y + height / 2),
                    ),
                )
            )
        calibrated.append(image_items)
    return calibrated


def evaluate(
    detections: list[list[Detection]],
    targets: list[torch.Tensor],
    calibration: dict[str, float],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    calibrated = apply_calibration(detections, calibration)
    tp, fp, fn = count_matches(
        calibrated, targets, num_classes=2, iou_threshold=iou_threshold
    )
    precision = tp[1] / (tp[1] + fp[1]) if tp[1] + fp[1] else 0.0
    recall = tp[1] / (tp[1] + fn[1]) if tp[1] + fn[1] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp[1],
        "false_positive": fp[1],
        "false_negative": fn[1],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def collect_decoded(
    model: TinyQuantDetector,
    dataset: YoloDetectionDataset,
    config: Any,
    postprocess: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[list[Detection]], list[torch.Tensor]]:
    thresholds = tuple(
        float(postprocess["confidence_thresholds"][name]) for name in config.class_names
    )
    nms_iou = tuple(float(postprocess["nms_iou"][name]) for name in config.class_names)
    constraints_config = postprocess.get("box_constraints")
    constraints = None
    if constraints_config is not None:
        constraints = tuple(constraints_config.get(name) for name in config.class_names)
    cached = collect_predictions(model, dataset, device=device, batch_size=batch_size)
    detections: list[list[Detection]] = []
    targets: list[torch.Tensor] = []
    for predictions, batch_targets in cached:
        detections.extend(
            decode_predictions(
                predictions,
                confidence_threshold=thresholds,
                nms_iou=nms_iou,
                box_constraints=constraints,
                anchors=config.anchors,
                slots_per_class=config.slots_per_class,
            )
        )
        targets.extend(batch_targets)
    return detections, targets


def parameter_values(name: str) -> list[float]:
    if name in {"width_scale", "height_scale"}:
        return [round(0.80 + 0.025 * index, 3) for index in range(21)]
    return [round(-0.04 + 0.005 * index, 3) for index in range(17)]


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    selected = json.loads(args.postprocess.read_text(encoding="utf-8"))["selected"]

    datasets = {
        split: YoloDetectionDataset(
            args.data,
            split,
            input_size=(config.image_width, config.image_height),
            num_classes=config.num_classes,
        )
        for split in ("valid", "test")
    }
    decoded = {
        split: collect_decoded(
            model,
            dataset,
            config,
            selected,
            device=device,
            batch_size=args.batch_size,
        )
        for split, dataset in datasets.items()
    }

    calibration = {
        "width_scale": 1.0,
        "height_scale": 1.0,
        "center_x_offset": 0.0,
        "center_y_offset": 0.0,
    }
    baseline = {
        split: evaluate(detections, targets, calibration)
        for split, (detections, targets) in decoded.items()
    }
    rows: list[dict[str, float | int | str]] = []
    parameters = tuple(calibration)
    for iteration in range(3):
        changed = False
        for parameter in parameters:
            candidates = []
            for value in parameter_values(parameter):
                candidate = dict(calibration)
                candidate[parameter] = value
                result = evaluate(*decoded["valid"], candidate)
                row: dict[str, float | int | str] = {
                    "iteration": iteration + 1,
                    "parameter": parameter,
                    "candidate": value,
                    **candidate,
                    **result,
                }
                rows.append(row)
                candidates.append((candidate, result))
            best_candidate, _ = max(
                candidates,
                key=lambda item: (
                    float(item[1]["f1"]),
                    float(item[1]["recall"]),
                    float(item[1]["precision"]),
                    -abs(item[0][parameter] - (1.0 if "scale" in parameter else 0.0)),
                ),
            )
            if best_candidate[parameter] != calibration[parameter]:
                changed = True
            calibration = best_candidate
        if not changed:
            break

    calibrated = {
        split: evaluate(detections, targets, calibration)
        for split, (detections, targets) in decoded.items()
    }
    relaxed = {
        split: evaluate(detections, targets, calibration, iou_threshold=0.3)
        for split, (detections, targets) in decoded.items()
    }
    accepted = float(calibrated["valid"]["f1"]) > float(baseline["valid"]["f1"])
    payload = {
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT).as_posix()),
        "selected_on": "validation",
        "accepted": accepted,
        "calibration": calibration if accepted else {
            "width_scale": 1.0,
            "height_scale": 1.0,
            "center_x_offset": 0.0,
            "center_y_offset": 0.0,
        },
        "baseline_iou50": baseline,
        "calibrated_iou50": calibrated,
        "calibrated_iou30": relaxed,
        "note": "Test metrics are reported once after validation-only selection.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "coordinate_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "box_geometry_calibration.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
