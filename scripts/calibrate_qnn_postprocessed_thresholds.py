#!/usr/bin/env python3
"""Calibrate class thresholds after fixed NMS, ROI, and box calibration."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.dataset import YoloDetectionDataset
from qnn.evaluate_qat import collect_predictions, evaluate_cached
from qnn.model import TinyQuantDetector, config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold-min", type=float, default=0.60)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def class_tuple(
    selected: dict[str, Any],
    field: str,
    class_names: tuple[str, ...],
) -> tuple[Any, ...] | None:
    values = selected.get(field)
    if values is None:
        return None
    return tuple(values.get(name) for name in class_names)


def threshold_values(
    minimum: float,
    maximum: float,
    step: float,
) -> list[float]:
    if step <= 0.0 or minimum > maximum:
        raise ValueError("Invalid threshold range")
    count = int(round((maximum - minimum) / step))
    return [round(minimum + index * step, 6) for index in range(count + 1)]


def combined_metrics(
    class_rows: tuple[dict[str, Any], ...],
) -> dict[str, float | int]:
    true_positive = sum(int(row["true_positive"]) for row in class_rows)
    false_positive = sum(int(row["false_positive"]) for row in class_rows)
    false_negative = sum(int(row["false_negative"]) for row in class_rows)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    macro_f1 = sum(float(row["f1"]) for row in class_rows) / len(class_rows)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = config_from_dict(checkpoint["config"])
    class_names = tuple(config.class_names)
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    postprocess_payload = json.loads(
        args.postprocess.read_text(encoding="utf-8")
    )
    selected = postprocess_payload.get("selected", postprocess_payload)
    nms_iou = tuple(
        float(selected["nms_iou"][name])
        for name in class_names
    )
    box_constraints = class_tuple(selected, "box_constraints", class_names)
    box_calibration = class_tuple(selected, "box_calibration", class_names)
    values = threshold_values(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )

    valid_dataset = YoloDetectionDataset(
        args.data,
        "valid",
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
    )
    valid_cache = collect_predictions(
        model,
        valid_dataset,
        device=device,
        batch_size=args.batch_size,
    )

    rows_by_class: list[list[dict[str, Any]]] = []
    sweep_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        class_rows = []
        for threshold in values:
            thresholds = [1.01] * config.num_classes
            thresholds[class_id] = threshold
            result = evaluate_cached(
                valid_cache,
                config=config,
                thresholds=tuple(thresholds),
                nms_iou=nms_iou,
                box_constraints=box_constraints,
                box_calibration=box_calibration,
            )["classes"][class_name]
            row = {
                "class": class_name,
                "threshold": threshold,
                **result,
            }
            class_rows.append(row)
            sweep_rows.append(row)
        rows_by_class.append(class_rows)

    candidates = []
    for class_rows in itertools.product(*rows_by_class):
        metrics = combined_metrics(class_rows)
        thresholds = tuple(float(row["threshold"]) for row in class_rows)
        candidates.append((thresholds, metrics))
    selected_thresholds, validation_score = max(
        candidates,
        key=lambda item: (
            float(item[1]["f1"]),
            float(item[1]["macro_f1"]),
            float(item[1]["recall"]),
            float(item[1]["precision"]),
        ),
    )

    calibrated_payload = copy.deepcopy(postprocess_payload)
    calibrated_selected = calibrated_payload.get(
        "selected",
        calibrated_payload,
    )
    calibrated_selected["confidence_thresholds"] = dict(
        zip(class_names, selected_thresholds)
    )
    calibrated_selected["threshold_selection_policy"] = (
        "maximize validation micro-F1 after fixed NMS, ROI, and box calibration"
    )

    test_dataset = YoloDetectionDataset(
        args.data,
        "test",
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
    )
    test_cache = collect_predictions(
        model,
        test_dataset,
        device=device,
        batch_size=args.batch_size,
    )
    validation = evaluate_cached(
        valid_cache,
        config=config,
        thresholds=selected_thresholds,
        nms_iou=nms_iou,
        box_constraints=box_constraints,
        box_calibration=box_calibration,
    )
    test = evaluate_cached(
        test_cache,
        config=config,
        thresholds=selected_thresholds,
        nms_iou=nms_iou,
        box_constraints=box_constraints,
        box_calibration=box_calibration,
    )
    for result in (validation, test):
        result["macro_f1"] = sum(
            float(values["f1"])
            for values in result["classes"].values()
        ) / config.num_classes

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "postprocess_source": str(args.postprocess),
        "selected_on": "validation",
        "selection_metric": "micro_f1",
        "validation_selection_score": validation_score,
        "selected_thresholds": dict(zip(class_names, selected_thresholds)),
        "validation": validation,
        "test": test,
        "note": "Test was evaluated once after validation-only selection.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "class_threshold_sweep.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    (args.output / "postprocess_config.json").write_text(
        json.dumps(calibrated_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "threshold_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
