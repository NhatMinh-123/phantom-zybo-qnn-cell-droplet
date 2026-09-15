#!/usr/bin/env python3
"""Evaluate a QNN checkpoint with a fixed validation-selected postprocess config."""

from __future__ import annotations

import argparse
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = config_from_dict(checkpoint["config"])
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    payload = json.loads(args.postprocess.read_text(encoding="utf-8"))
    selected = payload.get("selected", payload)
    thresholds = tuple(
        float(selected["confidence_thresholds"][name])
        for name in config.class_names
    )
    nms_iou = tuple(
        float(selected["nms_iou"][name])
        for name in config.class_names
    )
    box_constraints = class_tuple(
        selected,
        "box_constraints",
        config.class_names,
    )
    box_calibration = class_tuple(
        selected,
        "box_calibration",
        config.class_names,
    )

    results: dict[str, Any] = {}
    for split in ("valid", "test"):
        dataset = YoloDetectionDataset(
            args.data,
            split,
            input_size=(config.image_width, config.image_height),
            num_classes=config.num_classes,
        )
        cached = collect_predictions(
            model,
            dataset,
            device=device,
            batch_size=args.batch_size,
        )
        result = evaluate_cached(
            cached,
            config=config,
            thresholds=thresholds,
            nms_iou=nms_iou,
            box_constraints=box_constraints,
            box_calibration=box_calibration,
        )
        result["macro_f1"] = sum(
            float(values["f1"])
            for values in result["classes"].values()
        ) / config.num_classes
        results[split] = result

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "postprocess": str(args.postprocess),
        "selected_on": payload.get("selected_on", "validation"),
        "config": config.to_dict(),
        "validation": results["valid"],
        "test": results["test"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "postprocessed_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
