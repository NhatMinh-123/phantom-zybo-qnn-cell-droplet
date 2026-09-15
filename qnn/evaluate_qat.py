from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qnn.dataset import YoloDetectionDataset, detection_collate
from qnn.detection import count_matches, decode_predictions
from qnn.model import DetectorConfig, TinyQuantDetector, config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate the tiny QAT detector")
    parser.add_argument("--data", type=Path, default=Path("dataset/cell_droplet_yolo_grouped"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/qnn_cell_droplet/best.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/qnn_cell_droplet"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-bits",
        type=int,
        default=0,
        help="Override checkpoint output quantization bit width",
    )
    return parser.parse_args()


def config_from_checkpoint(checkpoint: dict[str, object]) -> DetectorConfig:
    return config_from_dict(checkpoint["config"])


def collect_predictions(
    model: TinyQuantDetector,
    dataset: YoloDetectionDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )
    cached: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model(images.to(device)).cpu()
            cached.append((predictions, targets))
    return cached


def evaluate_cached(
    cached: list[tuple[torch.Tensor, list[torch.Tensor]]],
    *,
    config: DetectorConfig,
    thresholds: tuple[float, ...],
    nms_iou: float | tuple[float, ...] = 0.45,
    box_constraints: tuple[dict[str, float] | None, ...] | None = None,
    box_calibration: tuple[dict[str, float] | None, ...] | None = None,
) -> dict[str, object]:
    true_positive = [0] * config.num_classes
    false_positive = [0] * config.num_classes
    false_negative = [0] * config.num_classes
    for predictions, targets in cached:
        detections = decode_predictions(
            predictions,
            confidence_threshold=thresholds,
            nms_iou=nms_iou,
            box_constraints=box_constraints,
            box_calibration=box_calibration,
            anchors=config.anchors,
            slots_per_class=config.slots_per_class,
        )
        tp, fp, fn = count_matches(
            detections,
            targets,
            num_classes=config.num_classes,
        )
        true_positive = [left + right for left, right in zip(true_positive, tp)]
        false_positive = [left + right for left, right in zip(false_positive, fp)]
        false_negative = [left + right for left, right in zip(false_negative, fn)]

    classes: dict[str, dict[str, float | int]] = {}
    for class_id, class_name in enumerate(config.class_names):
        tp = true_positive[class_id]
        fp = false_positive[class_id]
        fn = false_negative[class_id]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        classes[class_name] = {
            "threshold": thresholds[class_id],
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    tp_sum = sum(true_positive)
    fp_sum = sum(false_positive)
    fn_sum = sum(false_negative)
    precision = tp_sum / max(tp_sum + fp_sum, 1)
    recall = tp_sum / max(tp_sum + fn_sum, 1)
    return {
        "thresholds": dict(zip(config.class_names, thresholds)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "classes": classes,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_checkpoint(checkpoint)
    if args.output_bits:
        config = replace(config, output_bits=args.output_bits)
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    valid_dataset = YoloDetectionDataset(
        args.data,
        "valid",
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
    )
    test_dataset = YoloDetectionDataset(
        args.data,
        "test",
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
    )
    valid_cache = collect_predictions(
        model, valid_dataset, device=device, batch_size=args.batch_size
    )
    test_cache = collect_predictions(model, test_dataset, device=device, batch_size=args.batch_size)

    threshold_values = [round(value / 100, 2) for value in range(5, 96, 5)]
    rows: list[dict[str, float | str]] = []
    selected: list[float] = []
    for class_id, class_name in enumerate(config.class_names):
        best_row: dict[str, float | str] | None = None
        for threshold in threshold_values:
            thresholds = [1.01] * config.num_classes
            thresholds[class_id] = threshold
            result = evaluate_cached(
                valid_cache,
                config=config,
                thresholds=tuple(thresholds),
            )["classes"][class_name]
            row = {
                "class": class_name,
                "threshold": threshold,
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1"]),
            }
            rows.append(row)
            if best_row is None or (row["f1"], row["recall"]) > (
                best_row["f1"],
                best_row["recall"],
            ):
                best_row = row
        assert best_row is not None
        selected.append(float(best_row["threshold"]))

    thresholds = tuple(selected)
    valid_result = evaluate_cached(valid_cache, config=config, thresholds=thresholds)
    test_result = evaluate_cached(test_cache, config=config, thresholds=thresholds)
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "config": config.to_dict(),
        "selected_on": "valid",
        "validation": valid_result,
        "test": test_result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "threshold_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
