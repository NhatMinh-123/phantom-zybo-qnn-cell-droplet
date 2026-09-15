from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from qnn.dataset import YoloDetectionDataset
from qnn.evaluate_qat import collect_predictions, evaluate_cached
from qnn.model import TinyQuantDetector, config_from_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate class-specific confidence, NMS, and ROI constraints"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "cell_droplet_roi384_grouped",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "qnn_cell_droplet_v2_w4a8_grouped"
        / "best.pt",
    )
    parser.add_argument(
        "--baseline-evaluation",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qnn_cell_droplet_v2_w4a8_grouped"
        / "evaluation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qnn_droplet_postprocess",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--spatial-margin",
        type=float,
        default=0.05,
        help="Margin around each class center-y range measured on the train split",
    )
    parser.add_argument(
        "--cell-precision-floor",
        type=float,
        default=0.79,
        help="Minimum validation precision before maximizing cell recall",
    )
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def center_y_constraint(
    dataset: YoloDetectionDataset,
    *,
    class_id: int,
    margin: float,
) -> dict[str, float]:
    center_y = [
        float(row[2].item())
        for _, targets in dataset
        for row in targets
        if int(row[0].item()) == class_id
    ]
    if not center_y:
        raise RuntimeError(f"No class {class_id} labels in the training split")
    return {
        "center_y_min": max(0.0, min(center_y) - margin),
        "center_y_max": min(1.0, max(center_y) + margin),
    }


def sweep_class(
    cached: list[tuple[torch.Tensor, list[torch.Tensor]]],
    *,
    config,
    class_id: int,
    confidence_values: list[float],
    nms_values: list[float],
    box_constraints: tuple[dict[str, float] | None, ...],
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    class_name = config.class_names[class_id]
    rows: list[dict[str, float | int | str]] = []
    best: dict[str, float | int | str] | None = None
    for confidence in confidence_values:
        for class_nms in nms_values:
            thresholds = [1.01] * config.num_classes
            thresholds[class_id] = confidence
            nms_iou = [0.45] * config.num_classes
            nms_iou[class_id] = class_nms
            result = evaluate_cached(
                cached,
                config=config,
                thresholds=tuple(thresholds),
                nms_iou=tuple(nms_iou),
                box_constraints=box_constraints,
            )["classes"][class_name]
            row: dict[str, float | int | str] = {
                "class": class_name,
                "confidence": confidence,
                "nms_iou": class_nms,
                "true_positive": int(result["true_positive"]),
                "false_positive": int(result["false_positive"]),
                "false_negative": int(result["false_negative"]),
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1"]),
            }
            rows.append(row)
            if best is None or (
                float(row["f1"]),
                float(row["recall"]),
                float(row["precision"]),
                -abs(float(row["nms_iou"]) - 0.45),
            ) > (
                float(best["f1"]),
                float(best["recall"]),
                float(best["precision"]),
                -abs(float(best["nms_iou"]) - 0.45),
            ):
                best = row
    assert best is not None
    return {key: value for key, value in best.items() if key != "class"}, rows


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    if config.num_classes != 2 or config.class_names[1] != "droplet":
        raise ValueError("Expected class order: cell, droplet")

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
    train_dataset = YoloDetectionDataset(
        args.data,
        "train",
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
    )
    constraints_by_class = tuple(
        center_y_constraint(
            train_dataset,
            class_id=class_id,
            margin=args.spatial_margin,
        )
        for class_id in range(config.num_classes)
    )
    valid_cache = collect_predictions(
        model, valid_dataset, device=device, batch_size=args.batch_size
    )
    test_cache = collect_predictions(
        model, test_dataset, device=device, batch_size=args.batch_size
    )

    baseline_payload = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
    baseline_thresholds = tuple(
        float(baseline_payload["validation"]["thresholds"][class_name])
        for class_name in config.class_names
    )
    confidence_values_by_class = (
        [round(value / 100, 2) for value in range(80, 100)],
        [round(value / 100, 2) for value in range(70, 100)],
    )
    nms_values_by_class = (
        [round(value / 100, 2) for value in range(20, 91, 5)],
        [round(value / 100, 2) for value in range(10, 51, 5)],
    )
    selected_rows = []
    rows: list[dict[str, float | int | str]] = []
    for class_id in range(config.num_classes):
        selected, class_rows = sweep_class(
            valid_cache,
            config=config,
            class_id=class_id,
            confidence_values=confidence_values_by_class[class_id],
            nms_values=nms_values_by_class[class_id],
            box_constraints=constraints_by_class,
        )
        selected_rows.append(selected)
        rows.extend(class_rows)

    deployment_rows = list(selected_rows)
    cell_candidates = [
        row
        for row in rows
        if row["class"] == "cell"
        and float(row["precision"]) >= args.cell_precision_floor
    ]
    if cell_candidates:
        deployment_rows[0] = max(
            cell_candidates,
            key=lambda row: (
                float(row["recall"]),
                float(row["f1"]),
                float(row["precision"]),
                -abs(float(row["nms_iou"]) - 0.45),
            ),
        )

    selected_thresholds = tuple(float(row["confidence"]) for row in deployment_rows)
    selected_nms = tuple(float(row["nms_iou"]) for row in deployment_rows)
    validation = evaluate_cached(
        valid_cache,
        config=config,
        thresholds=selected_thresholds,
        nms_iou=selected_nms,
        box_constraints=constraints_by_class,
    )
    test = evaluate_cached(
        test_cache,
        config=config,
        thresholds=selected_thresholds,
        nms_iou=selected_nms,
        box_constraints=constraints_by_class,
    )
    baseline_validation = evaluate_cached(
        valid_cache,
        config=config,
        thresholds=baseline_thresholds,
        nms_iou=(0.45,) * config.num_classes,
    )
    baseline_test = evaluate_cached(
        test_cache,
        config=config,
        thresholds=baseline_thresholds,
        nms_iou=(0.45,) * config.num_classes,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "droplet_postprocess_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "checkpoint": str(args.checkpoint),
        "selected_on": "validation",
        "selected": {
            "confidence_thresholds": dict(zip(config.class_names, selected_thresholds)),
            "nms_iou": dict(zip(config.class_names, selected_nms)),
            "box_constraints": dict(zip(config.class_names, constraints_by_class)),
            "validation_sweep_winners": dict(zip(config.class_names, selected_rows)),
            "deployment_winners": dict(zip(config.class_names, deployment_rows)),
            "selection_policy": {
                "cell": (
                    "maximize validation recall subject to precision >= "
                    f"{args.cell_precision_floor:.2f}"
                ),
                "droplet": "maximize validation F1, then recall and precision",
            },
        },
        "baseline": {
            "confidence_thresholds": {
                class_name: threshold
                for class_name, threshold in zip(config.class_names, baseline_thresholds)
            },
            "nms_iou": {class_name: 0.45 for class_name in config.class_names},
            "validation": baseline_validation,
            "test": baseline_test,
        },
        "optimized": {"validation": validation, "test": test},
    }
    (args.output / "postprocess_config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
