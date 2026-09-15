#!/usr/bin/env python3
"""Separate localization-only misses from real detector failures."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.dataset import YoloDetectionDataset
from qnn.detection import Detection, box_iou, count_matches, decode_predictions
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
DEFAULT_DATA = ROOT / "dataset" / "cell_droplet_roi384_grouped"
DEFAULT_OUTPUT = ROOT / "reports" / "qnn_detection_failure_modes_w4a6_square192"
CLASS_COLORS = ((225, 45, 70), (20, 120, 225))
TILE_PATTERN = re.compile(
    r"^(?P<source>.+_src\d+)_tile(?P<tile>\d+)_x(?P<x>\d+)_y(?P<y>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--postprocess", type=Path, default=DEFAULT_POSTPROCESS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--edge-margin", type=float, default=0.03)
    return parser.parse_args()


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def target_as_detections(target: torch.Tensor) -> list[Detection]:
    detections: list[Detection] = []
    for row in target:
        class_id = int(row[0].item())
        center_x, center_y, width, height = (float(value) for value in row[1:5])
        detections.append(
            Detection(
                class_id=class_id,
                confidence=1.0,
                box=(
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
            )
        )
    return detections


def detections_as_target(detections: list[Detection]) -> torch.Tensor:
    rows = []
    for item in detections:
        x1, y1, x2, y2 = item.box
        rows.append(
            [
                float(item.class_id),
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
            ]
        )
    if not rows:
        return torch.empty((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def match_counts(
    detections: list[Detection], target: torch.Tensor, iou_threshold: float
) -> tuple[list[int], list[int], list[int]]:
    return count_matches(
        [detections], [target], num_classes=2, iou_threshold=iou_threshold
    )


def category_for(
    counts_50: tuple[list[int], list[int], list[int]],
    counts_30: tuple[list[int], list[int], list[int]],
    class_id: int,
) -> str:
    _, fp_50, fn_50 = counts_50
    _, fp_30, fn_30 = counts_30
    if fp_50[class_id] == 0 and fn_50[class_id] == 0:
        return "clean_iou50"
    if fp_30[class_id] == 0 and fn_30[class_id] == 0:
        return "localization_only"
    return "true_detection_error"


def target_boxes(target: torch.Tensor, class_id: int) -> torch.Tensor:
    rows = target[target[:, 0] == class_id]
    if not len(rows):
        return torch.empty((0, 4), dtype=torch.float32)
    centers = rows[:, 1:3]
    sizes = rows[:, 3:5]
    return torch.cat((centers - sizes / 2, centers + sizes / 2), dim=1)


def best_target_ious(
    detections: list[Detection], target: torch.Tensor, class_id: int
) -> list[float]:
    boxes = target_boxes(target, class_id)
    predicted = [item for item in detections if item.class_id == class_id]
    if not len(boxes):
        return []
    if not predicted:
        return [0.0] * len(boxes)
    predicted_boxes = torch.tensor([item.box for item in predicted], dtype=torch.float32)
    overlaps = box_iou(boxes, predicted_boxes)
    return [float(value) for value in overlaps.max(dim=1).values]


def edge_flags(target: torch.Tensor, class_id: int, margin: float) -> list[bool]:
    boxes = target_boxes(target, class_id)
    return [
        bool(
            box[0] <= margin
            or box[1] <= margin
            or box[2] >= 1.0 - margin
            or box[3] >= 1.0 - margin
        )
        for box in boxes
    ]


def metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_metrics(
    all_detections: list[list[Detection]],
    all_targets: list[torch.Tensor],
    iou_threshold: float,
) -> dict[str, Any]:
    tp, fp, fn = count_matches(
        all_detections,
        all_targets,
        num_classes=2,
        iou_threshold=iou_threshold,
    )
    return {
        "iou_threshold": iou_threshold,
        "cell": metrics(tp[0], fp[0], fn[0]),
        "droplet": metrics(tp[1], fp[1], fn[1]),
        "overall": metrics(sum(tp), sum(fp), sum(fn)),
    }


def draw_panel(
    image_path: Path,
    detections: list[Detection],
    title: str,
    class_names: tuple[str, ...],
    show_confidence: bool,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for item in detections:
        x1, y1, x2, y2 = item.box
        box = (
            round(x1 * image.width),
            round(y1 * image.height),
            round(x2 * image.width),
            round(y2 * image.height),
        )
        color = CLASS_COLORS[item.class_id]
        draw.rectangle(box, outline=color, width=3)
        label = class_names[item.class_id]
        if show_confidence:
            label += f" {item.confidence:.2f}"
        draw.text((box[0] + 2, max(0, box[1] - 11)), label, fill=color, font=font)
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((7, 8), title, fill="black", font=font)
    return canvas


def make_case_sheet(
    output: Path,
    indices: list[int],
    dataset: YoloDetectionDataset,
    all_targets: list[torch.Tensor],
    all_detections: list[list[Detection]],
    rows: list[dict[str, Any]],
    class_names: tuple[str, ...],
) -> None:
    if not indices:
        return
    sheet_rows = []
    for index in indices:
        row = rows[index]
        panels = [
            draw_panel(
                dataset.images[index],
                target_as_detections(all_targets[index]),
                "Ground truth",
                class_names,
                False,
            ),
            draw_panel(
                dataset.images[index],
                all_detections[index],
                (
                    f"Prediction: {row['droplet_category']}; "
                    f"FP/FN@.5={row['droplet_fp_iou50']}/{row['droplet_fn_iou50']}"
                ),
                class_names,
                True,
            ),
        ]
        combined = Image.new(
            "RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white"
        )
        offset = 0
        for panel in panels:
            combined.paste(panel, (offset, 0))
            offset += panel.width
        sheet_rows.append(combined)
    sheet = Image.new(
        "RGB",
        (max(row.width for row in sheet_rows), sum(row.height for row in sheet_rows)),
        "white",
    )
    offset = 0
    for row in sheet_rows:
        sheet.paste(row, (0, offset))
        offset += row.height
    sheet.save(output, quality=94)


def nms_global(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for class_id in range(2):
        candidates = sorted(
            (item for item in detections if item.class_id == class_id),
            key=lambda item: item.confidence,
            reverse=True,
        )
        while candidates:
            current = candidates.pop(0)
            kept.append(current)
            if not candidates:
                break
            current_box = torch.tensor(current.box, dtype=torch.float32).view(1, 4)
            other_boxes = torch.tensor(
                [item.box for item in candidates], dtype=torch.float32
            )
            overlaps = box_iou(current_box, other_boxes)[0]
            candidates = [
                item
                for item, overlap in zip(candidates, overlaps)
                if float(overlap) <= iou_threshold
            ]
    return sorted(kept, key=lambda item: item.confidence, reverse=True)


def grouped_roi_predictions(
    dataset: YoloDetectionDataset,
    all_targets: list[torch.Tensor],
    all_detections: list[list[Detection]],
) -> tuple[list[list[Detection]], list[torch.Tensor], list[dict[str, Any]]]:
    groups: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for index, image_path in enumerate(dataset.images):
        match = TILE_PATTERN.match(image_path.stem)
        if match is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        groups[match.group("source")].append(
            (index, int(match.group("x")), int(match.group("y")), width, height)
        )

    merged_predictions: list[list[Detection]] = []
    merged_targets: list[torch.Tensor] = []
    group_rows: list[dict[str, Any]] = []
    for source, tiles in sorted(groups.items()):
        tiles.sort(key=lambda item: item[1])
        centers = [x + width / 2 for _, x, _, width, _ in tiles]
        boundaries = [
            (centers[index] + centers[index + 1]) / 2
            for index in range(len(centers) - 1)
        ]
        roi_x1 = min(x for _, x, _, _, _ in tiles)
        roi_y1 = min(y for _, _, y, _, _ in tiles)
        roi_x2 = max(x + width for _, x, _, width, _ in tiles)
        roi_y2 = max(y + height for _, _, y, _, height in tiles)
        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1
        global_predictions: list[Detection] = []
        global_ground_truth: list[Detection] = []

        for tile_index, (index, x, y, width, height) in enumerate(tiles):
            owner_left = boundaries[tile_index - 1] if tile_index else float("-inf")
            owner_right = boundaries[tile_index] if tile_index < len(boundaries) else float("inf")
            for source_items, destination in (
                (all_detections[index], global_predictions),
                (target_as_detections(all_targets[index]), global_ground_truth),
            ):
                for item in source_items:
                    local_x1, local_y1, local_x2, local_y2 = item.box
                    absolute_box = (
                        x + local_x1 * width,
                        y + local_y1 * height,
                        x + local_x2 * width,
                        y + local_y2 * height,
                    )
                    center_x = (absolute_box[0] + absolute_box[2]) / 2
                    if not owner_left <= center_x < owner_right:
                        continue
                    destination.append(
                        Detection(
                            class_id=item.class_id,
                            confidence=item.confidence,
                            box=(
                                (absolute_box[0] - roi_x1) / roi_width,
                                (absolute_box[1] - roi_y1) / roi_height,
                                (absolute_box[2] - roi_x1) / roi_width,
                                (absolute_box[3] - roi_y1) / roi_height,
                            ),
                        )
                    )

        global_predictions = nms_global(global_predictions, iou_threshold=0.5)
        global_ground_truth = nms_global(global_ground_truth, iou_threshold=0.5)
        target = detections_as_target(global_ground_truth)
        counts_50 = match_counts(global_predictions, target, 0.5)
        counts_30 = match_counts(global_predictions, target, 0.3)
        merged_predictions.append(global_predictions)
        merged_targets.append(target)
        group_rows.append(
            {
                "source": source,
                "tiles": len(tiles),
                "gt_cell": sum(item.class_id == 0 for item in global_ground_truth),
                "pred_cell": sum(item.class_id == 0 for item in global_predictions),
                "cell_fp_iou50": counts_50[1][0],
                "cell_fn_iou50": counts_50[2][0],
                "gt_droplet": sum(item.class_id == 1 for item in global_ground_truth),
                "pred_droplet": sum(item.class_id == 1 for item in global_predictions),
                "droplet_fp_iou50": counts_50[1][1],
                "droplet_fn_iou50": counts_50[2][1],
                "droplet_fp_iou30": counts_30[1][1],
                "droplet_fn_iou30": counts_30[2][1],
            }
        )
    return merged_predictions, merged_targets, group_rows


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    checkpoint_path = args.checkpoint.resolve()
    postprocess_path = args.postprocess.resolve()
    data_root = args.data.resolve()
    output_dir = args.output.resolve()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = YoloDetectionDataset(
        data_root,
        args.split,
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
    )
    postprocess = json.loads(postprocess_path.read_text(encoding="utf-8"))["selected"]
    thresholds = tuple(
        float(postprocess["confidence_thresholds"][name]) for name in config.class_names
    )
    nms_iou = tuple(float(postprocess["nms_iou"][name]) for name in config.class_names)
    constraints_config = postprocess.get("box_constraints")
    constraints = None
    if constraints_config is not None:
        constraints = tuple(constraints_config.get(name) for name in config.class_names)
    calibration_config = postprocess.get("box_calibration")
    box_calibration = None
    if calibration_config is not None:
        box_calibration = tuple(
            calibration_config.get(name) for name in config.class_names
        )

    cached = collect_predictions(model, dataset, device=device, batch_size=args.batch_size)
    all_detections: list[list[Detection]] = []
    all_targets: list[torch.Tensor] = []
    for predictions, targets in cached:
        all_detections.extend(
            decode_predictions(
                predictions,
                confidence_threshold=thresholds,
                nms_iou=nms_iou,
                box_constraints=constraints,
                box_calibration=box_calibration,
                anchors=config.anchors,
                slots_per_class=config.slots_per_class,
            )
        )
        all_targets.extend(targets)

    rows: list[dict[str, Any]] = []
    missed_droplets = {"iou50": 0, "iou30": 0, "edge_iou50": 0, "edge_iou30": 0}
    for image_path, detections, target in zip(dataset.images, all_detections, all_targets):
        counts_50 = match_counts(detections, target, 0.5)
        counts_30 = match_counts(detections, target, 0.3)
        droplet_ious = best_target_ious(detections, target, 1)
        droplet_edges = edge_flags(target, 1, args.edge_margin)
        for best_iou, at_edge in zip(droplet_ious, droplet_edges):
            if best_iou < 0.5:
                missed_droplets["iou50"] += 1
                missed_droplets["edge_iou50"] += int(at_edge)
            if best_iou < 0.3:
                missed_droplets["iou30"] += 1
                missed_droplets["edge_iou30"] += int(at_edge)
        gt_counts = [int((target[:, 0] == class_id).sum()) for class_id in range(2)]
        pred_counts = [sum(item.class_id == class_id for item in detections) for class_id in range(2)]
        rows.append(
            {
                "image": image_path.name,
                "cell_category": category_for(counts_50, counts_30, 0),
                "droplet_category": category_for(counts_50, counts_30, 1),
                "gt_cell": gt_counts[0],
                "pred_cell": pred_counts[0],
                "cell_fp_iou50": counts_50[1][0],
                "cell_fn_iou50": counts_50[2][0],
                "cell_fp_iou30": counts_30[1][0],
                "cell_fn_iou30": counts_30[2][0],
                "gt_droplet": gt_counts[1],
                "pred_droplet": pred_counts[1],
                "droplet_fp_iou50": counts_50[1][1],
                "droplet_fn_iou50": counts_50[2][1],
                "droplet_fp_iou30": counts_30[1][1],
                "droplet_fn_iou30": counts_30[2][1],
                "droplet_gt_at_edge": sum(droplet_edges),
                "droplet_min_best_iou": min(droplet_ious) if droplet_ious else None,
                "droplet_mean_best_iou": (
                    sum(droplet_ious) / len(droplet_ious) if droplet_ious else None
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    categories = ("clean_iou50", "localization_only", "true_detection_error")
    for category in categories:
        indices = [
            index for index, row in enumerate(rows) if row["droplet_category"] == category
        ]
        indices.sort(
            key=lambda index: (
                int(rows[index]["droplet_fp_iou50"])
                + int(rows[index]["droplet_fn_iou50"]),
                int(rows[index]["gt_droplet"]),
            ),
            reverse=True,
        )
        make_case_sheet(
            output_dir / f"droplet_{category}_cases.jpg",
            indices[: args.examples],
            dataset,
            all_targets,
            all_detections,
            rows,
            config.class_names,
        )

    merged_predictions, merged_targets, grouped_rows = grouped_roi_predictions(
        dataset, all_targets, all_detections
    )
    with (output_dir / "grouped_roi.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped_rows[0]))
        writer.writeheader()
        writer.writerows(grouped_rows)

    summary = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT).as_posix()),
        "postprocess": str(postprocess_path.relative_to(ROOT).as_posix()),
        "data": str(data_root.relative_to(ROOT).as_posix()),
        "split": args.split,
        "input_size": [config.image_width, config.image_height],
        "images": len(dataset),
        "tile_overlap_policy": (
            "Assign each global box center to the nearest tile center, then apply "
            "class-wise global NMS at IoU 0.5."
        ),
        "per_tile": {
            "categories": {
                "cell": dict(Counter(row["cell_category"] for row in rows)),
                "droplet": dict(Counter(row["droplet_category"] for row in rows)),
            },
            "iou50": aggregate_metrics(all_detections, all_targets, 0.5),
            "iou30": aggregate_metrics(all_detections, all_targets, 0.3),
            "droplet_miss_diagnostic": missed_droplets,
        },
        "merged_roi": {
            "groups": len(grouped_rows),
            "iou50": aggregate_metrics(merged_predictions, merged_targets, 0.5),
            "iou30": aggregate_metrics(merged_predictions, merged_targets, 0.3),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    droplet_categories = summary["per_tile"]["categories"]["droplet"]
    merged_droplet = summary["merged_roi"]["iou50"]["droplet"]
    readme = f"""# W4A6 square-192 failure-mode audit

This report separates strict localization failures from real detection failures.
Thresholds and NMS were selected on validation data; the test split is diagnosis only.

## Per-tile cases

- Clean at IoU 0.50: {droplet_categories.get('clean_iou50', 0)}/{len(dataset)} images
- Localization-only (passes at IoU 0.30): {droplet_categories.get('localization_only', 0)}/{len(dataset)} images
- True detection error at IoU 0.30: {droplet_categories.get('true_detection_error', 0)}/{len(dataset)} images
- Missed droplet GT at IoU 0.50: {missed_droplets['iou50']} (edge: {missed_droplets['edge_iou50']})
- Missed droplet GT at IoU 0.30: {missed_droplets['iou30']} (edge: {missed_droplets['edge_iou30']})

## Five-tile merged ROI

- Groups: {len(grouped_rows)}
- Droplet precision: {100 * float(merged_droplet['precision']):.2f}%
- Droplet recall: {100 * float(merged_droplet['recall']):.2f}%
- Droplet F1: {100 * float(merged_droplet['f1']):.2f}%

Use `per_image.csv` to inspect each tile and `grouped_roi.csv` for the realtime ROI view.
The generated contact sheets show ground truth on the left and predictions on the right.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {output_dir}")


if __name__ == "__main__":
    main()
