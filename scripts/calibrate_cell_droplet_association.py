#!/usr/bin/env python3
"""Calibrate a zero-cost cell-inside-droplet post-processing prior."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.dataset import YoloDetectionDataset
from qnn.detection import Detection, count_matches, decode_predictions
from qnn.evaluate_qat import collect_predictions
from qnn.fpga_io import (
    decoder_box_calibration,
    decoder_box_constraints,
    decoder_nms_iou,
    load_manifest,
)
from qnn.model import TinyQuantDetector, config_from_dict


DEFAULT_DATA = ROOT / "dataset" / "cell_droplet_roi384_grouped"
DEFAULT_CHECKPOINT = (
    ROOT / "models" / "qnn_cell_droplet_v2_w4a6_square192_grouped" / "best.pt"
)
DEFAULT_MANIFEST = (
    ROOT
    / "final_results"
    / "arty_s7_25_qnn_detection"
    / "07_50fps_optimized"
    / "fpga_manifest_60fps.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "qnn_cell_droplet_association"


@dataclass(frozen=True)
class Candidate:
    expansion: float
    keep_orphans: bool
    orphan_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def point_in_expanded_box(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
    expansion: float,
) -> bool:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    half_width = (x2 - x1) * expansion / 2
    half_height = (y2 - y1) * expansion / 2
    return (
        center_x - half_width <= point[0] <= center_x + half_width
        and center_y - half_height <= point[1] <= center_y + half_height
    )


def apply_candidate(
    detections: list[Detection],
    candidate: Candidate | None,
) -> list[Detection]:
    if candidate is None:
        return detections
    droplets = [item for item in detections if item.class_id == 1]
    filtered: list[Detection] = []
    for item in detections:
        if item.class_id != 0:
            filtered.append(item)
            continue
        x1, y1, x2, y2 = item.box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        associated = any(
            point_in_expanded_box(center, droplet.box, candidate.expansion)
            for droplet in droplets
        )
        if associated or (
            candidate.keep_orphans
            and item.confidence >= candidate.orphan_threshold
        ):
            filtered.append(item)
    return filtered


def metric_block(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    detections: list[list[Detection]],
    targets: list[torch.Tensor],
    candidate: Candidate | None,
) -> dict[str, Any]:
    filtered = [apply_candidate(items, candidate) for items in detections]
    tp, fp, fn = count_matches(
        filtered,
        targets,
        num_classes=2,
        iou_threshold=0.5,
    )
    return {
        "candidate": asdict(candidate) if candidate is not None else None,
        "cell": metric_block(tp[0], fp[0], fn[0]),
        "droplet": metric_block(tp[1], fp[1], fn[1]),
        "overall": metric_block(sum(tp), sum(fp), sum(fn)),
    }


def decode_split(
    model: TinyQuantDetector,
    dataset: YoloDetectionDataset,
    *,
    manifest: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[list[list[Detection]], list[torch.Tensor]]:
    decoder = manifest["postprocessing"]["decoder"]
    cached = collect_predictions(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
    )
    detections: list[list[Detection]] = []
    targets: list[torch.Tensor] = []
    for predictions, batch_targets in cached:
        detections.extend(
            decode_predictions(
                predictions,
                confidence_threshold=tuple(
                    float(value) for value in decoder["confidence_thresholds"]
                ),
                nms_iou=decoder_nms_iou(decoder),
                box_constraints=decoder_box_constraints(decoder),
                box_calibration=decoder_box_calibration(decoder),
                pre_nms_topk=int(decoder["pre_nms_topk"]),
                max_detections=int(decoder["max_detections"]),
                anchors=tuple(
                    tuple(float(value) for value in pair)
                    for pair in decoder["anchors"]
                ),
                slots_per_class=tuple(
                    int(value) for value in decoder["slots_per_class"]
                ),
            )
        )
        targets.extend(batch_targets)
    return detections, targets


def ground_truth_association(
    targets: list[torch.Tensor],
    expansion: float,
) -> dict[str, float | int]:
    cells = 0
    associated = 0
    for target in targets:
        droplets = []
        for row in target[target[:, 0] == 1]:
            center_x, center_y, width, height = (
                float(value) for value in row[1:5]
            )
            droplets.append(
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                )
            )
        for row in target[target[:, 0] == 0]:
            cells += 1
            point = (float(row[1]), float(row[2]))
            associated += int(
                any(
                    point_in_expanded_box(point, droplet, expansion)
                    for droplet in droplets
                )
            )
    return {
        "cells": cells,
        "associated": associated,
        "fraction": associated / max(cells, 1),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    manifest = load_manifest(args.manifest)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = config_from_dict(checkpoint["config"])
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    split_data: dict[
        str,
        tuple[list[list[Detection]], list[torch.Tensor]],
    ] = {}
    ground_truth: dict[str, dict[str, dict[str, float | int]]] = {}
    for split in ("valid", "test"):
        dataset = YoloDetectionDataset(
            args.data,
            split,
            input_size=(config.image_width, config.image_height),
            num_classes=config.num_classes,
        )
        split_data[split] = decode_split(
            model,
            dataset,
            manifest=manifest,
            device=device,
            batch_size=args.batch_size,
        )
        ground_truth[split] = {
            f"{expansion:.2f}": ground_truth_association(
                split_data[split][1],
                expansion,
            )
            for expansion in (1.0, 1.15, 1.3, 1.5, 2.0)
        }

    candidates = [
        Candidate(expansion, keep_orphans, orphan_threshold)
        for expansion in (1.0, 1.15, 1.3, 1.5, 2.0)
        for keep_orphans, orphan_threshold in (
            (False, 1.01),
            (True, 0.90),
            (True, 0.95),
        )
    ]
    validation_rows = [
        evaluate(*split_data["valid"], candidate)
        for candidate in candidates
    ]
    baseline_valid = evaluate(*split_data["valid"], None)
    selected = max(
        validation_rows,
        key=lambda row: (
            float(row["overall"]["f1"]),
            float(row["cell"]["f1"]),
            float(row["overall"]["precision"]),
        ),
    )
    selected_candidate = Candidate(**selected["candidate"])
    baseline_test = evaluate(*split_data["test"], None)
    selected_test = evaluate(*split_data["test"], selected_candidate)

    valid_f1_gain = float(selected["overall"]["f1"]) - float(
        baseline_valid["overall"]["f1"]
    )
    valid_cell_f1_gain = float(selected["cell"]["f1"]) - float(
        baseline_valid["cell"]["f1"]
    )
    accepted = valid_f1_gain >= 0.005 and valid_cell_f1_gain >= 0.0
    report = {
        "schema_version": 1,
        "selection_policy": (
            "Select only on validation overall F1; accept only when overall F1 "
            "gains at least 0.5 percentage points without reducing cell F1. "
            "Report test once after selection."
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "ground_truth_cell_center_inside_droplet": ground_truth,
        "validation": {
            "baseline": baseline_valid,
            "selected": selected,
        },
        "test": {
            "baseline": baseline_test,
            "selected": selected_test,
        },
        "validation_gain": {
            "overall_f1": valid_f1_gain,
            "cell_f1": valid_cell_f1_gain,
        },
        "accepted": accepted,
        "decision": (
            "Use the association filter in runtime."
            if accepted
            else "Reject: validation improvement is too small or cell F1 decreases."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "validation_grid.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "expansion",
                "keep_orphans",
                "orphan_threshold",
                "overall_precision",
                "overall_recall",
                "overall_f1",
                "cell_precision",
                "cell_recall",
                "cell_f1",
            )
        )
        for row in validation_rows:
            writer.writerow(
                (
                    row["candidate"]["expansion"],
                    row["candidate"]["keep_orphans"],
                    row["candidate"]["orphan_threshold"],
                    row["overall"]["precision"],
                    row["overall"]["recall"],
                    row["overall"]["f1"],
                    row["cell"]["precision"],
                    row["cell"]["recall"],
                    row["cell"]["f1"],
                )
            )
    print(json.dumps(report, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
