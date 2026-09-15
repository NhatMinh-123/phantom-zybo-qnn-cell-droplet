#!/usr/bin/env python3
"""Calibrate guarded fusion using the exact 24x24 grid center available in RTL."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

import scripts.calibrate_15micro_dual_branch as base
import scripts.calibrate_15micro_dual_branch_v3 as v3
from scripts.compare_qnn_video_roi import load_runtime


@dataclass(frozen=True)
class GridGuardConfig:
    low_confidence: float
    high_confidence: float
    response_threshold: int
    support_radius: int


def grid_support(response: np.ndarray, box, radius: int) -> int:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    grid_x = min(23, max(0, int(cx * 24.0)))
    grid_y = min(23, max(0, int(cy * 24.0)))
    x = grid_x * 4 + 2
    y = grid_y * 4 + 2
    x1 = max(0, x - radius)
    x2 = min(96, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(96, y + radius + 1)
    return int(np.max(response[y1:y2, x1:x2]))


def evaluate(records, config: GridGuardConfig):
    baseline_counts = (0, 0, 0)
    guarded_counts = (0, 0, 0)
    accepted_low = 0
    rejected_low = 0
    for record in records:
        _, response, _, qnn_cells, truth = record
        baseline = [
            item.box for item in qnn_cells
            if item.confidence >= config.high_confidence
        ]
        guarded = list(baseline)
        for item in qnn_cells:
            if not (config.low_confidence <= item.confidence < config.high_confidence):
                continue
            if grid_support(response, item.box, config.support_radius) >= config.response_threshold:
                guarded.append(item.box)
                accepted_low += 1
            else:
                rejected_low += 1
        baseline_counts = base.add_counts(
            baseline_counts, base.match_counts(baseline, truth)
        )
        guarded_counts = base.add_counts(
            guarded_counts, base.match_counts(guarded, truth)
        )
    return {
        "config": asdict(config),
        "baseline": base.metrics(baseline_counts),
        "guarded": base.metrics(guarded_counts),
        "low_confidence_proposals": {
            "accepted": accepted_low,
            "rejected": rejected_low,
        },
    }


def selection_key(result):
    guarded = result["guarded"]
    baseline = result["baseline"]
    precision_floor = max(0.75, float(baseline["precision"]) - 0.08)
    precision = float(guarded["precision"])
    f1 = float(guarded["f1"])
    return f1 * min(1.0, precision / precision_floor), f1, guarded["recall"], precision


def main() -> None:
    args = v3.parse_args()
    args.output = base.ROOT / "reports/15micro_dual_branch_guarded_v4_grid"
    device = base.resolve_device(args.device)
    runtime = load_runtime("15micro W4A6", args.checkpoint, args.postprocess, device)
    high_confidence = float(runtime.thresholds[0])
    runtime.thresholds = (0.30, runtime.thresholds[1])
    validation = v3.collect_split("valid", args, runtime, device)
    sweep = []
    for low_confidence in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.88, 0.92):
        for response_threshold in (20, 24, 28, 32, 36, 40, 44, 48):
            for support_radius in (0, 1, 2, 3, 4):
                sweep.append(
                    evaluate(
                        validation,
                        GridGuardConfig(
                            low_confidence,
                            high_confidence,
                            response_threshold,
                            support_radius,
                        ),
                    )
                )
    selected = max(sweep, key=selection_key)
    selected_config = GridGuardConfig(**selected["config"])
    test = v3.collect_split("test", args, runtime, device)
    report = {
        "selection_policy": "validation F1 with precision guard; grid-center RTL contract",
        "validation_images": len(validation),
        "test_images": len(test),
        "selected_validation": selected,
        "held_out_test": evaluate(test, selected_config),
        "rtl_contract": {
            "grid": "24x24, pixel center=(grid*4)+2",
            "response": "8*center minus eight radius-3 samples",
            "contrast_threshold": "omitted because response>=threshold implies local contrast>=threshold",
            "cell_low_object_code": v3.object_code(selected_config.low_confidence, 0.030286800116300583),
            "cell_high_object_code": v3.object_code(selected_config.high_confidence, 0.030286800116300583),
            "droplet_path": "unchanged",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
