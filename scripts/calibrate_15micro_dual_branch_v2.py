#!/usr/bin/env python3
"""Calibrate radial cell detection and guarded fusion on valid/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.calibrate_15micro_dual_branch as base
from scripts.dual_branch_radial_v2 import RadialConfig, detect_radial_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/15micro_roi120_center_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/15micro/qnn_w4a6_96_roi120_v1/last.pt")
    parser.add_argument("--postprocess", type=Path, default=ROOT / "reports/15micro_qnn_roi120_v1/postprocess_balanced_epoch29.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/15micro_dual_branch_radial_v2")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def evaluate(records, config: RadialConfig) -> dict[str, object]:
    totals = {"qnn": (0, 0, 0), "classical": (0, 0, 0), "fusion": (0, 0, 0)}
    candidate_counts: list[int] = []
    for _, gray, qnn_cells, qnn_droplets, truth_cells in records:
        candidates = detect_radial_cells(gray, config, droplet_boxes=qnn_droplets)
        fused = base.fuse_cell_boxes(qnn_cells, candidates)
        candidate_counts.append(len(candidates))
        totals["qnn"] = base.add_counts(totals["qnn"], base.match_counts(qnn_cells, truth_cells))
        totals["classical"] = base.add_counts(
            totals["classical"], base.match_counts([item.box for item in candidates], truth_cells)
        )
        totals["fusion"] = base.add_counts(totals["fusion"], base.match_counts(fused, truth_cells))
    import numpy as np
    return {
        "config": asdict(config),
        "qnn": base.metrics(totals["qnn"]),
        "classical": base.metrics(totals["classical"]),
        "fusion": base.metrics(totals["fusion"]),
        "classical_candidates_per_image": {
            "mean": float(np.mean(candidate_counts)),
            "maximum": int(max(candidate_counts, default=0)),
        },
    }


def main() -> None:
    args = parse_args()
    device = base.resolve_device(args.device)
    runtime = base.load_runtime("15micro W4A6", args.checkpoint, args.postprocess, device)
    validation = base.collect_split("valid", args, runtime, device)
    sweep = []
    for response in range(4, 61, 4):
        for contrast in (12, 16, 20, 24, 28, 32, 40, 48):
            for distance in (4, 5, 6, 7):
                sweep.append(evaluate(validation, RadialConfig(response, contrast, distance)))
    selected = max(sweep, key=base.selection_key)
    selected_config = RadialConfig(**selected["config"])
    test = base.collect_split("test", args, runtime, device)
    report = {
        "selection_policy": "validation fusion F1 with precision >= 0.70 guard; test evaluated once",
        "validation_images": len(validation),
        "test_images": len(test),
        "selected_validation": selected,
        "held_out_test": evaluate(test, selected_config),
        "implementation_contract": {
            "input": "96x96 UINT8 grayscale",
            "response": "center*8 minus eight radius-3 samples, arithmetic shift by 3",
            "fusion_scope": "cell only; QNN remains authoritative for droplets",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
