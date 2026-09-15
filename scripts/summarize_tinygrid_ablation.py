#!/usr/bin/env python3
"""Summarize TinyGridNet FP32 feature and augmentation experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in sorted(args.models.resolve().glob("*/metrics_fp32.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        validation = metrics["validation"]
        test = metrics["test"]
        row: dict[str, object] = {
            "run": path.parent.name,
            "feature_mode": metrics["feature_mode"],
            "best_epoch": metrics["best_epoch"],
            "parameters": metrics["parameters"]["total"],
            "validation_macro_f1": validation["macro_f1"],
            "test_macro_f1": test["macro_f1"],
        }
        for split_name, report in (("valid", validation), ("test", test)):
            for class_name in ("cell", "droplet"):
                values = report["per_class"][class_name]
                for metric_name in ("precision", "recall", "f1"):
                    row[f"{split_name}_{class_name}_{metric_name}"] = values[metric_name]
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics_fp32.json found below {args.models}")
    rows.sort(key=lambda row: float(row["validation_macro_f1"]), reverse=True)
    with (output / "fp32_ablation.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = [str(row["run"]) for row in rows]
    valid_values = [float(row["validation_macro_f1"]) for row in rows]
    test_values = [float(row["test_macro_f1"]) for row in rows]
    positions = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar([value - 0.19 for value in positions], valid_values, width=0.38, label="Validation")
    axis.bar([value + 0.19 for value in positions], test_values, width=0.38, label="Test")
    axis.set_xticks(positions, names, rotation=24, ha="right")
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0.70, 0.93)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for index, value in enumerate(valid_values):
        axis.text(index - 0.19, value + 0.004, f"{value:.3f}", ha="center", fontsize=8)
    for index, value in enumerate(test_values):
        axis.text(index + 0.19, value + 0.004, f"{value:.3f}", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "fp32_ablation.png", dpi=180)
    plt.close(figure)

    summary = {
        "selection_rule": "Highest validation macro F1; test metrics are reporting only.",
        "selected_run": rows[0]["run"],
        "selected_checkpoint": str(
            (args.models.resolve() / str(rows[0]["run"]) / "best_fp32.pt").resolve()
        ),
        "runs": rows,
    }
    (output / "fp32_ablation.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
