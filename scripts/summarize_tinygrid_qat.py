#!/usr/bin/env python3
"""Compare FP32 and QAT TinyGridNet checkpoints."""

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
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--qat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_row(name: str, metrics: dict[str, object], kind: str) -> dict[str, object]:
    validation = metrics["validation"]
    test = metrics["test"]
    quantization = metrics.get("quantization", {})
    return {
        "run": name,
        "kind": kind,
        "weight_bits": quantization.get("weight_bits", 32),
        "activation_bits": quantization.get("activation_bits", 32),
        "validation_macro_f1": validation["macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "test_cell_precision": test["per_class"]["cell"]["precision"],
        "test_cell_recall": test["per_class"]["cell"]["recall"],
        "test_cell_f1": test["per_class"]["cell"]["f1"],
        "test_droplet_precision": test["per_class"]["droplet"]["precision"],
        "test_droplet_recall": test["per_class"]["droplet"]["recall"],
        "test_droplet_f1": test["per_class"]["droplet"]["f1"],
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fp32_metrics = json.loads(args.fp32.resolve().read_text(encoding="utf-8"))
    rows = [make_row("FP32", fp32_metrics, "fp32")]
    for path in sorted(args.qat_root.resolve().glob("*/metrics_qat.json")):
        rows.append(make_row(path.parent.name, json.loads(path.read_text(encoding="utf-8")), "qat"))
    with (output / "qat_comparison.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = [str(row["run"]) for row in rows]
    valid = [float(row["validation_macro_f1"]) for row in rows]
    test = [float(row["test_macro_f1"]) for row in rows]
    positions = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar([value - 0.18 for value in positions], valid, width=0.36, label="Validation")
    axis.bar([value + 0.18 for value in positions], test, width=0.36, label="Test")
    axis.set_xticks(positions, names, rotation=18, ha="right")
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0.75, 0.92)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for index, value in enumerate(valid):
        axis.text(index - 0.18, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    for index, value in enumerate(test):
        axis.text(index + 0.18, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "qat_comparison.png", dpi=180)
    plt.close(figure)

    qats = [row for row in rows if row["kind"] == "qat"]
    best_accuracy = max(qats, key=lambda row: float(row["validation_macro_f1"]))
    selected = next((row for row in qats if str(row["run"]).startswith("w4a6")), best_accuracy)
    result = {
        "selection_policy": (
            "Use W4A6 when its validation macro F1 is within 0.01 of the best QAT; "
            "final hardware choice remains conditional on FINN resource/timing reports."
        ),
        "highest_validation_qat": best_accuracy["run"],
        "provisional_fpga_qat": selected["run"],
        "rows": rows,
    }
    (output / "qat_comparison.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
