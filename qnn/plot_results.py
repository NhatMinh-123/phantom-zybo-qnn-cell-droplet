from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot QAT training and threshold calibration")
    parser.add_argument("--report", type=Path, default=Path("reports/qnn_cell_droplet"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    training = read_csv(args.report / "training_metrics.csv")
    sweep = read_csv(args.report / "threshold_sweep.csv")
    evaluation = json.loads((args.report / "evaluation.json").read_text(encoding="utf-8"))

    epochs = [int(row["epoch"]) for row in training]
    train_loss = [float(row["train_loss"]) for row in training]
    valid_loss = [float(row["valid_loss"]) for row in training]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, train_loss, label="Train", color="#2563eb")
    axes[0].plot(epochs, valid_loss, label="Validation", color="#dc2626")
    axes[0].set_title("QAT loss W4A4")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    colors = {"cell": "#0891b2", "droplet": "#7c3aed"}
    for class_name in ("cell", "droplet"):
        class_rows = [row for row in sweep if row["class"] == class_name]
        axes[1].plot(
            [float(row["threshold"]) for row in class_rows],
            [float(row["f1"]) for row in class_rows],
            label=class_name,
            color=colors[class_name],
        )
        selected = evaluation["validation"]["classes"][class_name]
        axes[1].scatter(
            [selected["threshold"]],
            [selected["f1"]],
            color=colors[class_name],
            s=55,
            zorder=3,
        )
    axes[1].set_title("Validation threshold sweep")
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("F1 at IoU 0.50")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    test = evaluation["test"]
    labels = ["Overall", "Cell", "Droplet"]
    precision = [
        test["precision"],
        test["classes"]["cell"]["precision"],
        test["classes"]["droplet"]["precision"],
    ]
    recall = [
        test["recall"],
        test["classes"]["cell"]["recall"],
        test["classes"]["droplet"]["recall"],
    ]
    f1 = [
        test["f1"],
        test["classes"]["cell"]["f1"],
        test["classes"]["droplet"]["f1"],
    ]
    positions = range(len(labels))
    width = 0.24
    axes[2].bar([value - width for value in positions], precision, width, label="Precision")
    axes[2].bar(positions, recall, width, label="Recall")
    axes[2].bar([value + width for value in positions], f1, width, label="F1")
    axes[2].set_xticks(list(positions), labels)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Independent test metrics")
    axes[2].set_ylabel("Score")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)

    figure.suptitle("TinyQuantDetector: 5,248 parameters, W4A4, grid 64x64")
    figure.tight_layout()
    output = args.report / "qat_evaluation.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

