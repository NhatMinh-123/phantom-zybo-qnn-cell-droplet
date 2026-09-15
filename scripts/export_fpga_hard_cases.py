#!/usr/bin/env python3
"""Export the lowest-F1 FPGA samples into a relabel/retraining review queue."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "fpga_sparse_stream_accuracy_full_20260727"
DEFAULT_DATA = ROOT / "dataset" / "cell_droplet_roi384_grouped"
DEFAULT_OUTPUT = ROOT / "reports" / "fpga_sparse_stream_hard_cases_20260727"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=12)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def priority(row: dict[str, str]) -> str:
    false_positive = int(row["false_positive"])
    false_negative = int(row["false_negative"])
    matched_iou = float(row["matched_mean_iou"] or 0.0)
    if matched_iou < 0.60 and false_positive + false_negative:
        return "localization_and_label_review"
    if false_negative > false_positive:
        return "missed_objects"
    if false_positive > false_negative:
        return "false_or_duplicate_objects"
    return "balanced_error_review"


def main() -> None:
    args = parse_args()
    report = args.report.resolve()
    data = args.data.resolve()
    output = args.output.resolve()
    rows = read_rows(report / "per_image.csv")
    rows.sort(
        key=lambda row: (
            float(row["f1"]),
            -(int(row["false_positive"]) + int(row["false_negative"])),
            float(row["matched_mean_iou"] or 0.0),
        )
    )
    selected = rows[: max(args.count, 0)]

    images_out = output / "images"
    labels_out = output / "labels"
    annotated_out = output / "fpga_annotated"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    annotated_out.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        image_name = row["image"]
        stem = Path(image_name).stem
        image_source = data / args.split / "images" / image_name
        label_source = data / args.split / "labels" / f"{stem}.txt"
        annotated_source = report / row["annotated"]
        if not image_source.is_file():
            raise FileNotFoundError(image_source)
        if not label_source.is_file():
            raise FileNotFoundError(label_source)
        if not annotated_source.is_file():
            raise FileNotFoundError(annotated_source)
        shutil.copy2(image_source, images_out / image_source.name)
        shutil.copy2(label_source, labels_out / label_source.name)
        shutil.copy2(annotated_source, annotated_out / annotated_source.name)
        exported.append(
            {
                "rank": rank,
                "image": image_name,
                "priority": priority(row),
                "true_positive": int(row["true_positive"]),
                "false_positive": int(row["false_positive"]),
                "false_negative": int(row["false_negative"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "matched_mean_iou": float(row["matched_mean_iou"] or 0.0),
            }
        )

    with (output / "review_queue.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    summary = {
        "source_report": str(report),
        "source_dataset": str(data),
        "split": args.split,
        "count": len(exported),
        "warning": (
            "These are test samples. Do not move them into train. Use them to "
            "audit labels and extract separately grouped neighboring frames."
        ),
        "items": exported,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = f"""# FPGA hard-case review queue

This folder contains the {len(exported)} lowest-F1 samples from the labeled FPGA
test run.

- `images`: unchanged source tiles
- `labels`: current YOLO ground truth
- `fpga_annotated`: FPGA predictions, with false/duplicate boxes highlighted
- `review_queue.csv`: FP, FN, F1 and localization priority

Important: these samples remain in the test split. Do not train on them. First
correct any objectively wrong labels in place. For additional training data,
extract nearby source frames that are not in train/valid/test, group them by
source frame, label both `cell` and `droplet`, and regenerate the grouped split.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Hard cases: {output}")


if __name__ == "__main__":
    main()
