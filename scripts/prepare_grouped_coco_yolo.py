#!/usr/bin/env python3
"""Convert a Roboflow COCO export to grouped YOLO train/valid/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_RE = re.compile(r"src(\d+)", re.IGNORECASE)
TILE_RE = re.compile(r"tile(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Roboflow COCO export root.")
    parser.add_argument("--output", type=Path, required=True, help="New YOLO dataset root.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--classes", default="cell,droplet")
    return parser.parse_args()


def clean_file_name(file_name: str) -> str:
    marker = "_jpg.rf."
    if marker in file_name:
        return file_name.split(marker, 1)[0] + ".jpg"
    return Path(file_name).name


def source_id_from_name(file_name: str) -> int:
    match = SOURCE_RE.search(file_name)
    if not match:
        raise ValueError(f"Missing srcXXXXXX in file name: {file_name}")
    return int(match.group(1))


def tile_id_from_name(file_name: str) -> int:
    match = TILE_RE.search(file_name)
    return int(match.group(1)) if match else 0


def load_records(root: Path, keep_classes: set[str]) -> list[dict]:
    records = []
    output_names = set()
    for original_split in ("train", "valid", "test"):
        split_dir = root / original_split
        annotation_path = split_dir / "_annotations.coco.json"
        if not annotation_path.exists():
            continue
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        category_names = {
            int(category["id"]): str(category["name"]).strip().lower()
            for category in data.get("categories", [])
        }
        annotations = defaultdict(list)
        for annotation in data.get("annotations", []):
            class_name = category_names.get(int(annotation["category_id"]), "")
            if class_name in keep_classes:
                annotations[int(annotation["image_id"])].append(
                    {"class_name": class_name, "bbox": annotation["bbox"]}
                )

        for image in data.get("images", []):
            file_name = str(image["file_name"])
            image_path = split_dir / file_name
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            output_name = clean_file_name(file_name)
            if output_name in output_names:
                raise ValueError(f"Duplicate normalized image name: {output_name}")
            output_names.add(output_name)
            records.append(
                {
                    "original_split": original_split,
                    "source_path": image_path,
                    "original_file": file_name,
                    "output_file": output_name,
                    "source_id": source_id_from_name(file_name),
                    "tile_id": tile_id_from_name(file_name),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "annotations": annotations.get(int(image["id"]), []),
                }
            )
    return records


def make_source_split(source_ids: list[int], train_ratio: float, valid_ratio: float) -> dict[int, str]:
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Ratios must leave non-empty train, valid, and test portions.")
    count = len(source_ids)
    train_count = round(count * train_ratio)
    valid_count = round(count * valid_ratio)
    train_count = min(count - 2, max(1, train_count))
    valid_count = min(count - train_count - 1, max(1, valid_count))
    assignment = {}
    for index, source_id in enumerate(source_ids):
        if index < train_count:
            split = "train"
        elif index < train_count + valid_count:
            split = "valid"
        else:
            split = "test"
        assignment[source_id] = split
    return assignment


def yolo_lines(record: dict, class_ids: dict[str, int]) -> tuple[list[str], Counter]:
    width = record["width"]
    height = record["height"]
    lines = []
    counts = Counter()
    for annotation in record["annotations"]:
        x, y, box_width, box_height = map(float, annotation["bbox"])
        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + box_width))
        y2 = max(0.0, min(float(height), y + box_height))
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue
        class_name = annotation["class_name"]
        lines.append(
            f"{class_ids[class_name]} "
            f"{((x1 + x2) / 2) / width:.6f} {((y1 + y2) / 2) / height:.6f} "
            f"{box_width / width:.6f} {box_height / height:.6f}"
        )
        counts[class_name] += 1
    return lines, counts


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_root}")

    class_names = [name.strip().lower() for name in args.classes.split(",") if name.strip()]
    if not class_names or len(set(class_names)) != len(class_names):
        raise SystemExit("--classes must contain unique class names.")
    class_ids = {name: index for index, name in enumerate(class_names)}
    records = load_records(input_root, set(class_names))
    if not records:
        raise SystemExit("No images found in the COCO export.")

    source_ids = sorted({record["source_id"] for record in records})
    source_split = make_source_split(source_ids, args.train_ratio, args.valid_ratio)
    for split in ("train", "valid", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    split_images = Counter()
    split_sources = defaultdict(set)
    split_classes = defaultdict(Counter)
    manifest_rows = []
    for record in sorted(records, key=lambda row: (row["source_id"], row["tile_id"])):
        split = source_split[record["source_id"]]
        image_target = output_root / split / "images" / record["output_file"]
        label_target = output_root / split / "labels" / (Path(record["output_file"]).stem + ".txt")
        shutil.copy2(record["source_path"], image_target)
        lines, class_counts = yolo_lines(record, class_ids)
        label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")

        split_images[split] += 1
        split_sources[split].add(record["source_id"])
        split_classes[split].update(class_counts)
        manifest_rows.append(
            {
                "source_id": record["source_id"],
                "tile_id": record["tile_id"],
                "original_split": record["original_split"],
                "new_split": split,
                "original_file": record["original_file"],
                "output_file": record["output_file"],
                **{f"boxes_{name}": class_counts[name] for name in class_names},
            }
        )

    with (output_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    yaml_lines = [
        f"path: {output_root.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "names: [" + ", ".join(repr(name) for name in class_names) + "]",
    ]
    (output_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="ascii")

    report_lines = [
        f"Input: {input_root}",
        f"Images: {len(records)}",
        f"Unique source frames: {len(source_ids)}",
        f"Classes: {', '.join(class_names)}",
    ]
    for split in ("train", "valid", "test"):
        report_lines.append(
            f"{split}: images={split_images[split]} sources={len(split_sources[split])} "
            + " ".join(f"{name}={split_classes[split][name]}" for name in class_names)
        )
    (output_root / "dataset_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"YOLO dataset: {output_root}")
    print(f"Data config: {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
