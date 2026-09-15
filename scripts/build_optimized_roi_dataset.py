#!/usr/bin/env python3
"""Build the selected compact square-ROI dataset while preserving group splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from benchmark_roi_candidates import image_paths, load_data, read_labels, source_geometry
from benchmark_square_roi import MODES, prepare_canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="compact_pad384")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_data = args.data.resolve()
    source_root, names = load_data(source_data)
    mode_lookup = {mode.name: mode for mode in MODES}
    if args.mode not in mode_lookup:
        raise SystemExit(f"Unknown mode {args.mode}; choose from {sorted(mode_lookup)}")
    mode = mode_lookup[args.mode]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    class_counts = {split: {class_id: 0 for class_id in names} for split in ("train", "valid", "test")}
    image_counts: dict[str, int] = {}

    for split in ("train", "valid", "test"):
        images_dir = output / split / "images"
        labels_dir = output / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for image_path in image_paths(source_root, split):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read {image_path}")
            height, width = image.shape[:2]
            labels = read_labels(
                source_root / split / "labels" / f"{image_path.stem}.txt",
                width,
                height,
            )
            canvas, transformed = prepare_canvas(image, labels, mode)
            target_image = images_dir / image_path.name
            if not cv2.imwrite(str(target_image), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {target_image}")
            target_label = labels_dir / f"{image_path.stem}.txt"
            lines: list[str] = []
            for label in transformed:
                x1, y1, x2, y2 = label.box
                center_x = ((x1 + x2) / 2.0) / mode.canvas_size
                center_y = ((y1 + y2) / 2.0) / mode.canvas_size
                box_width = (x2 - x1) / mode.canvas_size
                box_height = (y2 - y1) / mode.canvas_size
                lines.append(
                    f"{label.class_id} {center_x:.8f} {center_y:.8f} "
                    f"{box_width:.8f} {box_height:.8f}"
                )
                class_counts[split][label.class_id] += 1
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
            count += 1
        image_counts[split] = count

    data = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": [names[index] for index in sorted(names)],
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="ascii"
    )
    roi_x, roi_y, roi_width, roi_height = source_geometry(mode.crop)
    report = {
        "source_dataset": str(source_data),
        "mode": mode.name,
        "model_input": [mode.canvas_size, mode.canvas_size],
        "reference_frame": [1280, 800],
        "source_roi": {
            "x": roi_x,
            "y": roi_y,
            "width": roi_width,
            "height": roi_height,
        },
        "images": image_counts,
        "labels": {
            split: {names[class_id]: count for class_id, count in counts.items()}
            for split, counts in class_counts.items()
        },
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="ascii"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
