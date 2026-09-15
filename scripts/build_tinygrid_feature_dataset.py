#!/usr/bin/env python3
"""Create fixed-point TinyGridNet feature maps from labeled video frames."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygrid_qnn.config import TinyGridConfig
from tinygrid_qnn.labels import boxes_to_occupancy, canvas_boxes_to_compact, read_yolo_boxes
from tinygrid_qnn.preprocess import crop_gray_roi, feature_maps_from_gray, quantize_unsigned


ROOT = Path(__file__).resolve().parents[1]
NAME_PATTERN = re.compile(
    r"^(?P<prefix>.+)_src(?P<frame>\d+)_tile(?P<tile>\d+)_x(?P<x>\d+)_y(?P<y>\d+)$"
)


@dataclass(frozen=True)
class SourceSample:
    stem: str
    source_frame: int
    tile_index: int
    tile_x: int
    tile_y: int
    label_path: Path
    original_split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "dataset" / "cell_droplet_roi384_grouped",
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "raw" / "3.4.mp4")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "cell_droplet_tinygrid_feature_v1",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("chronological", "preserve"),
        default="chronological",
    )
    parser.add_argument("--input-bits", type=int, default=8)
    parser.add_argument("--preview-count", type=int, default=6)
    return parser.parse_args()


def collect_samples(root: Path) -> list[SourceSample]:
    samples: list[SourceSample] = []
    for split in ("train", "valid", "test"):
        for label_path in sorted((root / split / "labels").glob("*.txt")):
            match = NAME_PATTERN.match(label_path.stem)
            if match is None:
                raise ValueError(f"Unsupported label filename: {label_path.name}")
            samples.append(
                SourceSample(
                    label_path.stem,
                    int(match.group("frame")),
                    int(match.group("tile")),
                    int(match.group("x")),
                    int(match.group("y")),
                    label_path,
                    split,
                )
            )
    if not samples:
        raise RuntimeError(f"No labels found under {root}")
    if len({sample.stem for sample in samples}) != len(samples):
        raise RuntimeError("Duplicate sample stems found")
    return samples


def assign_splits(samples: list[SourceSample], strategy: str) -> dict[int, str]:
    if strategy == "preserve":
        return {sample.source_frame: sample.original_split for sample in samples}
    frames = sorted({sample.source_frame for sample in samples})
    if len(frames) < 10:
        raise RuntimeError("Chronological split requires at least ten source frames")
    train_end = round(len(frames) * 0.70)
    valid_end = train_end + round(len(frames) * 0.15)
    result: dict[int, str] = {}
    for index, frame in enumerate(frames):
        result[frame] = "train" if index < train_end else "valid" if index < valid_end else "test"
    return result


def read_frame_pair(capture: cv2.VideoCapture, one_based_frame: int) -> tuple[np.ndarray, np.ndarray]:
    current_index = one_based_frame - 1
    previous_index = max(0, current_index - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, previous_index)
    ok, previous = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {previous_index + 1}")
    if previous_index == current_index:
        return previous, previous.copy()
    ok, current = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {current_index + 1}")
    return previous, current


def make_preview(features: np.ndarray, target: np.ndarray, caption: str) -> np.ndarray:
    panels = [cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR) for channel in features]
    overlay = panels[0].copy()
    cell_width = overlay.shape[1] / target.shape[2]
    cell_height = overlay.shape[0] / target.shape[1]
    colors = ((40, 220, 40), (255, 150, 20))
    for class_id, color in enumerate(colors):
        for grid_y, grid_x in np.argwhere(target[class_id] > 0):
            x1 = round(grid_x * cell_width)
            y1 = round(grid_y * cell_height)
            x2 = round((grid_x + 1) * cell_width) - 1
            y2 = round((grid_y + 1) * cell_height) - 1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    panels.append(overlay)
    scale = 2
    panels = [cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for panel in panels]
    strip = np.concatenate(panels, axis=1)
    header = np.zeros((30, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate((header, strip), axis=0)


def main() -> None:
    args = parse_args()
    config = TinyGridConfig()
    labels_root = args.labels.resolve()
    video_path = args.video.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(labels_root)
    frame_splits = assign_splits(samples, args.split_strategy)
    by_frame: dict[int, list[SourceSample]] = {}
    for sample in samples:
        by_frame.setdefault(sample.source_frame, []).append(sample)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))

    rows: list[dict[str, object]] = []
    previews: list[np.ndarray] = []
    split_counts = {split: 0 for split in ("train", "valid", "test")}
    box_counts = {split: np.zeros(config.output_channels, dtype=np.int64) for split in split_counts}
    occupancy_counts = {split: np.zeros(config.output_channels, dtype=np.int64) for split in split_counts}
    overlapping_cells = {split: 0 for split in split_counts}

    for source_frame in sorted(by_frame):
        previous_frame, current_frame = read_frame_pair(capture, source_frame)
        split = frame_splits[source_frame]
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for sample in sorted(by_frame[source_frame], key=lambda item: item.tile_index):
            roi_x = sample.tile_x + config.source_roi_offset_x
            roi_y = sample.tile_y + config.source_roi_offset_y
            current_gray = crop_gray_roi(
                current_frame,
                x=roi_x,
                y=roi_y,
                width=config.source_roi_width,
                height=config.source_roi_height,
                output_width=config.input_width,
                output_height=config.input_height,
            )
            previous_gray = crop_gray_roi(
                previous_frame,
                x=roi_x,
                y=roi_y,
                width=config.source_roi_width,
                height=config.source_roi_height,
                output_width=config.input_width,
                output_height=config.input_height,
            )
            features = quantize_unsigned(
                feature_maps_from_gray(current_gray, previous_gray), args.input_bits
            )
            boxes = canvas_boxes_to_compact(
                read_yolo_boxes(sample.label_path, config.output_channels), config
            )
            target = boxes_to_occupancy(boxes, config)
            target_path = split_dir / f"{sample.stem}.npz"
            np.savez_compressed(target_path, features=features, target=target)

            per_class_boxes = [sum(box.class_id == class_id for box in boxes) for class_id in range(config.output_channels)]
            positives = target.sum(axis=(1, 2), dtype=np.int64)
            split_counts[split] += 1
            box_counts[split] += per_class_boxes
            occupancy_counts[split] += positives
            overlapping_cells[split] += int(np.logical_and(target[0], target[1]).sum())
            rows.append(
                {
                    "split": split,
                    "stem": sample.stem,
                    "source_frame": source_frame,
                    "source_time_s": f"{(source_frame - 1) / source_fps:.6f}",
                    "tile_index": sample.tile_index,
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "roi_width": config.source_roi_width,
                    "roi_height": config.source_roi_height,
                    "cell_boxes": per_class_boxes[0],
                    "droplet_boxes": per_class_boxes[1],
                    "cell_positive_cells": int(positives[0]),
                    "droplet_positive_cells": int(positives[1]),
                    "original_split": sample.original_split,
                    "label_path": str(sample.label_path),
                    "feature_file": str(target_path),
                }
            )
            if len(previews) < args.preview_count:
                previews.append(make_preview(features, target, f"{split} | {sample.stem}"))
    capture.release()

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    preview = np.concatenate(previews, axis=0)
    cv2.imwrite(str(output / "feature_preview.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 94])

    report = {
        "format": "TinyGridNet fixed-point feature-map dataset v1",
        "video": str(video_path),
        "source_video": {
            "width": source_width,
            "height": source_height,
            "fps": source_fps,
        },
        "label_source": str(labels_root),
        "split_strategy": args.split_strategy,
        "split_limitation": "One labeled video is available; chronological blocks are used until independent videos are labeled.",
        "frame_groups": {
            split: sorted(frame for frame, assigned in frame_splits.items() if assigned == split)
            for split in split_counts
        },
        "samples": split_counts,
        "boxes": {
            split: {name: int(box_counts[split][index]) for index, name in enumerate(config.class_names)}
            for split in split_counts
        },
        "positive_grid_cells": {
            split: {
                name: int(occupancy_counts[split][index])
                for index, name in enumerate(config.class_names)
            }
            for split in split_counts
        },
        "overlapping_cell_droplet_grid_cells": overlapping_cells,
        "input_bits": args.input_bits,
        "channel_order": list(config.feature_names),
        "resize": "integer-phase nearest neighbor",
        "config": config.to_dict(),
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
