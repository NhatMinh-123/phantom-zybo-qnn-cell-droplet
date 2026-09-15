#!/usr/bin/env python3
"""Create a center-cropped YOLO dataset matching the 120 x 120 runtime ROI."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
CLASS_NAMES = ("cell", "droplet")


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Sample:
    image: Path
    labels: tuple[Box, ...]
    crop: tuple[int, int, int, int]
    ambiguous_fragment: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop the labeled 15 um tiles to the centered ROI120 field of view."
    )
    parser.add_argument(
        "--source", type=Path, default=Path("dataset/15micro_yolov11_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset/15micro_roi120_center_v1")
    )
    parser.add_argument("--reference-size", type=int, default=640)
    parser.add_argument("--crop-x", type=int, default=170)
    parser.add_argument("--crop-y", type=int, default=170)
    parser.add_argument("--crop-size", type=int, default=300)
    parser.add_argument("--minimum-visible-fraction", type=float, default=0.35)
    parser.add_argument("--maximum-background-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {label_path}:{line_number}")
        class_id = int(fields[0])
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"Unexpected class {class_id} at {label_path}:{line_number}")
        cx, cy, box_width, box_height = map(float, fields[1:])
        boxes.append(
            Box(
                class_id=class_id,
                x1=(cx - box_width / 2.0) * width,
                y1=(cy - box_height / 2.0) * height,
                x2=(cx + box_width / 2.0) * width,
                y2=(cy + box_height / 2.0) * height,
            )
        )
    return boxes


def scaled_crop(
    width: int,
    height: int,
    reference_size: int,
    crop_x: int,
    crop_y: int,
    crop_size: int,
) -> tuple[int, int, int, int]:
    x1 = round(crop_x * width / reference_size)
    y1 = round(crop_y * height / reference_size)
    x2 = round((crop_x + crop_size) * width / reference_size)
    y2 = round((crop_y + crop_size) * height / reference_size)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Crop {(x1, y1, x2, y2)} is outside image {width}x{height}")
    return x1, y1, x2, y2


def clip_boxes(
    boxes: list[Box],
    crop: tuple[int, int, int, int],
    minimum_visible_fraction: float,
) -> tuple[tuple[Box, ...], bool]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    retained: list[Box] = []
    ambiguous_fragment = False
    for box in boxes:
        intersection_x1 = max(box.x1, crop_x1)
        intersection_y1 = max(box.y1, crop_y1)
        intersection_x2 = min(box.x2, crop_x2)
        intersection_y2 = min(box.y2, crop_y2)
        intersection_width = max(0.0, intersection_x2 - intersection_x1)
        intersection_height = max(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        original_area = max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)
        if intersection_area <= 0.0 or original_area <= 0.0:
            continue
        center_x = (box.x1 + box.x2) / 2.0
        center_y = (box.y1 + box.y2) / 2.0
        center_inside = crop_x1 <= center_x <= crop_x2 and crop_y1 <= center_y <= crop_y2
        visible_fraction = intersection_area / original_area
        keep = center_inside or visible_fraction >= minimum_visible_fraction
        if keep and intersection_width >= 2.0 and intersection_height >= 2.0:
            retained.append(
                Box(
                    class_id=box.class_id,
                    x1=intersection_x1 - crop_x1,
                    y1=intersection_y1 - crop_y1,
                    x2=intersection_x2 - crop_x1,
                    y2=intersection_y2 - crop_y1,
                )
            )
        else:
            ambiguous_fragment = True
    return tuple(retained), ambiguous_fragment


def inspect_split(args: argparse.Namespace, split: str) -> list[Sample]:
    image_dir = args.source / split / "images"
    label_dir = args.source / split / "labels"
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    samples: list[Sample] = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        height, width = image.shape
        crop = scaled_crop(
            width,
            height,
            args.reference_size,
            args.crop_x,
            args.crop_y,
            args.crop_size,
        )
        boxes = read_boxes(label_dir / f"{image_path.stem}.txt", width, height)
        retained, ambiguous_fragment = clip_boxes(
            boxes, crop, args.minimum_visible_fraction
        )
        samples.append(
            Sample(
                image=image_path,
                labels=retained,
                crop=crop,
                ambiguous_fragment=ambiguous_fragment and not retained,
            )
        )
    return samples


def choose_samples(
    samples: list[Sample], maximum_background_ratio: float, seed: int
) -> tuple[list[Sample], dict[str, int]]:
    positive = [sample for sample in samples if sample.labels]
    clean_background = [
        sample
        for sample in samples
        if not sample.labels and not sample.ambiguous_fragment
    ]
    ambiguous = [sample for sample in samples if sample.ambiguous_fragment]
    maximum_backgrounds = round(len(positive) * maximum_background_ratio)
    rng = random.Random(seed)
    rng.shuffle(clean_background)
    selected_background = clean_background[:maximum_backgrounds]
    selected = sorted(positive + selected_background, key=lambda sample: sample.image.name)
    stats = {
        "source_images": len(samples),
        "positive_images": len(positive),
        "clean_background_candidates": len(clean_background),
        "selected_background_images": len(selected_background),
        "dropped_ambiguous_fragments": len(ambiguous),
        "selected_images": len(selected),
    }
    return selected, stats


def write_sample(args: argparse.Namespace, split: str, sample: Sample) -> None:
    image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {sample.image}")
    crop_x1, crop_y1, crop_x2, crop_y2 = sample.crop
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
    image_output = args.output / split / "images" / sample.image.name
    label_output = args.output / split / "labels" / f"{sample.image.stem}.txt"
    if not cv2.imwrite(str(image_output), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Cannot write image: {image_output}")
    crop_height, crop_width = cropped.shape[:2]
    rows = []
    for box in sample.labels:
        center_x = (box.x1 + box.x2) / (2.0 * crop_width)
        center_y = (box.y1 + box.y2) / (2.0 * crop_height)
        box_width = (box.x2 - box.x1) / crop_width
        box_height = (box.y2 - box.y1) / crop_height
        rows.append(
            f"{box.class_id} {center_x:.8f} {center_y:.8f} "
            f"{box_width:.8f} {box_height:.8f}"
        )
    label_output.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_visible_fraction <= 1.0:
        raise ValueError("minimum-visible-fraction must be between 0 and 1")
    if not 0.0 <= args.maximum_background_ratio <= 1.0:
        raise ValueError("maximum-background-ratio must be between 0 and 1")
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if args.output.exists() and any(args.output.iterdir()) and not args.dry_run:
        raise FileExistsError(f"Output directory is not empty: {args.output}")

    split_summary: dict[str, dict[str, object]] = {}
    selected_by_split: dict[str, list[Sample]] = {}
    for split_index, split in enumerate(("train", "valid", "test")):
        samples = inspect_split(args, split)
        selected, stats = choose_samples(
            samples,
            args.maximum_background_ratio,
            args.seed + split_index,
        )
        class_boxes = {
            CLASS_NAMES[class_id]: sum(
                box.class_id == class_id
                for sample in selected
                for box in sample.labels
            )
            for class_id in range(len(CLASS_NAMES))
        }
        stats["class_boxes"] = class_boxes
        split_summary[split] = stats
        selected_by_split[split] = selected

    summary = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "runtime_mapping": {
            "source_tile_reference": [256, 256],
            "previous_runtime_roi": [568, 350, 240, 240],
            "new_runtime_roi": [628, 410, 120, 120],
            "dataset_reference": [args.reference_size, args.reference_size],
            "dataset_crop": [args.crop_x, args.crop_y, args.crop_size, args.crop_size],
            "recommended_qnn_input": [96, 96],
        },
        "minimum_visible_fraction": args.minimum_visible_fraction,
        "maximum_background_ratio": args.maximum_background_ratio,
        "splits": split_summary,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    for split, selected in selected_by_split.items():
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)
        for sample in selected:
            write_sample(args, split, sample)

    data_yaml = (
        f"path: {args.output.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {list(CLASS_NAMES)}\n"
    )
    (args.output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
