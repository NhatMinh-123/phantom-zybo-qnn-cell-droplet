#!/usr/bin/env python3
"""Build a train-only YOLO teacher-refined dataset for QNN distillation."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "cell_droplet_yolo11n" / "best.pt"
DEFAULT_DATA = ROOT / "dataset" / "cell_droplet_roi384_grouped"
DEFAULT_OUTPUT = ROOT / "dataset" / "cell_droplet_roi384_teacher_refined_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--blend", type=float, default=0.35)
    parser.add_argument("--cell-add-confidence", type=float, default=0.75)
    parser.add_argument("--droplet-add-confidence", type=float, default=0.85)
    parser.add_argument("--duplicate-iou", type=float, default=0.30)
    return parser.parse_args()


def xywh_to_xyxy(values: Iterable[float]) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = values
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def xyxy_to_xywh(values: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = values
    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        x2 - x1,
        y2 - y1,
    )


def box_iou(first: Box, second: Box) -> float:
    left = max(first.xyxy[0], second.xyxy[0])
    top = max(first.xyxy[1], second.xyxy[1])
    right = min(first.xyxy[2], second.xyxy[2])
    bottom = min(first.xyxy[3], second.xyxy[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first.xyxy[2] - first.xyxy[0]) * max(
        0.0, first.xyxy[3] - first.xyxy[1]
    )
    second_area = max(0.0, second.xyxy[2] - second.xyxy[0]) * max(
        0.0, second.xyxy[3] - second.xyxy[1]
    )
    union = first_area + second_area - intersection
    return intersection / max(union, 1e-9)


def read_labels(path: Path) -> list[Box]:
    boxes: list[Box] = []
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label: {path}: {line!r}")
        boxes.append(
            Box(
                class_id=int(fields[0]),
                xyxy=xywh_to_xyxy(float(value) for value in fields[1:]),
            )
        )
    return boxes


def blend_box(ground_truth: Box, teacher: Box, weight: float) -> Box:
    return Box(
        class_id=ground_truth.class_id,
        xyxy=tuple(
            (1.0 - weight) * original + weight * refined
            for original, refined in zip(ground_truth.xyxy, teacher.xyxy)
        ),
    )


def refine_labels(
    ground_truth: list[Box],
    teacher: list[Box],
    *,
    match_iou: float,
    blend: float,
    add_confidence: tuple[float, float],
    duplicate_iou: float,
) -> tuple[list[Box], dict[str, int | float]]:
    refined: list[Box] = []
    used_teacher: set[int] = set()
    matched = 0
    match_ious: list[float] = []
    for target in ground_truth:
        choices = [
            (box_iou(target, candidate), index)
            for index, candidate in enumerate(teacher)
            if index not in used_teacher
            and candidate.class_id == target.class_id
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_index >= 0 and best_iou >= match_iou:
            refined.append(blend_box(target, teacher[best_index], blend))
            used_teacher.add(best_index)
            matched += 1
            match_ious.append(best_iou)
        else:
            refined.append(target)

    added = 0
    for index, candidate in sorted(
        enumerate(teacher),
        key=lambda item: item[1].confidence,
        reverse=True,
    ):
        threshold = add_confidence[candidate.class_id]
        if index in used_teacher or candidate.confidence < threshold:
            continue
        same_class = [item for item in refined if item.class_id == candidate.class_id]
        if max((box_iou(candidate, item) for item in same_class), default=0.0) >= duplicate_iou:
            continue
        refined.append(Box(candidate.class_id, candidate.xyxy))
        added += 1
    refined.sort(key=lambda item: (item.class_id, item.xyxy[0], item.xyxy[1]))
    return refined, {
        "ground_truth": len(ground_truth),
        "teacher": len(teacher),
        "matched": matched,
        "added": added,
        "output": len(refined),
        "mean_match_iou": (
            float(np.mean(match_ious)) if match_ious else 0.0
        ),
    }


def write_labels(path: Path, boxes: list[Box]) -> None:
    lines = []
    for item in boxes:
        center_x, center_y, width, height = xyxy_to_xywh(item.xyxy)
        values = (
            min(1.0, max(0.0, center_x)),
            min(1.0, max(0.0, center_y)),
            min(1.0, max(1e-8, width)),
            min(1.0, max(1e-8, height)),
        )
        lines.append(
            f"{item.class_id} {values[0]:.8f} {values[1]:.8f} "
            f"{values[2]:.8f} {values[3]:.8f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")


def copy_split(source: Path, destination: Path, split: str) -> None:
    for kind in ("images", "labels"):
        target = destination / split / kind
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted((source / split / kind).iterdir()):
            if path.is_file() and (
                kind == "labels" and path.suffix.lower() == ".txt"
                or kind == "images" and path.suffix.lower() in IMAGE_SUFFIXES
            ):
                shutil.copy2(path, target / path.name)


def main() -> None:
    args = parse_args()
    source = args.data.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("--blend must be in [0, 1]")
    for split in ("train", "valid", "test"):
        copy_split(source, output, split)

    train_images = sorted(
        path
        for path in (source / "train" / "images").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    model = YOLO(str(args.model.resolve()))
    results = model.predict(
        source=[str(path) for path in train_images],
        imgsz=args.imgsz,
        batch=args.batch,
        conf=min(args.cell_add_confidence, args.droplet_add_confidence, 0.25),
        iou=0.5,
        max_det=300,
        device=args.device,
        verbose=False,
        stream=True,
    )
    rows = []
    totals = {
        "ground_truth": 0,
        "teacher": 0,
        "matched": 0,
        "added": 0,
        "output": 0,
    }
    match_iou_weighted = 0.0
    for image_path, result in zip(train_images, results):
        ground_truth = read_labels(
            source / "train" / "labels" / f"{image_path.stem}.txt"
        )
        height, width = result.orig_shape
        teacher = []
        if result.boxes is not None:
            for coordinates, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                teacher.append(
                    Box(
                        class_id=int(class_id),
                        confidence=float(confidence),
                        xyxy=(
                            float(coordinates[0] / width),
                            float(coordinates[1] / height),
                            float(coordinates[2] / width),
                            float(coordinates[3] / height),
                        ),
                    )
                )
        refined, metrics = refine_labels(
            ground_truth,
            teacher,
            match_iou=args.match_iou,
            blend=args.blend,
            add_confidence=(
                args.cell_add_confidence,
                args.droplet_add_confidence,
            ),
            duplicate_iou=args.duplicate_iou,
        )
        write_labels(
            output / "train" / "labels" / f"{image_path.stem}.txt",
            refined,
        )
        rows.append({"image": image_path.name, **metrics})
        for key in totals:
            totals[key] += int(metrics[key])
        match_iou_weighted += float(metrics["mean_match_iou"]) * int(
            metrics["matched"]
        )

    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output).replace("\\", "/"),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": ["cell", "droplet"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "source_dataset": str(source),
        "teacher_model": str(args.model.resolve()),
        "policy": {
            "match_iou": args.match_iou,
            "teacher_box_blend": args.blend,
            "cell_add_confidence": args.cell_add_confidence,
            "droplet_add_confidence": args.droplet_add_confidence,
            "duplicate_iou": args.duplicate_iou,
            "validation_and_test_labels": "unchanged",
        },
        "train_images": len(train_images),
        "totals": {
            **totals,
            "mean_match_iou": match_iou_weighted / max(totals["matched"], 1),
        },
        "per_image": rows,
    }
    (output / "teacher_refinement.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["totals"], indent=2))
    print(f"TEACHER_REFINED_DATASET_READY: {output}")


if __name__ == "__main__":
    main()
