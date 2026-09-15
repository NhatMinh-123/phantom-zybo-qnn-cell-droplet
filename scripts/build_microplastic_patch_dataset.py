"""Build a grouped 32x32 particle/background patch dataset.

Class-0 boxes from the existing cell/droplet detector dataset are used as
bootstrap particle positives. Hard negatives are sampled from high black-hat
responses inside droplet interiors while excluding every annotated class-0
box. The source train/valid/test split is preserved exactly.

The generated dataset is intentionally separate from all detector datasets and
models. Candidate patches mined from an unlabeled video are not added here;
they must be reviewed first to avoid reinforcing annotation mistakes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Box:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def x1(self) -> float:
        return self.center_x - self.width / 2.0

    @property
    def y1(self) -> float:
        return self.center_y - self.height / 2.0

    @property
    def x2(self) -> float:
        return self.center_x + self.width / 2.0

    @property
    def y2(self) -> float:
        return self.center_y + self.height / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grouped particle/background patches from YOLO labels."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "dataset" / "cell_droplet_roi384_grouped",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "microplastic_patch32_grouped_v1",
    )
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--context-scale", type=float, default=2.4)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.75)
    parser.add_argument("--blackhat-kernel", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected five YOLO values"
            )
        class_id = int(parts[0])
        center_x, center_y, width, height = map(float, parts[1:])
        boxes.append(
            Box(
                class_id=class_id,
                center_x=center_x * image_width,
                center_y=center_y * image_height,
                width=width * image_width,
                height=height * image_height,
            )
        )
    return boxes


def crop_square(
    gray: np.ndarray,
    center_x: float,
    center_y: float,
    side: int,
    output_size: int,
) -> np.ndarray:
    side = max(4, int(round(side)))
    half = side // 2 + 2
    padded = cv2.copyMakeBorder(
        gray,
        half,
        half,
        half,
        half,
        cv2.BORDER_REFLECT_101,
    )
    cx = int(round(center_x)) + half
    cy = int(round(center_y)) + half
    x1 = cx - side // 2
    y1 = cy - side // 2
    crop = padded[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=(
            cv2.INTER_AREA if side >= output_size else cv2.INTER_CUBIC
        ),
    )


def square_iou(
    center_x: float,
    center_y: float,
    side: float,
    box: Box,
) -> float:
    x1 = center_x - side / 2.0
    y1 = center_y - side / 2.0
    x2 = center_x + side / 2.0
    y2 = center_y + side / 2.0
    intersection_width = max(0.0, min(x2, box.x2) - max(x1, box.x1))
    intersection_height = max(0.0, min(y2, box.y2) - max(y1, box.y1))
    intersection = intersection_width * intersection_height
    union = side * side + box.width * box.height - intersection
    return intersection / max(union, 1e-9)


def is_safe_negative(
    x: float,
    y: float,
    side: float,
    positives: list[Box],
) -> bool:
    for box in positives:
        normalized_dx = abs(x - box.center_x) / max(box.width, 1.0)
        normalized_dy = abs(y - box.center_y) / max(box.height, 1.0)
        if normalized_dx < 1.2 and normalized_dy < 1.2:
            return False
        if square_iou(x, y, side, box) > 0.02:
            return False
    return True


def droplet_interior_mask(
    shape: tuple[int, int],
    droplets: list[Box],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for box in droplets:
        axes = (
            max(3, int(round(box.width * 0.31))),
            max(3, int(round(box.height * 0.31))),
        )
        cv2.ellipse(
            mask,
            (int(round(box.center_x)), int(round(box.center_y))),
            axes,
            0,
            0,
            360,
            255,
            -1,
        )
    return mask


def suppress_positive_regions(
    mask: np.ndarray,
    positives: list[Box],
    margin: float,
) -> None:
    height, width = mask.shape
    for box in positives:
        half_width = box.width * margin / 2.0
        half_height = box.height * margin / 2.0
        x1 = max(0, int(math.floor(box.center_x - half_width)))
        y1 = max(0, int(math.floor(box.center_y - half_height)))
        x2 = min(width, int(math.ceil(box.center_x + half_width)))
        y2 = min(height, int(math.ceil(box.center_y + half_height)))
        mask[y1:y2, x1:x2] = 0


def hard_negative_candidates(
    gray: np.ndarray,
    positives: list[Box],
    droplets: list[Box],
    *,
    side: int,
    kernel_size: int,
) -> list[tuple[float, float, float, str]]:
    if not droplets:
        return []
    interior = droplet_interior_mask(gray.shape, droplets)
    suppress_positive_regions(interior, positives, margin=3.0)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    response = blackhat.copy()
    response[interior == 0] = 0
    nonzero = response[response > 0]
    if not len(nonzero):
        return []
    threshold = max(3.0, float(np.percentile(nonzero, 75)))
    dilated = cv2.dilate(response, np.ones((5, 5), dtype=np.uint8))
    maxima = (
        (response == dilated)
        & (response >= threshold)
        & (interior > 0)
    )
    ys, xs = np.nonzero(maxima)
    ranked = sorted(
        (
            (float(response[y, x]), float(x), float(y))
            for x, y in zip(xs, ys)
        ),
        reverse=True,
    )
    selected: list[tuple[float, float, float, str]] = []
    minimum_distance = max(side * 0.55, 5.0)
    for score, x, y in ranked:
        if not is_safe_negative(x, y, side, positives):
            continue
        if any(
            math.hypot(x - old_x, y - old_y) < minimum_distance
            for old_x, old_y, _, _ in selected
        ):
            continue
        selected.append((x, y, score / 255.0, "blackhat_hard"))
    return selected


def random_negative_candidates(
    gray: np.ndarray,
    positives: list[Box],
    droplets: list[Box],
    *,
    side: int,
    rng: random.Random,
    maximum: int,
) -> list[tuple[float, float, float, str]]:
    interior = droplet_interior_mask(gray.shape, droplets)
    suppress_positive_regions(interior, positives, margin=3.0)
    ys, xs = np.nonzero(interior)
    if not len(xs):
        return []
    indices = list(range(len(xs)))
    rng.shuffle(indices)
    selected: list[tuple[float, float, float, str]] = []
    minimum_distance = max(side * 0.45, 4.0)
    for index in indices:
        x = float(xs[index])
        y = float(ys[index])
        if not is_safe_negative(x, y, side, positives):
            continue
        if any(
            math.hypot(x - old_x, y - old_y) < minimum_distance
            for old_x, old_y, _, _ in selected
        ):
            continue
        selected.append((x, y, 0.0, "interior_random"))
        if len(selected) >= maximum:
            break
    return selected


def write_patch(
    patch: np.ndarray,
    directory: Path,
    filename: str,
) -> Path:
    path = directory / filename
    if not cv2.imwrite(str(path), patch):
        raise RuntimeError(f"Could not write patch: {path}")
    return path


def source_group_from_stem(stem: str) -> str:
    marker = "_src"
    if marker not in stem:
        return stem
    return stem.split(marker, 1)[1].split("_", 1)[0]


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output must be new or empty to preserve prior data: {output}"
        )
    if args.patch_size < 16:
        raise ValueError("patch-size must be at least 16")
    if args.blackhat_kernel < 3 or args.blackhat_kernel % 2 == 0:
        raise ValueError("blackhat-kernel must be odd and at least 3")
    if args.negative_ratio < 0:
        raise ValueError("negative-ratio must be non-negative")
    if not 0 <= args.hard_negative_fraction <= 1:
        raise ValueError("hard-negative-fraction must be between zero and one")

    rng = random.Random(args.seed)
    manifest_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "configuration": {
            "patch_size": args.patch_size,
            "context_scale": args.context_scale,
            "negative_ratio": args.negative_ratio,
            "hard_negative_fraction": args.hard_negative_fraction,
            "blackhat_kernel": args.blackhat_kernel,
            "seed": args.seed,
        },
        "split": {},
        "label_policy": {
            "particle": "existing YOLO class 0 (cell) with context",
            "background": (
                "high black-hat response or random location inside droplet "
                "interior, excluding expanded class-0 regions"
            ),
            "unreviewed_video_candidates": "excluded",
        },
    }

    for split in SPLITS:
        image_directory = source / split / "images"
        label_directory = source / split / "labels"
        if not image_directory.exists() or not label_directory.exists():
            raise FileNotFoundError(f"Missing source split: {split}")
        particle_directory = output / split / "particle"
        background_directory = output / split / "background"
        particle_directory.mkdir(parents=True, exist_ok=True)
        background_directory.mkdir(parents=True, exist_ok=True)

        split_counts = {
            "source_images": 0,
            "particle": 0,
            "background": 0,
            "background_blackhat_hard": 0,
            "background_interior_random": 0,
        }
        image_paths = sorted(
            path
            for path in image_directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = image.shape
            boxes = read_yolo_labels(
                label_directory / f"{image_path.stem}.txt",
                width,
                height,
            )
            positives = [box for box in boxes if box.class_id == 0]
            droplets = [box for box in boxes if box.class_id == 1]
            split_counts["source_images"] += 1

            positive_sides: list[int] = []
            for positive_index, box in enumerate(positives):
                side = int(
                    round(max(box.width, box.height) * args.context_scale)
                )
                side = int(np.clip(side, 16, 72))
                positive_sides.append(side)
                patch = crop_square(
                    image,
                    box.center_x,
                    box.center_y,
                    side,
                    args.patch_size,
                )
                filename = (
                    f"{image_path.stem}_p{positive_index:03d}"
                    f"_x{int(round(box.center_x)):03d}"
                    f"_y{int(round(box.center_y)):03d}.png"
                )
                patch_path = write_patch(
                    patch,
                    particle_directory,
                    filename,
                )
                manifest_rows.append(
                    {
                        "split": split,
                        "class_name": "particle",
                        "class_id": 1,
                        "reason": "annotated_class0",
                        "source_image": str(image_path.resolve()),
                        "source_group": source_group_from_stem(
                            image_path.stem
                        ),
                        "center_x": box.center_x,
                        "center_y": box.center_y,
                        "source_side": side,
                        "response": "",
                        "patch": str(patch_path.resolve()),
                    }
                )
                split_counts["particle"] += 1

            typical_side = (
                int(round(float(np.median(positive_sides))))
                if positive_sides
                else max(args.patch_size, 24)
            )
            target_negatives = max(
                1 if droplets else 0,
                int(round(len(positives) * args.negative_ratio)),
            )
            target_hard = int(
                round(target_negatives * args.hard_negative_fraction)
            )
            hard = hard_negative_candidates(
                image,
                positives,
                droplets,
                side=typical_side,
                kernel_size=args.blackhat_kernel,
            )
            selected = hard[:target_hard]
            random_candidates = random_negative_candidates(
                image,
                positives,
                droplets,
                side=typical_side,
                rng=rng,
                maximum=target_negatives * 3 + 8,
            )
            for candidate in random_candidates:
                if len(selected) >= target_negatives:
                    break
                x, y, _, _ = candidate
                if any(
                    math.hypot(x - old_x, y - old_y)
                    < typical_side * 0.45
                    for old_x, old_y, _, _ in selected
                ):
                    continue
                selected.append(candidate)
            for candidate in hard[target_hard:]:
                if len(selected) >= target_negatives:
                    break
                selected.append(candidate)

            for negative_index, (x, y, response, reason) in enumerate(
                selected[:target_negatives]
            ):
                patch = crop_square(
                    image,
                    x,
                    y,
                    typical_side,
                    args.patch_size,
                )
                filename = (
                    f"{image_path.stem}_n{negative_index:03d}"
                    f"_x{int(round(x)):03d}_y{int(round(y)):03d}.png"
                )
                patch_path = write_patch(
                    patch,
                    background_directory,
                    filename,
                )
                manifest_rows.append(
                    {
                        "split": split,
                        "class_name": "background",
                        "class_id": 0,
                        "reason": reason,
                        "source_image": str(image_path.resolve()),
                        "source_group": source_group_from_stem(
                            image_path.stem
                        ),
                        "center_x": x,
                        "center_y": y,
                        "source_side": typical_side,
                        "response": response,
                        "patch": str(patch_path.resolve()),
                    }
                )
                split_counts["background"] += 1
                split_counts[f"background_{reason}"] += 1

        summary["split"][split] = split_counts

    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "split",
                "class_name",
                "class_id",
                "reason",
                "source_image",
                "source_group",
                "center_x",
                "center_y",
                "source_side",
                "response",
                "patch",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["split"], indent=2))
    print(f"Dataset: {output}")


if __name__ == "__main__":
    main()
