from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def translate_horizontal(
    pixels: np.ndarray,
    labels: torch.Tensor,
    shift_pixels: int,
    *,
    minimum_visible_fraction: float = 0.25,
) -> tuple[np.ndarray, torch.Tensor]:
    """Translate an image horizontally and clip YOLO boxes at the new edges."""

    if shift_pixels == 0:
        return pixels, labels
    if pixels.ndim != 2:
        raise ValueError("Expected a grayscale HxW image")
    width = pixels.shape[1]
    if abs(shift_pixels) >= width:
        raise ValueError("Horizontal shift must be smaller than the image width")

    translated = np.empty_like(pixels)
    if shift_pixels > 0:
        translated[:, :shift_pixels] = pixels[:, :1]
        translated[:, shift_pixels:] = pixels[:, :-shift_pixels]
    else:
        amount = -shift_pixels
        translated[:, -amount:] = pixels[:, -1:]
        translated[:, :-amount] = pixels[:, amount:]

    if not len(labels):
        return translated, labels
    shifted = labels.clone()
    delta = shift_pixels / width
    original_width = shifted[:, 3].clone()
    left = (shifted[:, 1] - original_width / 2 + delta).clamp(0.0, 1.0)
    right = (shifted[:, 1] + original_width / 2 + delta).clamp(0.0, 1.0)
    visible_width = right - left
    keep = (visible_width > 0.0) & (
        visible_width >= original_width * minimum_visible_fraction
    )
    shifted = shifted[keep]
    if len(shifted):
        shifted[:, 1] = (left[keep] + right[keep]) / 2
        shifted[:, 3] = visible_width[keep]
    return translated, shifted


@dataclass(frozen=True)
class DatasetAudit:
    split: str
    images: int
    boxes: int
    boxes_per_class: tuple[int, ...]
    collisions_per_grid: dict[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "images": self.images,
            "boxes": self.boxes,
            "boxes_per_class": list(self.boxes_per_class),
            "collisions_per_grid": self.collisions_per_grid,
        }


def _load_labels(label_path: Path, num_classes: int) -> torch.Tensor:
    rows: list[list[float]] = []
    if not label_path.exists():
        return torch.empty((0, 5), dtype=torch.float32)

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{label_path}:{line_number}: box outside normalized range")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError(f"{label_path}:{line_number}: non-positive box size")
        rows.append([float(class_id), *values])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, 5))


class YoloDetectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: Path | str,
        split: str,
        *,
        input_size: int | tuple[int, int] = 256,
        num_classes: int = 2,
        augment: bool = False,
        horizontal_shift: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        if isinstance(input_size, int):
            self.input_width = self.input_height = input_size
        else:
            self.input_width, self.input_height = input_size
        self.num_classes = num_classes
        self.augment = augment
        if not 0.0 <= horizontal_shift < 1.0:
            raise ValueError("horizontal_shift must be in [0, 1)")
        self.horizontal_shift = horizontal_shift
        self.image_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(f"Missing YOLO split directories under {self.root / split}")
        self.images = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        labels = _load_labels(label_path, self.num_classes)

        with Image.open(image_path) as source:
            image = source.convert("L").resize(
                (self.input_width, self.input_height), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(image, dtype=np.float32) / 255.0

        if self.augment:
            if random.random() < 0.5:
                pixels = np.ascontiguousarray(np.fliplr(pixels))
                if len(labels):
                    labels = labels.clone()
                    labels[:, 1] = 1.0 - labels[:, 1]
            maximum_shift = round(self.input_width * self.horizontal_shift)
            if maximum_shift:
                pixels, labels = translate_horizontal(
                    pixels,
                    labels,
                    random.randint(-maximum_shift, maximum_shift),
                )
            gain = random.uniform(0.9, 1.1)
            offset = random.uniform(-0.05, 0.05)
            pixels = np.clip(pixels * gain + offset, 0.0, 1.0)

        tensor = torch.from_numpy(np.ascontiguousarray(pixels)).unsqueeze(0)
        return tensor, labels


def detection_collate(
    samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images, targets = zip(*samples)
    return torch.stack(images), list(targets)


def audit_split(
    root: Path | str,
    split: str,
    *,
    num_classes: int = 2,
    grid_sizes: tuple[int, ...] = (16, 32, 64),
) -> DatasetAudit:
    dataset = YoloDetectionDataset(root, split, num_classes=num_classes)
    class_counts = [0] * num_classes
    collisions = {grid_size: 0 for grid_size in grid_sizes}
    box_count = 0

    for image_path in dataset.images:
        labels = _load_labels(dataset.label_dir / f"{image_path.stem}.txt", num_classes)
        box_count += len(labels)
        for class_id in labels[:, 0].to(torch.int64).tolist():
            class_counts[class_id] += 1
        for grid_size in grid_sizes:
            occupied: set[tuple[int, int, int]] = set()
            for row in labels:
                class_id = int(row[0].item())
                grid_x = min(int(row[1].item() * grid_size), grid_size - 1)
                grid_y = min(int(row[2].item() * grid_size), grid_size - 1)
                key = (class_id, grid_y, grid_x)
                if key in occupied:
                    collisions[grid_size] += 1
                occupied.add(key)

    return DatasetAudit(
        split=split,
        images=len(dataset),
        boxes=box_count,
        boxes_per_class=tuple(class_counts),
        collisions_per_grid=collisions,
    )
