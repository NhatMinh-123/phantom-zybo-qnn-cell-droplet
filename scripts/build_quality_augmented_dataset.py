#!/usr/bin/env python3
"""Create a deterministic image-quality augmented YOLO training dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset" / "cell_droplet_roi384_grouped"
DEFAULT_OUTPUT = ROOT / "dataset" / "cell_droplet_roi384_quality_aug_v1"
DEFAULT_REPORT = ROOT / "reports" / "cell_droplet_roi384_quality_aug_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VARIANTS = (
    "original",
    "dark_low_contrast",
    "bright_uneven",
    "soft_blur",
    "noise_jpeg",
)
CLASS_NAMES = ("cell", "droplet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--contact-sheet-samples", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a deterministic interrupted build without deleting files",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rng(seed: int, image_stem: str, variant: str) -> np.random.Generator:
    token = f"{seed}:{image_stem}:{variant}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(value)


def clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def gamma_adjust(image: np.ndarray, gamma: float) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    return np.power(normalized, gamma) * 255.0


def contrast_adjust(image: np.ndarray, factor: float) -> np.ndarray:
    gray = cv2.cvtColor(clip_uint8(image), cv2.COLOR_BGR2GRAY)
    center = float(np.mean(gray))
    return (image.astype(np.float32) - center) * factor + center


def apply_dark_low_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    gamma = float(rng.uniform(1.18, 1.52))
    contrast = float(rng.uniform(0.72, 0.91))
    gain = float(rng.uniform(0.84, 0.97))
    offset = float(rng.uniform(-8.0, 1.0))
    transformed = contrast_adjust(image, contrast)
    transformed = gamma_adjust(clip_uint8(transformed), gamma)
    transformed = transformed * gain + offset
    return clip_uint8(transformed), {
        "gamma": gamma,
        "contrast": contrast,
        "gain": gain,
        "offset": offset,
    }


def apply_bright_uneven(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    gamma = float(rng.uniform(0.72, 0.91))
    contrast = float(rng.uniform(0.90, 1.08))
    amplitude = float(rng.uniform(8.0, 24.0))
    center_x = float(rng.uniform(0.20, 0.80))
    center_y = float(rng.uniform(0.20, 0.80))
    sigma = float(rng.uniform(0.30, 0.55))
    channel_gains = rng.uniform(0.96, 1.05, size=3).astype(np.float32)

    transformed = gamma_adjust(image, gamma)
    transformed = contrast_adjust(clip_uint8(transformed), contrast)
    height, width = image.shape[:2]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    illumination = amplitude * np.exp(
        -(
            (grid_x - center_x) ** 2
            + (grid_y - center_y) ** 2
        )
        / (2.0 * sigma**2)
    )
    transformed = transformed * channel_gains.reshape(1, 1, 3)
    transformed = transformed + illumination[:, :, None]
    return clip_uint8(transformed), {
        "gamma": gamma,
        "contrast": contrast,
        "illumination_amplitude": amplitude,
        "illumination_center": [center_x, center_y],
        "illumination_sigma": sigma,
        "channel_gains_bgr": channel_gains.tolist(),
    }


def motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D(
        (length / 2 - 0.5, length / 2 - 0.5),
        angle_degrees,
        1.0,
    )
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    total = float(kernel.sum())
    return kernel / total if total > 0.0 else kernel


def apply_soft_blur(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = ("gaussian", "motion", "downsample")[int(rng.integers(0, 3))]
    if mode == "gaussian":
        sigma = float(rng.uniform(0.45, 0.90))
        transformed = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma)
        parameters: dict[str, Any] = {"mode": mode, "sigma": sigma}
    elif mode == "motion":
        length = int(rng.choice([3, 5]))
        angle = float(rng.uniform(-15.0, 15.0))
        transformed = cv2.filter2D(
            image,
            -1,
            motion_kernel(length, angle),
            borderType=cv2.BORDER_REPLICATE,
        )
        parameters = {"mode": mode, "length": length, "angle_degrees": angle}
    else:
        scale = float(rng.uniform(0.72, 0.88))
        height, width = image.shape[:2]
        small = cv2.resize(
            image,
            (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        transformed = cv2.resize(
            small,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        parameters = {"mode": mode, "scale": scale}

    gain = float(rng.uniform(0.96, 1.04))
    offset = float(rng.uniform(-3.0, 3.0))
    transformed = transformed.astype(np.float32) * gain + offset
    parameters.update({"gain": gain, "offset": offset})
    return clip_uint8(transformed), parameters


def jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise RuntimeError("OpenCV failed to encode a JPEG augmentation")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV failed to decode a JPEG augmentation")
    return decoded


def apply_noise_jpeg(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    noise_sigma = float(rng.uniform(3.0, 8.0))
    salt_pepper_fraction = float(rng.uniform(0.0, 0.0006))
    jpeg_quality = int(rng.integers(42, 73))
    noise = rng.normal(0.0, noise_sigma, size=image.shape).astype(np.float32)
    transformed = clip_uint8(image.astype(np.float32) + noise)

    affected = round(image.shape[0] * image.shape[1] * salt_pepper_fraction)
    if affected:
        ys = rng.integers(0, image.shape[0], size=affected)
        xs = rng.integers(0, image.shape[1], size=affected)
        values = rng.choice([0, 255], size=affected).astype(np.uint8)
        transformed[ys, xs] = values[:, None]
    transformed = jpeg_round_trip(transformed, jpeg_quality)
    return transformed, {
        "noise_sigma": noise_sigma,
        "salt_pepper_fraction": salt_pepper_fraction,
        "jpeg_quality": jpeg_quality,
    }


def apply_quality_variant(
    image: np.ndarray,
    variant: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR HxWx3 image")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image pixels")
    if variant == "original":
        return image.copy(), {}
    if variant == "dark_low_contrast":
        return apply_dark_low_contrast(image, rng)
    if variant == "bright_uneven":
        return apply_bright_uneven(image, rng)
    if variant == "soft_blur":
        return apply_soft_blur(image, rng)
    if variant == "noise_jpeg":
        return apply_noise_jpeg(image, rng)
    raise ValueError(f"Unknown quality variant: {variant}")


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "brightness_mean": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "sharpness_laplacian_var": float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
        "clipped_dark_fraction": float(np.mean(gray <= 3)),
        "clipped_bright_fraction": float(np.mean(gray >= 252)),
    }


def image_paths(root: Path, split: str) -> list[Path]:
    directory = root / split / "images"
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No images found in {directory}")
    return paths


def parse_label_rows(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        coordinates = tuple(float(value) for value in fields[1:])
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"{path}:{line_number}: invalid class {class_id}")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise ValueError(f"{path}:{line_number}: non-positive box size")
        rows.append((class_id, *coordinates))
    return rows


def draw_boxes(image: np.ndarray, label_path: Path) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    colors = ((40, 40, 235), (235, 145, 20))
    for class_id, center_x, center_y, box_width, box_height in parse_label_rows(
        label_path
    ):
        x1 = round((center_x - box_width / 2.0) * width)
        y1 = round((center_y - box_height / 2.0) * height)
        x2 = round((center_x + box_width / 2.0) * width)
        y2 = round((center_y + box_height / 2.0) * height)
        cv2.rectangle(result, (x1, y1), (x2, y2), colors[class_id], 2)
    return result


def write_contact_sheet(
    source_root: Path,
    output_root: Path,
    report_root: Path,
    sample_count: int,
) -> None:
    train_images = image_paths(source_root, "train")
    sample_count = min(max(sample_count, 1), len(train_images))
    indices = np.linspace(0, len(train_images) - 1, sample_count, dtype=int)
    tile_size = 256
    title_height = 30
    sheet = np.full(
        (
            sample_count * (tile_size + title_height),
            len(VARIANTS) * tile_size,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    for row_index, source_index in enumerate(indices):
        source_path = train_images[int(source_index)]
        for column_index, variant in enumerate(VARIANTS):
            if variant == "original":
                image_path = output_root / "train" / "images" / source_path.name
                label_path = (
                    output_root / "train" / "labels" / f"{source_path.stem}.txt"
                )
            else:
                stem = f"{source_path.stem}__aug_{variant}"
                image_path = output_root / "train" / "images" / f"{stem}.jpg"
                label_path = output_root / "train" / "labels" / f"{stem}.txt"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read generated image: {image_path}")
            image = draw_boxes(image, label_path)
            image = cv2.resize(
                image,
                (tile_size, tile_size),
                interpolation=cv2.INTER_AREA,
            )
            y1 = row_index * (tile_size + title_height)
            x1 = column_index * tile_size
            sheet[y1 + title_height : y1 + title_height + tile_size, x1 : x1 + tile_size] = image
            cv2.putText(
                sheet,
                variant,
                (x1 + 7, y1 + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    path = report_root / "quality_variants_contact_sheet.jpg"
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write contact sheet: {path}")


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def write_quality_plot(
    records: list[dict[str, Any]],
    report_root: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grouped = {
        variant: [record for record in records if record["variant"] == variant]
        for variant in VARIANTS
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    fields = (
        ("brightness_mean", "Brightness mean"),
        ("contrast_std", "Contrast standard deviation"),
        ("sharpness_laplacian_var", "Laplacian sharpness"),
    )
    labels = [variant.replace("_", "\n") for variant in VARIANTS]
    for axis, (field, title) in zip(axes, fields):
        axis.boxplot(
            [[float(row[field]) for row in grouped[variant]] for variant in VARIANTS],
            labels=labels,
            showfliers=False,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Train-only microscopy quality augmentation")
    figure.tight_layout()
    figure.savefig(
        report_root / "quality_distributions.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def copy_original_split(source: Path, output: Path, split: str) -> None:
    images = image_paths(source, split)
    image_output = output / split / "images"
    label_output = output / split / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        label_path = source / split / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        parse_label_rows(label_path)
        shutil.copy2(image_path, image_output / image_path.name)
        shutil.copy2(label_path, label_output / label_path.name)


def verify_train_outputs(
    source: Path,
    output: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    source_shapes: dict[str, tuple[int, ...]] = {}
    hashes_by_source: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_name = str(record["source_image"])
        generated_name = str(record["generated_image"])
        source_image_path = source / "train" / "images" / source_name
        generated_image_path = output / "train" / "images" / generated_name
        source_label_path = (
            source / "train" / "labels" / f"{Path(source_name).stem}.txt"
        )
        generated_label_path = (
            output / "train" / "labels" / f"{Path(generated_name).stem}.txt"
        )

        if source_name not in source_shapes:
            source_image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
            if source_image is None:
                raise RuntimeError(f"Could not read source image: {source_image_path}")
            source_shapes[source_name] = source_image.shape

        generated_image = cv2.imread(
            str(generated_image_path),
            cv2.IMREAD_COLOR,
        )
        if generated_image is None:
            raise RuntimeError(
                f"Could not read generated image: {generated_image_path}"
            )
        if generated_image.shape != source_shapes[source_name]:
            raise AssertionError(
                f"Image shape changed for {generated_image_path}: "
                f"{generated_image.shape} != {source_shapes[source_name]}"
            )
        if sha256(generated_label_path) != sha256(source_label_path):
            raise AssertionError(
                f"Bounding boxes changed for {generated_label_path}"
            )
        hashes_by_source[source_name].add(sha256(generated_image_path))

    duplicate_sources = sorted(
        source_name
        for source_name, image_hashes in hashes_by_source.items()
        if len(image_hashes) != len(VARIANTS)
    )
    if duplicate_sources:
        raise AssertionError(
            "One or more augmentation variants are byte-identical: "
            + ", ".join(duplicate_sources[:5])
        )
    return {
        "images_verified": len(records),
        "all_images_readable": True,
        "dimensions_preserved": True,
        "labels_byte_identical_to_source": True,
        "variants_unique_per_source": True,
    }

def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output dataset already exists: {output}")
    if report.exists() and not args.resume:
        raise FileExistsError(f"Output report already exists: {report}")
    if not (source / "data.yaml").is_file():
        raise FileNotFoundError(f"Missing source data.yaml: {source}")
    report.mkdir(parents=True, exist_ok=args.resume)

    for split in ("train", "valid", "test"):
        copy_original_split(source, output, split)

    records: list[dict[str, Any]] = []
    box_counts = defaultdict(int)
    train_images = image_paths(source, "train")
    for image_index, image_path in enumerate(train_images, start=1):
        label_path = source / "train" / "labels" / f"{image_path.stem}.txt"
        label_rows = parse_label_rows(label_path)
        for class_id, *_ in label_rows:
            box_counts[CLASS_NAMES[class_id]] += 1
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read source image: {image_path}")

        original_metrics = quality_metrics(image)
        records.append(
            {
                "source_image": image_path.name,
                "generated_image": image_path.name,
                "variant": "original",
                "parameters": "{}",
                "boxes": len(label_rows),
                **original_metrics,
            }
        )
        for variant in VARIANTS[1:]:
            rng = stable_rng(args.seed, image_path.stem, variant)
            transformed, parameters = apply_quality_variant(image, variant, rng)
            generated_stem = f"{image_path.stem}__aug_{variant}"
            generated_image = (
                output / "train" / "images" / f"{generated_stem}.jpg"
            )
            generated_label = (
                output / "train" / "labels" / f"{generated_stem}.txt"
            )
            if not cv2.imwrite(
                str(generated_image),
                transformed,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"Could not write {generated_image}")
            shutil.copy2(label_path, generated_label)
            records.append(
                {
                    "source_image": image_path.name,
                    "generated_image": generated_image.name,
                    "variant": variant,
                    "parameters": json.dumps(parameters, separators=(",", ":")),
                    "boxes": len(label_rows),
                    **quality_metrics(transformed),
                }
            )
        if image_index % 20 == 0 or image_index == len(train_images):
            print(f"Augmented {image_index}/{len(train_images)} train images")

    data_yaml = (
        f"path: {output.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "names:\n"
        "- cell\n"
        "- droplet\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8", newline="\n")

    source_split_manifest = json.loads(
        (source / "split_manifest.json").read_text(encoding="utf-8")
    )
    split_manifest = source_split_manifest.copy()
    split_manifest["source_dataset"] = str(source)
    split_manifest["quality_augmentation"] = {
        "seed": args.seed,
        "train_only": True,
        "variants": list(VARIANTS),
        "copies_per_train_image": len(VARIANTS),
    }
    split_manifest["splits"] = {
        name: values.copy()
        for name, values in source_split_manifest["splits"].items()
    }
    split_manifest["splits"]["train"].update(
        {
            "images": len(train_images) * len(VARIANTS),
            "cell_boxes": box_counts["cell"] * len(VARIANTS),
            "droplet_boxes": box_counts["droplet"] * len(VARIANTS),
            "original_images": len(train_images),
            "augmented_images": len(train_images) * (len(VARIANTS) - 1),
        }
    )
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (report / "augmentation_manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    grouped_metrics: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in records if row["variant"] == variant]
        grouped_metrics[variant] = {
            "images": len(variant_rows),
            "brightness_mean": distribution(
                [float(row["brightness_mean"]) for row in variant_rows]
            ),
            "contrast_std": distribution(
                [float(row["contrast_std"]) for row in variant_rows]
            ),
            "sharpness_laplacian_var": distribution(
                [
                    float(row["sharpness_laplacian_var"])
                    for row in variant_rows
                ]
            ),
        }

    split_counts = {}
    unchanged_hashes = {}
    for split in ("train", "valid", "test"):
        output_images = image_paths(output, split)
        output_labels = list((output / split / "labels").glob("*.txt"))
        split_counts[split] = {
            "images": len(output_images),
            "labels": len(output_labels),
            "boxes": sum(
                len(parse_label_rows(label_path))
                for label_path in output_labels
            ),
        }
        if split in {"valid", "test"}:
            source_images = image_paths(source, split)
            source_hashes = {
                path.name: sha256(path)
                for path in source_images
            }
            output_hashes = {
                path.name: sha256(path)
                for path in output_images
            }
            source_labels = {
                path.name: sha256(path)
                for path in (source / split / "labels").glob("*.txt")
            }
            output_label_hashes = {
                path.name: sha256(path)
                for path in (output / split / "labels").glob("*.txt")
            }
            unchanged_hashes[split] = {
                "images_byte_identical": source_hashes == output_hashes,
                "labels_byte_identical": source_labels == output_label_hashes,
            }
            if not all(unchanged_hashes[split].values()):
                raise AssertionError(f"{split} changed during train-only augmentation")

    expected_train_images = len(train_images) * len(VARIANTS)
    if split_counts["train"]["images"] != expected_train_images:
        raise AssertionError(
            f"Expected {expected_train_images} train images, "
            f"got {split_counts['train']['images']}"
        )
    for split, values in split_counts.items():
        if values["images"] != values["labels"]:
            raise AssertionError(f"Image/label count mismatch in {split}: {values}")

    train_integrity = verify_train_outputs(source, output, records)
    summary = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "policy": {
            "train_only": True,
            "geometry_changed": False,
            "bounding_boxes_copied_exactly": True,
            "variants": list(VARIANTS),
        },
        "splits": split_counts,
        "train_integrity": train_integrity,
        "valid_test_integrity": unchanged_hashes,
        "quality_distributions": grouped_metrics,
    }
    (report / "quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_contact_sheet(
        source,
        output,
        report,
        args.contact_sheet_samples,
    )
    write_quality_plot(records, report)

    command = (
        "E:\\fpga\\.venv-yolo\\Scripts\\python.exe -m qnn.train_qat "
        "--data dataset/cell_droplet_roi384_quality_aug_v1 "
        "--output models/qnn_cell_droplet_v2_w4a6_square192_quality_aug_v1 "
        "--report reports/qnn_cell_droplet_v2_w4a6_square192_quality_aug_v1 "
        "--init-checkpoint "
        "models/qnn_cell_droplet_v2_w4a6_square192_grouped/best.pt "
        "--epochs 80 --batch-size 16 --learning-rate 0.0002 "
        "--weight-bits 4 --activation-bits 6 --output-bits 8 "
        "--input-width 192 --input-height 192 --downsample 4 "
        "--channels 12 16 24 24 --slots-per-class 2 1 "
        "--horizontal-shift 0.03 --patience 20 --device cuda --workers 2"
    )
    recommendation = f"""# Training recommendation

This dataset expands only `train` from 140 to 700 images. `valid` and `test`
remain byte-identical to the grouped source dataset.

Fine-tune the current W4A6 checkpoint first:

```powershell
{command}
```

Accept the new model only when validation improves without a material test
regression. Recalibrate confidence thresholds and box geometry before creating
a new FPGA manifest and bitstream.
"""
    (report / "TRAINING_RECOMMENDATION.md").write_text(
        recommendation,
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary["splits"], indent=2))
    print(f"QUALITY_AUGMENTATION_COMPLETE: {output}")
    return summary


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
