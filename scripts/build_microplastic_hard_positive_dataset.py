"""Merge localized missed-particle patches into the reviewed QNN dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="CSV exported by the missed-particle localization application.",
    )
    parser.add_argument(
        "--localization-package",
        type=Path,
        default=ROOT / "review_packages" / "microplastic_miss_localization_v1",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=ROOT / "dataset" / "microplastic_patch32_reviewed_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "microplastic_patch32_reviewed_v3_hard_positive",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def centered_crop(image: np.ndarray, x: int, y: int, size: int = 32) -> np.ndarray:
    half = size // 2
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT_101)
    crop = padded[y : y + size, x : x + size]
    if crop.shape != (size, size):
        raise RuntimeError(f"Expected {size}x{size} crop, got {crop.shape}")
    return crop


def sequence_to_split(base_summary: dict[str, object]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    counts = base_summary["counts"]
    assert isinstance(counts, dict)
    for split in SPLITS:
        values = counts[split]
        assert isinstance(values, dict)
        for sequence in values["sequence_ids"]:
            numeric = int(sequence)
            if numeric in mapping:
                raise RuntimeError(f"Sequence {numeric} appears in more than one split")
            mapping[numeric] = split
    return mapping


def main() -> None:
    args = parse_args()
    labels = args.labels.expanduser().resolve()
    package = args.localization_package.expanduser().resolve()
    base = args.base_dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")

    base_summary = json.loads((base / "dataset_summary.json").read_text(encoding="utf-8"))
    split_by_sequence = sequence_to_split(base_summary)
    assets_by_id = {
        row["review_id"]: row
        for row in read_csv(package / "localization_manifest.csv")
    }
    selected = [
        row for row in read_csv(labels)
        if row.get("review_status") == "reviewed"
        and row.get("reviewed_label") == "particle"
        and row.get("points_json") not in (None, "", "[]")
    ]
    if not selected:
        raise RuntimeError("No confirmed particle points in localization CSV")

    for split in SPLITS:
        for label in ("background", "particle"):
            source = base / split / label
            destination = output / split / label
            destination.mkdir(parents=True, exist_ok=True)
            for image in source.glob("*.png"):
                shutil.copy2(image, destination / image.name)

    additions: list[dict[str, object]] = []
    for row in selected:
        review_id = row["review_id"]
        asset_row = assets_by_id.get(review_id)
        if asset_row is None:
            raise KeyError(f"Missing localization manifest item: {review_id}")
        sequence = int(row["droplet_sequence"])
        split = split_by_sequence.get(sequence)
        if split is None:
            raise KeyError(f"Sequence {sequence} is absent from base grouped split")
        center = cv2.imread(str(package / asset_row["center_asset"]), cv2.IMREAD_GRAYSCALE)
        if center is None:
            raise RuntimeError(f"Could not read center crop for {review_id}")
        expected_size = int(row["processing_size"])
        if center.shape != (expected_size, expected_size):
            raise RuntimeError(f"Unexpected center crop shape for {review_id}: {center.shape}")
        points = json.loads(row["points_json"])
        for point_index, point in enumerate(points, start=1):
            x, y = int(point["x"]), int(point["y"])
            if not (0 <= x < expected_size and 0 <= y < expected_size):
                raise ValueError(f"Point outside ROI for {review_id}: {(x, y)}")
            patch = centered_crop(center, x, y)
            name = f"{review_id}_p{point_index:02d}_seq{sequence:04d}_frame{int(row['best_frame']):05d}.png"
            destination = output / split / "particle" / name
            if destination.exists():
                raise FileExistsError(destination)
            cv2.imwrite(str(destination), patch)
            additions.append(
                {
                    "review_id": review_id,
                    "source_review_id": row["source_review_id"],
                    "split": split,
                    "droplet_sequence": sequence,
                    "frame": int(row["best_frame"]),
                    "patch_x": x,
                    "patch_y": y,
                    "dataset_path": destination.relative_to(output).as_posix(),
                }
            )

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {
            label: len(list((output / split / label).glob("*.png")))
            for label in ("background", "particle")
        }
        counts[split]["total"] = counts[split]["background"] + counts[split]["particle"]
    with (output / "hard_positive_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(additions[0]))
        writer.writeheader()
        writer.writerows(additions)
    shutil.copy2(labels, output / "missed_particle_locations_source.csv")
    summary = {
        "name": "microplastic_patch32_reviewed_v3_hard_positive",
        "purpose": "Reviewed candidate patches plus manually localized missed particles",
        "base_dataset": str(base),
        "base_dataset_summary": str(base / "dataset_summary.json"),
        "locations_csv": str(labels),
        "locations_csv_sha256": sha256(labels),
        "localization_package": str(package),
        "patch_size": 32,
        "hard_positive_patches": len(additions),
        "hard_positive_frames": len(selected),
        "counts": counts,
        "group_key": "droplet_sequence",
        "split_policy": "inherited from reviewed_v2 sequence split; no sequence crosses split",
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
