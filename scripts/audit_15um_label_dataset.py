#!/usr/bin/env python3
"""Audit the prepared 15 um Roboflow labeling package."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def average(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Manifest is empty")

    group_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    profile_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    dimensions: set[tuple[int, int]] = set()
    missing: list[str] = []
    for row in rows:
        upload_path = Path(row["upload_path"])
        native_path = Path(row["native_reference_path"])
        if not upload_path.exists() or not native_path.exists():
            missing.append(row["file"])
            continue
        image = cv2.imread(str(upload_path))
        if image is None:
            missing.append(row["file"])
            continue
        dimensions.add((image.shape[1], image.shape[0]))
        group_splits[row["group_id"]].add(row["split"])
        video_splits[row["source_video"]].add(row["split"])
        profile_rows[row["quality_profile"]].append(row)

    leaked_groups = {key: sorted(value) for key, value in group_splits.items() if len(value) != 1}
    leaked_videos = {key: sorted(value) for key, value in video_splits.items() if len(value) != 1}
    profile_statistics: dict[str, dict[str, float | int]] = {}
    for profile, items in sorted(profile_rows.items()):
        brightness_ratios = [
            float(row["upload_brightness"]) / max(float(row["native_brightness"]), 1e-6)
            for row in items
        ]
        sharpness_ratios = [
            float(row["upload_sharpness"]) / max(float(row["native_sharpness"]), 1e-6)
            for row in items
        ]
        profile_statistics[profile] = {
            "images": len(items),
            "mean_brightness_ratio": average(brightness_ratios),
            "minimum_brightness_ratio": min(brightness_ratios),
            "maximum_brightness_ratio": max(brightness_ratios),
            "mean_sharpness_ratio": average(sharpness_ratios),
            "minimum_sharpness_ratio": min(sharpness_ratios),
            "maximum_sharpness_ratio": max(sharpness_ratios),
        }

    profile_names = sorted(profile_rows)
    tile = 320
    caption = 34
    sheet = np.full((len(profile_names) * (tile + caption), tile * 2, 3), 245, dtype=np.uint8)
    for index, profile in enumerate(profile_names):
        row = profile_rows[profile][len(profile_rows[profile]) // 2]
        native = cv2.imread(row["native_reference_path"])
        upload = cv2.imread(row["upload_path"])
        native = cv2.resize(native, (tile, tile), interpolation=cv2.INTER_AREA)
        upload = cv2.resize(upload, (tile, tile), interpolation=cv2.INTER_AREA)
        y = index * (tile + caption)
        sheet[y : y + tile, :tile] = native
        sheet[y : y + tile, tile:] = upload
        text = f"{profile}: native (left) | upload (right)"
        cv2.putText(
            sheet,
            text,
            (8, y + tile + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(
        str(output / "quality_profile_native_vs_upload.jpg"),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    failures: list[str] = []
    if missing:
        failures.append(f"missing_or_unreadable={len(missing)}")
    if dimensions != {(640, 640)}:
        failures.append(f"unexpected_dimensions={sorted(dimensions)}")
    if leaked_groups:
        failures.append(f"group_leakage={len(leaked_groups)}")
    if leaked_videos:
        failures.append(f"video_leakage={len(leaked_videos)}")
    if any(row["split"] != "train" and row["quality_profile"] != "native" for row in rows):
        failures.append("synthetic_variant_outside_train")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "images": len(rows),
        "dimensions": [list(value) for value in sorted(dimensions)],
        "source_groups": len(group_splits),
        "source_videos": len(video_splits),
        "missing_or_unreadable": missing,
        "group_leakage": leaked_groups,
        "video_leakage": leaked_videos,
        "profile_statistics": profile_statistics,
        "failures": failures,
    }
    (output / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
