#!/usr/bin/env python3
"""Audit the flat 15 um labeling package before manual annotation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"E:\fpga\roboflow_upload\cell_droplet_15um_selected9_label_all_900_v1")
IMAGES = ROOT / "images_to_label"
REFERENCES = ROOT / "native_reference_do_not_upload"
MANIFEST = ROOT / "manifest.csv"
OUT = ROOT / "prelabel_audit"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeffs = cv2.dct(small)[:8, :8]
    median = np.median(coeffs[1:, :])
    bits = (coeffs > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_fraction": float(np.mean(gray < 6)),
        "bright_fraction": float(np.mean(gray > 249)),
    }


def percentile(values: list[float], marks: tuple[int, ...] = (1, 5, 50, 95, 99)) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {f"p{mark}": round(float(np.percentile(array, mark)), 4) for mark in marks}


def write_montage(rows: list[dict[str, str]]) -> Path:
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["source_video"]].append(row)
    chosen: list[dict[str, str]] = []
    for video in sorted(by_video):
        # Native references preserve the actual camera geometry; choose clearest source group.
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in by_video[video]:
            groups[row["group_id"]].append(row)
        best = max(groups.values(), key=lambda group: float(group[0]["native_sharpness"]))
        chosen.extend(sorted(best, key=lambda row: int(row["tile_index"])))
    side, header, cols = 185, 28, 5
    sheet = np.full((len(by_video) * (side + header), cols * side, 3), 246, dtype=np.uint8)
    for index, row in enumerate(chosen):
        block, col = divmod(index, cols)
        y, x = block * (side + header), col * side
        image = cv2.imread(row["native_reference_path"])
        sheet[y:y + side, x:x + side] = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
        label = f"{row['source_video']} tile{row['tile_index']}" if col == 0 else f"tile{row['tile_index']}"
        cv2.putText(sheet, label, (x + 3, y + side + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (15, 15, 15), 1, cv2.LINE_AA)
    path = OUT / "representative_native_roi_montage.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    errors, all_hashes, phashes = [], [], []
    numeric: dict[str, list[float]] = defaultdict(list)
    profile_counts = Counter()
    per_video_tile: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for row in rows:
        path = Path(row["upload_path"])
        reference = Path(row["native_reference_path"])
        image = cv2.imread(str(path))
        if image is None:
            errors.append(f"unreadable:{path.name}")
            continue
        if image.shape[:2] != (640, 640):
            errors.append(f"wrong_shape:{path.name}:{image.shape[:2]}")
        if not reference.is_file():
            errors.append(f"missing_reference:{path.name}")
        file_hash = hash_file(path)
        all_hashes.append(file_hash)
        image_metrics = metrics(image)
        for name, value in image_metrics.items():
            numeric[name].append(value)
        profile_counts[row["quality_profile"]] += 1
        p_hash = phash(image)
        phashes.append((p_hash, row["file"]))
        per_video_tile[(row["source_video"], row["tile_index"])].append((int(row["source_frame"]), p_hash, row["file"]))

    near_duplicate_pairs = []
    for key, entries in per_video_tile.items():
        entries.sort()
        for left, right in zip(entries, entries[1:]):
            distance = hamming(left[1], right[1])
            if distance <= 2:
                near_duplicate_pairs.append({"video": key[0], "tile": key[1], "left": left[2], "right": right[2], "hamming": distance})
    montage = write_montage(rows)
    report = {
        "images_expected": 900,
        "images_manifest": len(rows),
        "images_readable": len(rows) - len([item for item in errors if item.startswith("unreadable:")]),
        "dimensions_required": [640, 640],
        "errors": errors,
        "exact_duplicate_hashes": len(all_hashes) - len(set(all_hashes)),
        "adjacent_same_video_tile_near_duplicate_pairs_phash_le_2": len(near_duplicate_pairs),
        "near_duplicate_examples": near_duplicate_pairs[:30],
        "quality_profile_counts": dict(profile_counts),
        "image_metric_percentiles": {name: percentile(values) for name, values in numeric.items()},
        "roi_geometry": {"x_positions": [170, 365, 560, 755, 950], "y": 342, "size": 256, "annotation_size": 640},
        "labeling_decision": "READY only when errors=0 and exact_duplicate_hashes=0. Near duplicates are review candidates, not automatic removals.",
        "representative_native_roi_montage": str(montage),
    }
    (OUT / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Pre-label audit", "", f"- Images: {len(rows)}", f"- Read errors: {len(errors)}", f"- Exact duplicates: {report['exact_duplicate_hashes']}", f"- Near-duplicate candidates: {len(near_duplicate_pairs)}", f"- Montage: `{montage.name}`", "", "The ROI includes full droplets and a small margin around their border. It does not use a tiny particle-only crop; therefore the final detector can learn droplet context and particles near the wall."]
    (OUT / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
