#!/usr/bin/env python3
"""Keep only saved crops whose outer droplet ring is still tightly centered."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_one_droplet_frames_autocalibrated_v4 import (  # noqa: E402
    hough_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-center-error", type=float, default=7.0)
    parser.add_argument("--radius-tolerance", type=float, default=10.0)
    parser.add_argument("--max-per-video", type=int, default=300)
    return parser.parse_args()


def write_contact_sheets(
    paths: list[Path],
    output_dir: Path,
    *,
    page_size: int = 40,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = 8
    thumb = 144
    label_height = 22
    pages = 0
    for start in range(0, len(paths), page_size):
        selected = paths[start : start + page_size]
        rows = int(np.ceil(len(selected) / columns))
        sheet = np.full(
            (rows * (thumb + label_height), columns * thumb, 3),
            245,
            dtype=np.uint8,
        )
        for index, path in enumerate(selected):
            image = cv2.imread(str(path))
            image = cv2.resize(image, (thumb, thumb))
            row, column = divmod(index, columns)
            x = column * thumb
            y = row * (thumb + label_height)
            sheet[y : y + thumb, x : x + thumb] = image
            cv2.putText(
                sheet,
                path.stem[-18:],
                (x + 3, y + thumb + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        pages += 1
        cv2.imwrite(
            str(output_dir / f"contact_sheet_{pages:03d}.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
    return pages


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    output_images = output / "images_to_upload"
    output_sheets = output / "contact_sheets"
    output_images.mkdir(parents=True, exist_ok=True)

    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    with (source / "manifest.csv").open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        manifest = {row["image"]: row for row in csv.DictReader(handle)}

    kept_rows: list[dict[str, object]] = []
    video_results: list[dict[str, object]] = []
    for video in summary["videos"]:
        slug = str(video["slug"])
        expected_radius = float(video["calibrated_radius"])
        verified: list[tuple[Path, float, float]] = []
        candidates = sorted((source / "images_to_upload").glob(f"{slug}*.jpg"))
        for path in candidates:
            image = cv2.imread(str(path))
            if image is None or image.shape[0] != image.shape[1]:
                continue
            half = image.shape[0] / 2.0
            circles = hough_candidates(
                image,
                min_radius=max(
                    22,
                    int(round(expected_radius - args.radius_tolerance - 1)),
                ),
                max_radius=min(
                    int(half - 5),
                    int(round(expected_radius + args.radius_tolerance + 1)),
                ),
                accumulator_threshold=25,
            )
            circles = [
                circle
                for circle in circles
                if abs(circle[2] - expected_radius) <= args.radius_tolerance
            ]
            if not circles:
                continue
            center_x, center_y, radius = min(
                circles,
                key=lambda item: np.hypot(
                    item[0] - half,
                    item[1] - half,
                ),
            )
            center_error = float(
                np.hypot(center_x - half, center_y - half)
            )
            if center_error > args.max_center_error:
                continue
            if radius + 5.0 > half:
                continue
            verified.append((path, center_error, radius))

        if len(verified) > args.max_per_video:
            indices = np.linspace(
                0,
                len(verified) - 1,
                args.max_per_video,
            ).round().astype(int)
            verified = [verified[int(index)] for index in np.unique(indices)]

        copied_paths: list[Path] = []
        errors: list[float] = []
        for path, center_error, radius in verified:
            destination = output_images / path.name
            shutil.copy2(path, destination)
            row = dict(manifest[path.name])
            row["verified_center_error_px"] = f"{center_error:.4f}"
            row["verified_outer_radius_px"] = f"{radius:.4f}"
            row["label_status"] = "pending"
            kept_rows.append(row)
            copied_paths.append(destination)
            errors.append(center_error)

        pages = write_contact_sheets(
            copied_paths,
            output_sheets / slug,
        )
        video_results.append(
            {
                "slug": slug,
                "source_candidates": len(candidates),
                "strictly_centered_before_cap": sum(
                    1
                    for path in candidates
                    if path.name
                    in {
                        item[0].name
                        for item in verified
                    }
                ),
                "selected_images": len(copied_paths),
                "center_error_median_px": (
                    float(np.median(errors)) if errors else None
                ),
                "center_error_max_px": max(errors) if errors else None,
                "contact_sheet_pages": pages,
            }
        )
        print(
            f"{slug}: input={len(candidates)} "
            f"selected={len(copied_paths)}"
        )

    if not kept_rows:
        raise RuntimeError("No images passed the strict centering filter")
    fieldnames = list(kept_rows[0])
    for row in kept_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with (output / "manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    output_summary = {
        "source": str(source),
        "output": str(output),
        "policy": {
            "outer_ring_center_error_max_px": args.max_center_error,
            "outer_radius_tolerance_px": args.radius_tolerance,
            "max_per_video": args.max_per_video,
            "filter_applied_after_image_encoding": True,
        },
        "videos": video_results,
        "total_images": len(kept_rows),
    }
    (output / "summary.json").write_text(
        json.dumps(output_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        (
            "# Strict centered one-droplet labeling set\n\n"
            "- Upload only `images_to_upload` or the provided ZIP.\n"
            "- Roboflow project type: Object Detection.\n"
            "- Use one class: `particle`.\n"
            "- Draw tight boxes around real particles only.\n"
            "- Mark no-particle images as null/background.\n"
            "- Do not upload contact sheets.\n"
            "- Older v1-v6 folders must not be used for labeling.\n"
        ),
        encoding="utf-8",
    )
    print(f"Strict final images: {len(kept_rows)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
