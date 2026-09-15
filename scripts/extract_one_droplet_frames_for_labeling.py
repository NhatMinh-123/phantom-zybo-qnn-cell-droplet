#!/usr/bin/env python3
"""Extract diverse one-droplet crops from a directory of videos for labeling."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_microplastic_one_droplet_hybrid import (  # noqa: E402
    DropletSequenceTracker,
    Rect,
    build_background,
    centered_crop,
    crop_rect,
    find_droplet,
)


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass
class CropRecord:
    frame_index: int
    timestamp_sec: float
    sequence_id: int
    center_x: float
    center_y: float
    radius: float
    score: float
    bright_score: float
    crop: np.ndarray


def bright_compact_score(crop: np.ndarray, droplet_radius: float) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    response = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    core = np.zeros_like(gray)
    center = (gray.shape[1] // 2, gray.shape[0] // 2)
    radius = max(
        4,
        int(round(min(droplet_radius * 0.58, min(center) - 2))),
    )
    cv2.circle(core, center, radius, 255, -1)
    values = response[core > 0]
    return float(np.percentile(values, 99.5)) if values.size else 0.0


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-x", type=int, default=621)
    parser.add_argument("--roi-y", type=int, default=40)
    parser.add_argument("--roi-width", type=int, default=180)
    parser.add_argument("--roi-height", type=int, default=260)
    parser.add_argument("--channel-center-x", type=float, default=90.0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--background-samples", type=int, default=80)
    parser.add_argument("--background-threshold", type=int, default=11)
    parser.add_argument("--droplet-min-area", type=float, default=350.0)
    parser.add_argument("--droplet-max-area", type=float, default=30000.0)
    parser.add_argument("--droplet-radius", type=float, default=64.0)
    parser.add_argument("--new-droplet-jump", type=float, default=55.0)
    parser.add_argument("--droplet-max-missing", type=int, default=3)
    parser.add_argument("--samples-per-sequence", type=int, default=2)
    parser.add_argument("--max-per-video", type=int, default=220)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    return parser.parse_args()


def extract_video(
    video_path: Path,
    *,
    acquisition_roi: Rect,
    crop_size: int,
    channel_center_x: float,
    background_samples: int,
    background_threshold: int,
    droplet_min_area: float,
    droplet_max_area: float,
    droplet_radius: float,
    new_droplet_jump: float,
    droplet_max_missing: int,
) -> tuple[list[CropRecord], dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    acquisition_roi.validate(width, height)

    background = build_background(
        capture,
        acquisition_roi,
        total_frames=frame_count,
        sample_count=min(background_samples, max(frame_count, 3)),
    )
    tracker = DropletSequenceTracker(
        new_droplet_jump=new_droplet_jump,
        max_missing=droplet_max_missing,
    )

    records: list[CropRecord] = []
    detected_frames = 0
    complete_frames = 0
    frame_index = 0
    half = crop_size / 2.0

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        roi_bgr = crop_rect(frame, acquisition_roi)
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        observation, _, _ = find_droplet(
            gray_roi,
            background,
            threshold=background_threshold,
            min_area=droplet_min_area,
            max_area=droplet_max_area,
            channel_center_x=channel_center_x,
            expected_radius=droplet_radius,
        )
        observation, _ = tracker.update(observation, frame_index)

        if observation is not None:
            detected_frames += 1
            complete = (
                observation.center_y >= half
                and observation.center_y < acquisition_roi.height - half
            )
            if complete:
                complete_frames += 1
                crop = centered_crop(
                    roi_bgr,
                    observation.center_x,
                    observation.center_y,
                    crop_size,
                )
                records.append(
                    CropRecord(
                        frame_index=frame_index,
                        timestamp_sec=(frame_index / fps if fps > 0 else 0.0),
                        sequence_id=tracker.sequence_id,
                        center_x=observation.center_x,
                        center_y=observation.center_y,
                        radius=observation.radius,
                        score=observation.score,
                        bright_score=bright_compact_score(
                            crop,
                            droplet_radius,
                        ),
                        crop=crop,
                    )
                )
        frame_index += 1

    capture.release()
    metadata = {
        "video": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_count,
        "detected_frames": detected_frames,
        "complete_droplet_frames": complete_frames,
        "droplet_sequences": tracker.sequence_id,
    }
    return records, metadata


def choose_diverse_records(
    records: list[CropRecord],
    *,
    samples_per_sequence: int,
    max_per_video: int,
) -> list[CropRecord]:
    grouped: dict[int, list[CropRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sequence_id].append(record)

    selected: list[CropRecord] = []
    for sequence_id in sorted(grouped):
        sequence = grouped[sequence_id]
        count = min(samples_per_sequence, len(sequence))
        priority = [
            int(np.argmax([item.bright_score for item in sequence])),
            int(np.argmin([item.bright_score for item in sequence])),
        ]
        priority.extend(
            np.linspace(0, len(sequence) - 1, count).round().astype(int)
        )
        indices = list(dict.fromkeys(int(index) for index in priority))[:count]
        selected.extend(sequence[index] for index in indices)

    selected.sort(key=lambda item: item.frame_index)
    if len(selected) > max_per_video:
        indices = np.linspace(0, len(selected) - 1, max_per_video)
        selected = [selected[int(round(index))] for index in indices]
    return selected


def write_contact_sheets(
    records: list[tuple[Path, CropRecord]],
    output_dir: Path,
    *,
    page_size: int = 40,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = 8
    thumb = 144
    label_height = 24
    pages = 0
    for page_start in range(0, len(records), page_size):
        page_records = records[page_start : page_start + page_size]
        rows = int(np.ceil(len(page_records) / columns))
        sheet = np.full(
            (rows * (thumb + label_height), columns * thumb, 3),
            245,
            dtype=np.uint8,
        )
        for index, (path, record) in enumerate(page_records):
            row, column = divmod(index, columns)
            image = cv2.resize(record.crop, (thumb, thumb))
            y = row * (thumb + label_height)
            x = column * thumb
            sheet[y : y + thumb, x : x + thumb] = image
            cv2.putText(
                sheet,
                path.stem[-20:],
                (x + 3, y + thumb + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        pages += 1
        cv2.imwrite(
            str(output_dir / f"contact_sheet_{pages:03d}.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    return pages


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output = args.output.resolve()
    images_dir = output / "images_to_upload"
    sheets_dir = output / "contact_sheets"
    images_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise RuntimeError(f"No videos found in {source_dir}")

    acquisition_roi = Rect(
        args.roi_x,
        args.roi_y,
        args.roi_width,
        args.roi_height,
    )
    manifest_rows: list[dict[str, object]] = []
    all_selected: list[tuple[Path, CropRecord]] = []
    video_summaries: list[dict[str, object]] = []

    for video_index, video_path in enumerate(videos, start=1):
        records, metadata = extract_video(
            video_path,
            acquisition_roi=acquisition_roi,
            crop_size=args.crop_size,
            channel_center_x=args.channel_center_x,
            background_samples=args.background_samples,
            background_threshold=args.background_threshold,
            droplet_min_area=args.droplet_min_area,
            droplet_max_area=args.droplet_max_area,
            droplet_radius=args.droplet_radius,
            new_droplet_jump=args.new_droplet_jump,
            droplet_max_missing=args.droplet_max_missing,
        )
        selected = choose_diverse_records(
            records,
            samples_per_sequence=args.samples_per_sequence,
            max_per_video=args.max_per_video,
        )
        slug = f"v{video_index:02d}_{slugify(video_path.stem)}"
        written_for_video: list[tuple[Path, CropRecord]] = []
        for record in selected:
            filename = (
                f"{slug}_seq{record.sequence_id:04d}"
                f"_frame{record.frame_index:06d}.jpg"
            )
            image_path = images_dir / filename
            ok = cv2.imwrite(
                str(image_path),
                record.crop,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not ok:
                raise RuntimeError(f"Could not write {image_path}")
            split_group = f"{slug}_seq{record.sequence_id:04d}"
            manifest_rows.append(
                {
                    "image": filename,
                    "source_video": video_path.name,
                    "frame_index": record.frame_index,
                    "timestamp_sec": f"{record.timestamp_sec:.6f}",
                    "droplet_sequence": record.sequence_id,
                    "split_group": split_group,
                    "center_x_in_gate": f"{record.center_x:.3f}",
                    "center_y_in_gate": f"{record.center_y:.3f}",
                    "detected_radius": f"{record.radius:.3f}",
                    "localization_score": f"{record.score:.6f}",
                    "bright_compact_score": f"{record.bright_score:.3f}",
                    "label_status": "pending",
                }
            )
            written_for_video.append((image_path, record))
            all_selected.append((image_path, record))

        pages = write_contact_sheets(
            written_for_video,
            sheets_dir / slug,
        )
        metadata.update(
            {
                "slug": slug,
                "selected_images": len(selected),
                "contact_sheet_pages": pages,
            }
        )
        video_summaries.append(metadata)
        print(
            f"{video_path.name}: complete={metadata['complete_droplet_frames']} "
            f"sequences={metadata['droplet_sequences']} "
            f"selected={len(selected)}"
        )

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source_directory": str(source_dir),
        "output_directory": str(output),
        "configuration": {
            "acquisition_roi": {
                "x": acquisition_roi.x,
                "y": acquisition_roi.y,
                "width": acquisition_roi.width,
                "height": acquisition_roi.height,
            },
            "crop_size": args.crop_size,
            "samples_per_sequence": args.samples_per_sequence,
            "max_per_video": args.max_per_video,
        },
        "videos": video_summaries,
        "total_selected_images": len(all_selected),
        "upload_directory": str(images_dir),
        "manifest": str(manifest_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        (
            "# Roboflow one-droplet labeling set\n\n"
            f"- Upload only the `{images_dir.name}` directory.\n"
            "- Project type: Object Detection.\n"
            "- Class: `particle`.\n"
            "- Draw a tight box around each real particle inside the droplet.\n"
            "- Keep frames with no particle and mark them as null/background.\n"
            "- Do not label the droplet boundary, glare, dust, or channel wall.\n"
            "- Keep every `split_group` in only one of train/valid/test.\n"
            "- Contact sheets are for review only; do not upload them.\n"
        ),
        encoding="utf-8",
    )
    print(f"Total selected images: {len(all_selected)}")
    print(f"Upload directory: {images_dir}")


if __name__ == "__main__":
    main()
