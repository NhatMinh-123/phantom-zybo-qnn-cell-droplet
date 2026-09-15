#!/usr/bin/env python3
"""Merge contiguous teacher-video segments and validate their frame indices."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    return parser.parse_args()


def merge_csv(first: Path, second: Path, output: Path, key: str) -> int:
    with first.open(newline="", encoding="ascii") as stream:
        first_rows = list(csv.DictReader(stream))
    with second.open(newline="", encoding="ascii") as stream:
        second_rows = list(csv.DictReader(stream))
    rows = first_rows + second_rows
    if not rows:
        raise RuntimeError(f"No rows in {first} or {second}")
    values = [int(row[key]) for row in rows]
    if values != sorted(values):
        raise RuntimeError(f"{output.name}: {key} is not monotonic")
    with output.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def video_frames(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    result = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return result


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    first_video = args.first.resolve() / "pc_gpu_teacher_roi256.mp4"
    second_video = args.second.resolve() / "pc_gpu_teacher_roi256.mp4"
    first_geometry = video_frames(first_video)
    second_geometry = video_frames(second_video)
    if first_geometry[1:] != second_geometry[1:]:
        raise RuntimeError(
            f"Segment geometry differs: first={first_geometry}, second={second_geometry}"
        )

    concat_path = output / "concat.txt"
    concat_path.write_text(
        f"file '{first_video.as_posix()}'\nfile '{second_video.as_posix()}'\n",
        encoding="ascii",
    )
    merged_video = output / "pc_gpu_teacher_roi256_full.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(merged_video),
        ],
        check=True,
    )
    merged_geometry = video_frames(merged_video)
    if merged_geometry[0] != args.expected_frames:
        raise RuntimeError(
            f"Merged video has {merged_geometry[0]} frames, expected {args.expected_frames}"
        )

    frame_rows = merge_csv(
        args.first / "frame_summary.csv",
        args.second / "frame_summary.csv",
        output / "frame_summary.csv",
        "frame",
    )
    with (output / "frame_summary.csv").open(newline="", encoding="ascii") as stream:
        frame_indices = [int(row["frame"]) for row in csv.DictReader(stream)]
    if frame_indices != list(range(args.expected_frames)):
        raise RuntimeError("frame_summary.csv is not the continuous source-frame range")
    detection_rows = merge_csv(
        args.first / "detections.csv",
        args.second / "detections.csv",
        output / "detections.csv",
        "frame",
    )
    manifest = {
        "runtime_label": "PC GPU reference",
        "execution_note": "This is PC GPU teacher inference, not FPGA inference.",
        "status": "complete",
        "source_frames": args.expected_frames,
        "merged_video_frames": merged_geometry[0],
        "source_fps": merged_geometry[1],
        "frame_size": [merged_geometry[2], merged_geometry[3]],
        "frame_summary_rows": frame_rows,
        "detection_rows": detection_rows,
        "roi_frame_coordinates": [560, 342, 816, 598],
        "model_input": [640, 640],
        "thresholds": {"cell": 0.55, "droplet": 0.54},
        "segments": [str(args.first.resolve()), str(args.second.resolve())],
        "tracking_note": (
            "The tracker and cumulative crossing counter restart at source frame 5024. "
            "Detection boxes and source-frame indices remain continuous."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
