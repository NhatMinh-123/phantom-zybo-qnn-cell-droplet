#!/usr/bin/env python3
"""Survey 15 um microscopy videos against the legacy five-tile ROI geometry."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


TILE_XS = (170, 365, 560, 755, 950)
TILE_Y = 342
TILE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quality_metrics(frame: np.ndarray) -> dict[str, float]:
    x1 = min(TILE_XS)
    x2 = max(TILE_XS) + TILE_SIZE
    roi = frame[TILE_Y : TILE_Y + TILE_SIZE, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "clipped_fraction": float(np.mean((gray <= 4) | (gray >= 251))),
    }


def annotate(frame: np.ndarray, title: str) -> np.ndarray:
    result = frame.copy()
    for index, tile_x in enumerate(TILE_XS, start=1):
        cv2.rectangle(
            result,
            (tile_x, TILE_Y),
            (tile_x + TILE_SIZE, TILE_Y + TILE_SIZE),
            (20, 220, 20),
            2,
        )
        cv2.putText(
            result,
            str(index),
            (tile_x + 6, TILE_Y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 230),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (15, 18, 22), -1)
    cv2.putText(
        result,
        title,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return cv2.resize(result, (480, 300), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panels: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    fractions = (0.2, 0.5, 0.8)
    for video_path in sorted(args.videos.resolve().glob("*.mp4")):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        for fraction in fractions:
            frame_index = min(total_frames - 1, max(0, round((total_frames - 1) * fraction)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index + 1} from {video_path}")
            metrics = quality_metrics(frame)
            title = f"{video_path.name}  frame={frame_index + 1}  t={frame_index / fps:.1f}s"
            panels.append(annotate(frame, title))
            rows.append(
                {
                    "video": video_path.name,
                    "frame": frame_index + 1,
                    "time_s": frame_index / fps,
                    **metrics,
                }
            )
        capture.release()

    columns = 3
    panel_height, panel_width = panels[0].shape[:2]
    sheet_rows = math.ceil(len(panels) / columns)
    sheet = np.full((sheet_rows * panel_height, columns * panel_width, 3), 18, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet[
            row * panel_height : (row + 1) * panel_height,
            column * panel_width : (column + 1) * panel_width,
        ] = panel
    cv2.imwrite(str(output / "legacy_roi_geometry_contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with (output / "video_quality_survey.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"videos={len(rows) // len(fractions)} panels={len(panels)}")
    print(output / "legacy_roi_geometry_contact_sheet.jpg")


if __name__ == "__main__":
    main()
