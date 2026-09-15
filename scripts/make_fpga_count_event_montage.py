#!/usr/bin/env python3
"""Create an annotated ROI-B montage at every recorded crossing event."""

from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--padding", type=int, default=35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
    with (result_dir / "count_events.csv").open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    if not events:
        raise RuntimeError("No count events are available for a montage")
    roi = report["rois"]["verification"]
    x1 = max(0, int(roi["x"]) - args.padding)
    y1 = max(0, int(roi["y"]) - args.padding)
    x2 = int(roi["x"] + roi["width"]) + args.padding
    y2 = int(roi["y"] + roi["height"]) + args.padding
    capture = cv2.VideoCapture(str(report["output_video"]))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {report['output_video']}")

    tile_width = x2 - x1
    tile_height = y2 - y1 + 27
    tiles: list[np.ndarray] = []
    for event in events:
        frame_index = int(event["frame_index"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        crop = frame[y1:y2, x1:x2]
        tile = np.full((tile_height, tile_width, 3), 18, dtype=np.uint8)
        tile[27:, :] = crop
        verified = event["roi_a_verified"].lower() == "true"
        label = (
            f"#{event['event_index']} f{frame_index} {event['class_name']} "
            f"ID{event['track_id']} {'A-B' if verified else 'B-only'}"
        )
        cv2.putText(
            tile,
            label,
            (4, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (70, 225, 90) if verified else (40, 185, 245),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    capture.release()
    if not tiles:
        raise RuntimeError("Could not read any event frames")

    columns = max(1, min(args.columns, len(tiles)))
    rows = ceil(len(tiles) / columns)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 10, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    output = result_dir / "count_event_montage.jpg"
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    preview = canvas.copy()
    if preview.shape[1] > 900:
        scale = 900 / preview.shape[1]
        preview = cv2.resize(
            preview,
            (900, int(round(preview.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(
        str(result_dir / "count_event_montage_preview.jpg"),
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, 55],
    )
    print(output)


if __name__ == "__main__":
    main()
