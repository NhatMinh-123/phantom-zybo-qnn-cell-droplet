#!/usr/bin/env python3
"""Preview the fixed downstream-channel ROIs used by the detector."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 800


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the realtime ROI and model input tiles.")
    parser.add_argument(
        "--source",
        required=True,
        help="Camera index such as 0, or a video file path.",
    )
    parser.add_argument("--tile-xs", type=parse_int_list, default=[170, 365, 560, 755, 950])
    parser.add_argument("--tile-y", type=int, default=342)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--model-input-size", type=int, default=640)
    parser.add_argument("--display-width", type=int, default=1200)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=800)
    parser.add_argument("--loop", action="store_true", help="Loop when a video file reaches the end.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 is unlimited.")
    parser.add_argument("--headless", action="store_true", help="Do not open an OpenCV window.")
    parser.add_argument("--preview-output", type=Path, default=None)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("reports/realtime_roi_snapshots"),
    )
    return parser.parse_args()


def resolve_source(value: str) -> int | str:
    path = Path(value)
    if value.isdigit() and not path.exists():
        return int(value)
    return str(path.resolve())


def scaled_geometry(
    width: int,
    height: int,
    tile_xs: list[int],
    tile_y: int,
    tile_size: int,
) -> list[tuple[int, int, int, int]]:
    scale_x = width / REFERENCE_WIDTH
    scale_y = height / REFERENCE_HEIGHT
    boxes = []
    for x in tile_xs:
        x1 = int(round(x * scale_x))
        y1 = int(round(tile_y * scale_y))
        x2 = int(round((x + tile_size) * scale_x))
        y2 = int(round((tile_y + tile_size) * scale_y))
        boxes.append((x1, y1, x2, y2))
    return boxes


def extract_tiles(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    output_size: int,
) -> list[np.ndarray]:
    tiles = []
    for x1, y1, x2, y2 in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise RuntimeError(f"Empty ROI at {(x1, y1, x2, y2)}")
        tiles.append(
            cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        )
    return tiles


def compose_monitor(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    tiles: list[np.ndarray],
    display_width: int,
    fps: float,
) -> np.ndarray:
    annotated = frame.copy()
    colors = [(0, 80, 255), (0, 180, 255), (0, 220, 80), (255, 170, 0), (220, 70, 220)]
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        color = colors[index % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"ROI {index + 1}",
            (x1 + 5, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    frame_height, frame_width = frame.shape[:2]
    full_height = int(round(frame_height * display_width / frame_width))
    full_view = cv2.resize(annotated, (display_width, full_height), interpolation=cv2.INTER_AREA)

    tile_width = display_width // len(tiles)
    tile_strip = np.zeros((tile_width + 28, display_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        color = colors[index % len(colors)]
        thumbnail = cv2.resize(tile, (tile_width, tile_width), interpolation=cv2.INTER_AREA)
        x = index * tile_width
        tile_strip[:tile_width, x : x + tile_width] = thumbnail
        cv2.rectangle(tile_strip, (x, 0), (x + tile_width - 1, tile_width - 1), color, 2)
        cv2.putText(
            tile_strip,
            f"MODEL INPUT {index + 1}",
            (x + 6, tile_width + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    header = np.zeros((36, display_width, 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"Realtime ROI Monitor | {frame_width}x{frame_height} | {fps:.1f} FPS | Q: quit  S: snapshot",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, full_view, tile_strip])


def save_snapshot(
    directory: Path,
    frame: np.ndarray,
    monitor: np.ndarray,
    tiles: list[np.ndarray],
) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = directory / timestamp
    target.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target / "frame.jpg"), frame)
    cv2.imwrite(str(target / "monitor.jpg"), monitor)
    for index, tile in enumerate(tiles, start=1):
        cv2.imwrite(str(target / f"roi_{index:02d}_640.jpg"), tile)
    print(f"Snapshot: {target.resolve()}")


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)
    capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    processed = 0
    fps_ema = 0.0
    previous_time = time.perf_counter()
    preview_written = False
    while True:
        ok, frame = capture.read()
        if not ok:
            if args.loop and not isinstance(source, int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        height, width = frame.shape[:2]
        boxes = scaled_geometry(width, height, args.tile_xs, args.tile_y, args.tile_size)
        tiles = extract_tiles(frame, boxes, args.model_input_size)

        now = time.perf_counter()
        instant_fps = 1.0 / max(1e-6, now - previous_time)
        previous_time = now
        fps_ema = instant_fps if processed == 0 else 0.9 * fps_ema + 0.1 * instant_fps
        monitor = compose_monitor(frame, boxes, tiles, args.display_width, fps_ema)
        processed += 1

        if args.preview_output is not None and not preview_written:
            args.preview_output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.preview_output), monitor)
            preview_written = True

        if not args.headless:
            cv2.imshow("Realtime ROI Monitor", monitor)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_snapshot(args.snapshot_dir, frame, monitor, tiles)

        if args.max_frames > 0 and processed >= args.max_frames:
            break

    capture.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"Processed frames: {processed}")
    print("Model reads the five 640x640 images shown in the bottom strip.")


if __name__ == "__main__":
    main()
