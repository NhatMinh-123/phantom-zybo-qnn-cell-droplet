#!/usr/bin/env python3
"""High-precision offline cropper for the Roboflow labeling dataset.

This wrapper replaces the realtime background locator with a stricter Hough
circle locator. Speed is not important here; incomplete or uncertain droplets
are skipped instead of being estimated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.extract_one_droplet_frames_for_labeling as base  # noqa: E402
from scripts.run_microplastic_one_droplet_hybrid import (  # noqa: E402
    DropletObservation,
    DropletSequenceTracker,
    Rect,
    centered_crop,
    crop_rect,
)


def find_precise_circle(
    roi: np.ndarray,
    *,
    channel_center_x: float,
    crop_size: int,
) -> tuple[float, float, float] | None:
    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(grayscale)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=55,
        param1=45,
        param2=30,
        minRadius=28,
        maxRadius=74,
    )
    if circles is None:
        return None

    half = crop_size / 2.0
    height, width = grayscale.shape
    candidates: list[tuple[float, float, float, float]] = []
    for center_x, center_y, radius in circles[0]:
        center_x = float(center_x)
        center_y = float(center_y)
        radius = float(radius)
        if abs(center_x - channel_center_x) > 12.0:
            continue
        if center_x < half or center_x >= width - half:
            continue
        if center_y < half or center_y >= height - half:
            continue
        if radius + 5.0 > half:
            continue

        ring_mask = np.zeros_like(grayscale)
        cv2.circle(
            ring_mask,
            (int(round(center_x)), int(round(center_y))),
            int(round(radius)),
            255,
            3,
        )
        ring_values = cv2.Laplacian(
            blurred,
            cv2.CV_32F,
            ksize=3,
        )[ring_mask > 0]
        ring_strength = float(np.mean(np.abs(ring_values)))
        center_penalty = abs(center_x - channel_center_x)
        score = ring_strength + 0.10 * radius - 0.25 * center_penalty
        candidates.append((score, center_x, center_y, radius))

    if not candidates:
        return None
    _, center_x, center_y, radius = max(candidates)
    return center_x, center_y, radius


def extract_video_hough(
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
) -> tuple[list[base.CropRecord], dict[str, object]]:
    del (
        background_samples,
        background_threshold,
        droplet_min_area,
        droplet_max_area,
        droplet_radius,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    acquisition_roi.validate(width, height)

    tracker = DropletSequenceTracker(
        new_droplet_jump=new_droplet_jump,
        max_missing=droplet_max_missing,
    )
    records: list[base.CropRecord] = []
    hough_frames = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        roi_bgr = crop_rect(frame, acquisition_roi)
        circle = find_precise_circle(
            roi_bgr,
            channel_center_x=channel_center_x,
            crop_size=crop_size,
        )

        observation = None
        if circle is not None:
            center_x, center_y, radius = circle
            observation = DropletObservation(
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                area=float(np.pi * radius * radius),
                score=float(radius),
                bbox=(
                    int(round(center_x - radius)),
                    int(round(center_y - radius)),
                    int(round(2 * radius)),
                    int(round(2 * radius)),
                ),
            )
            hough_frames += 1

        observation, _ = tracker.update(observation, frame_index)
        if observation is not None and circle is not None:
            crop = centered_crop(
                roi_bgr,
                observation.center_x,
                observation.center_y,
                crop_size,
            )
            records.append(
                base.CropRecord(
                    frame_index=frame_index,
                    timestamp_sec=(frame_index / fps if fps > 0 else 0.0),
                    sequence_id=tracker.sequence_id,
                    center_x=observation.center_x,
                    center_y=observation.center_y,
                    radius=observation.radius,
                    score=observation.score,
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
        "detected_frames": hough_frames,
        "complete_droplet_frames": len(records),
        "droplet_sequences": tracker.sequence_id,
        "locator": "strict_hough_circle",
    }
    return records, metadata


def main() -> None:
    base.extract_video = extract_video_hough
    base.main()


if __name__ == "__main__":
    main()
