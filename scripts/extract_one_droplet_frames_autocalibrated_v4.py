#!/usr/bin/env python3
"""Autocalibrated, double-validated one-droplet dataset extractor."""

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


def hough_candidates(
    image: np.ndarray,
    *,
    min_radius: int,
    max_radius: int,
    accumulator_threshold: float,
) -> list[tuple[float, float, float]]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=45,
        param1=40,
        param2=accumulator_threshold,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    return [
        (float(center_x), float(center_y), float(radius))
        for center_x, center_y, radius in circles[0]
    ]


def calibrate_video(
    capture: cv2.VideoCapture,
    *,
    frame_count: int,
    search_roi: Rect,
) -> tuple[float, float, int]:
    center_x_values: list[float] = []
    radius_values: list[float] = []
    positions = np.linspace(0, max(frame_count - 1, 0), min(48, frame_count))
    for frame_index in positions.round().astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        roi = crop_rect(frame, search_roi)
        candidates = hough_candidates(
            roi,
            min_radius=22,
            max_radius=90,
            accumulator_threshold=23,
        )
        if not candidates:
            continue
        # In each sampled frame, prefer the strongest physical droplet scale:
        # large channel-wall arcs are rejected by max radius.
        center_x, _, radius = max(candidates, key=lambda item: item[2])
        center_x_values.append(center_x)
        radius_values.append(radius)

    if len(center_x_values) < 8:
        raise RuntimeError("Not enough circles to calibrate this video")

    center_x = float(np.median(center_x_values))
    radius = float(np.median(radius_values))
    inliers = [
        index
        for index, value in enumerate(center_x_values)
        if abs(value - center_x) <= 24.0
    ]
    if len(inliers) >= 8:
        center_x = float(np.median([center_x_values[index] for index in inliers]))
        radius = float(np.median([radius_values[index] for index in inliers]))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return center_x, radius, len(inliers)


def circle_is_centered(crop: np.ndarray, expected_radius: float) -> bool:
    half = crop.shape[0] / 2.0
    candidates = hough_candidates(
        crop,
        min_radius=max(20, int(round(expected_radius - 18))),
        max_radius=min(int(half - 4), int(round(expected_radius + 18))),
        accumulator_threshold=25,
    )
    for center_x, center_y, radius in candidates:
        if (
            np.hypot(center_x - half, center_y - half) <= 7.0
            and radius + 4.0 <= half
        ):
            return True
    return False


def extract_video_autocalibrated(
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
        channel_center_x,
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
    calibrated_x, calibrated_radius, calibration_inliers = calibrate_video(
        capture,
        frame_count=frame_count,
        search_roi=acquisition_roi,
    )

    tracker = DropletSequenceTracker(
        new_droplet_jump=new_droplet_jump,
        max_missing=droplet_max_missing,
    )
    records: list[base.CropRecord] = []
    raw_circles = 0
    validated_circles = 0
    frame_index = 0
    half = crop_size / 2.0

    min_radius = max(22, int(round(calibrated_radius - 18)))
    max_radius = min(int(half - 5), int(round(calibrated_radius + 18)))
    if min_radius >= max_radius:
        min_radius = max(22, max_radius - 20)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        roi = crop_rect(frame, acquisition_roi)
        candidates = hough_candidates(
            roi,
            min_radius=min_radius,
            max_radius=max_radius,
            accumulator_threshold=25,
        )
        ranked = [
            (
                abs(center_x - calibrated_x)
                + 0.35 * abs(radius - calibrated_radius),
                center_x,
                center_y,
                radius,
            )
            for center_x, center_y, radius in candidates
            if abs(center_x - calibrated_x) <= 18.0
        ]
        raw_observation = None
        selected_circle = None
        if ranked:
            _, center_x, center_y, radius = min(ranked)
            global_x = acquisition_roi.x + center_x
            global_y = acquisition_roi.y + center_y
            if (
                global_x >= half
                and global_x < width - half
                and global_y >= half
                and global_y < height - half
                and radius + 5.0 <= half
            ):
                raw_circles += 1
                selected_circle = (global_x, global_y, radius)
                raw_observation = DropletObservation(
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

        tracker.update(raw_observation, frame_index)
        if selected_circle is not None:
            global_x, global_y, radius = selected_circle
            crop = centered_crop(frame, global_x, global_y, crop_size)
            if circle_is_centered(crop, radius):
                validated_circles += 1
                records.append(
                    base.CropRecord(
                        frame_index=frame_index,
                        timestamp_sec=(frame_index / fps if fps > 0 else 0.0),
                        sequence_id=tracker.sequence_id,
                        center_x=global_x,
                        center_y=global_y,
                        radius=radius,
                        score=radius,
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
        "detected_frames": raw_circles,
        "complete_droplet_frames": validated_circles,
        "droplet_sequences": tracker.sequence_id,
        "locator": "per-video autocalibrated Hough with crop revalidation",
        "calibrated_center_x_in_search_roi": calibrated_x,
        "calibrated_radius": calibrated_radius,
        "calibration_inliers": calibration_inliers,
    }
    return records, metadata


def main() -> None:
    base.extract_video = extract_video_autocalibrated
    base.main()


if __name__ == "__main__":
    main()
