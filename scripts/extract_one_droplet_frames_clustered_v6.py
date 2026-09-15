#!/usr/bin/env python3
"""Cluster-calibrated one-droplet cropper with strict outer-ring matching."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.extract_one_droplet_frames_for_labeling as base  # noqa: E402
from scripts.extract_one_droplet_frames_autocalibrated_v4 import (  # noqa: E402
    hough_candidates,
)
from scripts.run_microplastic_one_droplet_hybrid import (  # noqa: E402
    DropletObservation,
    DropletSequenceTracker,
    Rect,
    centered_crop,
    crop_rect,
)


def calibrate_dominant_circle(
    capture: cv2.VideoCapture,
    *,
    frame_count: int,
    search_roi: Rect,
) -> tuple[float, float, int]:
    observations: list[tuple[float, float]] = []
    positions = np.linspace(0, max(frame_count - 1, 0), min(64, frame_count))
    for frame_index in positions.round().astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        roi = crop_rect(frame, search_roi)
        for center_x, _, radius in hough_candidates(
            roi,
            min_radius=24,
            max_radius=88,
            accumulator_threshold=23,
        ):
            observations.append((center_x, radius))

    if len(observations) < 12:
        raise RuntimeError("Not enough Hough observations for calibration")

    densities: list[int] = []
    for center_x, radius in observations:
        density = sum(
            abs(other_x - center_x) <= 12.0
            and abs(other_radius - radius) <= 7.0
            for other_x, other_radius in observations
        )
        densities.append(density)
    seed_index = int(np.argmax(densities))
    seed_x, seed_radius = observations[seed_index]
    inliers = [
        (center_x, radius)
        for center_x, radius in observations
        if abs(center_x - seed_x) <= 14.0
        and abs(radius - seed_radius) <= 9.0
    ]
    if len(inliers) < 10:
        raise RuntimeError("Dominant circle cluster is too small")

    calibrated_x = float(np.median([item[0] for item in inliers]))
    calibrated_radius = float(np.median([item[1] for item in inliers]))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return calibrated_x, calibrated_radius, len(inliers)


def crop_matches_outer_ring(
    crop: np.ndarray,
    *,
    expected_radius: float,
) -> bool:
    half = crop.shape[0] / 2.0
    candidates = hough_candidates(
        crop,
        min_radius=max(22, int(round(expected_radius - 10))),
        max_radius=min(int(half - 5), int(round(expected_radius + 10))),
        accumulator_threshold=26,
    )
    return any(
        np.hypot(center_x - half, center_y - half) <= 5.5
        and abs(radius - expected_radius) <= 9.0
        and radius + 5.0 <= half
        for center_x, center_y, radius in candidates
    )


def extract_video_clustered(
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
    calibrated_x, calibrated_radius, cluster_size = calibrate_dominant_circle(
        capture,
        frame_count=frame_count,
        search_roi=acquisition_roi,
    )

    tracker = DropletSequenceTracker(
        new_droplet_jump=new_droplet_jump,
        max_missing=droplet_max_missing,
    )
    records: list[base.CropRecord] = []
    raw_matches = 0
    validated_matches = 0
    frame_index = 0
    half = crop_size / 2.0
    min_radius = max(22, int(round(calibrated_radius - 11)))
    max_radius = min(int(half - 5), int(round(calibrated_radius + 11)))

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
        matches = [
            (center_x, center_y, radius)
            for center_x, center_y, radius in candidates
            if abs(center_x - calibrated_x) <= 14.0
            and abs(radius - calibrated_radius) <= 10.0
        ]
        selected = None
        if matches:
            selected = min(
                matches,
                key=lambda item: (
                    abs(item[0] - calibrated_x)
                    + 1.1 * abs(item[2] - calibrated_radius)
                ),
            )

        raw_observation = None
        global_circle = None
        if selected is not None:
            center_x, center_y, radius = selected
            global_x = acquisition_roi.x + center_x
            global_y = acquisition_roi.y + center_y
            if (
                global_x >= half
                and global_x < width - half
                and global_y >= half
                and global_y < height - half
                and radius + 5.0 <= half
            ):
                raw_matches += 1
                global_circle = (global_x, global_y, radius)
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
        if global_circle is not None:
            global_x, global_y, radius = global_circle
            crop = centered_crop(frame, global_x, global_y, crop_size)
            if crop_matches_outer_ring(
                crop,
                expected_radius=calibrated_radius,
            ):
                validated_matches += 1
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
        "detected_frames": raw_matches,
        "complete_droplet_frames": validated_matches,
        "droplet_sequences": tracker.sequence_id,
        "locator": "dominant x-radius Hough cluster with strict crop validation",
        "calibrated_center_x_in_search_roi": calibrated_x,
        "calibrated_radius": calibrated_radius,
        "calibration_cluster_size": cluster_size,
    }
    return records, metadata


def main() -> None:
    base.extract_video = extract_video_clustered
    base.main()


if __name__ == "__main__":
    main()
