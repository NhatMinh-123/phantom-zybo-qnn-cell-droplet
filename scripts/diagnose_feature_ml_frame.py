#!/usr/bin/env python3
"""Diagnose proposal masks for one frame of the feature-ML pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_droplet_microplastic_features import (  # noqa: E402
    Rect,
    calibrate_dominant_circle,
    centered_crop,
    crop_rect,
    hough_candidates,
)
from scripts.run_microplastic_one_droplet_hybrid import (  # noqa: E402
    robust_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=530)
    parser.add_argument("--search-roi-x", type=int, default=560)
    parser.add_argument("--search-roi-y", type=int, default=0)
    parser.add_argument("--search-roi-width", type=int, default=320)
    parser.add_argument("--search-roi-height", type=int, default=360)
    parser.add_argument("--processing-size", type=int, default=128)
    parser.add_argument("--core-radius-ratio", type=float, default=0.30)
    parser.add_argument("--blackhat-kernel", type=int, default=7)
    parser.add_argument("--blackhat-sigma", type=float, default=3.2)
    parser.add_argument("--temporal-sigma", type=float, default=2.5)
    parser.add_argument("--dark-sigma", type=float, default=1.5)
    return parser.parse_args()


def select_circle(
    frame: np.ndarray,
    *,
    search_roi: Rect,
    calibrated_x: float,
    calibrated_radius: float,
    min_radius: int,
    max_radius: int,
) -> tuple[float, float, float] | None:
    roi = crop_rect(frame, search_roi)
    circles = hough_candidates(
        roi,
        min_radius=min_radius,
        max_radius=max_radius,
        accumulator_threshold=25,
    )
    matches = [
        circle
        for circle in circles
        if abs(circle[0] - calibrated_x) <= 14.0
        and abs(circle[2] - calibrated_radius) <= 10.0
    ]
    if not matches:
        return None
    local_x, local_y, radius = min(
        matches,
        key=lambda item: (
            abs(item[0] - calibrated_x)
            + 1.1 * abs(item[2] - calibrated_radius)
        ),
    )
    return search_roi.x + local_x, search_roi.y + local_y, radius


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def display_plane(plane: np.ndarray, title: str, size: int = 300) -> np.ndarray:
    normalized = cv2.normalize(plane, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    image = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(image, (0, 0), (size, 28), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def count_components(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return int(
        sum(
            2 <= int(stats[index, cv2.CC_STAT_AREA]) <= 100
            for index in range(1, count)
        )
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    search_roi = Rect(
        args.search_roi_x,
        args.search_roi_y,
        args.search_roi_width,
        args.search_roi_height,
    )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    calibrated_x, calibrated_radius, cluster_size = calibrate_dominant_circle(
        capture,
        frame_count=frame_count,
        search_roi=search_roi,
    )
    min_radius = max(18, int(round(calibrated_radius - 11)))
    max_radius = min(94, int(round(calibrated_radius + 11)))

    current_frame = read_frame(capture, args.frame)
    previous_frame = read_frame(capture, max(0, args.frame - 1))
    current_circle = select_circle(
        current_frame,
        search_roi=search_roi,
        calibrated_x=calibrated_x,
        calibrated_radius=calibrated_radius,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    previous_circle = select_circle(
        previous_frame,
        search_roi=search_roi,
        calibrated_x=calibrated_x,
        calibrated_radius=calibrated_radius,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    if current_circle is None or previous_circle is None:
        raise RuntimeError(
            f"Droplet localization failed: current={current_circle}, "
            f"previous={previous_circle}"
        )

    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_patch = centered_crop(
        current_gray,
        current_circle[0],
        current_circle[1],
        args.processing_size,
    )
    previous_patch = centered_crop(
        previous_gray,
        previous_circle[0],
        previous_circle[1],
        args.processing_size,
    )
    core_radius = min(
        args.processing_size // 2 - 4,
        max(10, int(round(current_circle[2] * args.core_radius_ratio))),
    )
    core_mask = np.zeros_like(current_patch)
    cv2.circle(
        core_mask,
        (args.processing_size // 2, args.processing_size // 2),
        core_radius,
        255,
        -1,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.blackhat_kernel, args.blackhat_kernel),
    )
    blackhat = cv2.morphologyEx(
        current_patch,
        cv2.MORPH_BLACKHAT,
        kernel,
    )
    temporal = cv2.absdiff(current_patch, previous_patch)
    background = cv2.GaussianBlur(current_patch, (11, 11), 0)
    dark_response = cv2.subtract(background, current_patch)
    blackhat_threshold = robust_threshold(
        blackhat,
        core_mask,
        args.blackhat_sigma,
        3.0,
    )
    temporal_threshold = robust_threshold(
        temporal,
        core_mask,
        args.temporal_sigma,
        2.0,
    )
    dark_threshold = robust_threshold(
        dark_response,
        core_mask,
        args.dark_sigma,
        2.0,
    )
    blackhat_mask = (
        (blackhat >= blackhat_threshold) & (core_mask > 0)
    ).astype(np.uint8) * 255
    motion_dark_mask = (
        (temporal >= temporal_threshold)
        & (dark_response >= dark_threshold)
        & (core_mask > 0)
    ).astype(np.uint8) * 255
    fused_mask = cv2.bitwise_or(blackhat_mask, motion_dark_mask)

    montage = np.vstack(
        [
            np.hstack(
                [
                    display_plane(current_patch, "current patch"),
                    display_plane(blackhat, "black-hat response"),
                    display_plane(temporal, "temporal difference"),
                ]
            ),
            np.hstack(
                [
                    display_plane(blackhat_mask, "old proposal mask"),
                    display_plane(motion_dark_mask, "motion + dark mask"),
                    display_plane(fused_mask, "fused proposal mask"),
                ]
            ),
        ]
    )
    cv2.imwrite(str(output / f"frame_{args.frame:06d}_diagnostic.jpg"), montage)

    annotated = current_frame.copy()
    cx, cy, radius = current_circle
    half = args.processing_size // 2
    cv2.rectangle(
        annotated,
        (int(round(cx)) - half, int(round(cy)) - half),
        (int(round(cx)) + half, int(round(cy)) + half),
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(output / f"frame_{args.frame:06d}_source.jpg"), annotated)

    core = core_mask > 0
    report = {
        "frame": args.frame,
        "calibration": {
            "x": calibrated_x,
            "radius": calibrated_radius,
            "cluster_size": cluster_size,
        },
        "circle": {
            "x": current_circle[0],
            "y": current_circle[1],
            "radius": current_circle[2],
        },
        "processing_size": args.processing_size,
        "core_radius": core_radius,
        "thresholds": {
            "blackhat": blackhat_threshold,
            "temporal": temporal_threshold,
            "dark": dark_threshold,
        },
        "core_maxima": {
            "blackhat": int(np.max(blackhat[core])),
            "temporal": int(np.max(temporal[core])),
            "dark": int(np.max(dark_response[core])),
        },
        "components_area_2_to_100": {
            "old_blackhat_only": count_components(blackhat_mask),
            "motion_dark_only": count_components(motion_dark_mask),
            "fused": count_components(fused_mask),
        },
        "artifacts": {
            "diagnostic": str(
                (output / f"frame_{args.frame:06d}_diagnostic.jpg").resolve()
            ),
            "source": str(
                (output / f"frame_{args.frame:06d}_source.jpg").resolve()
            ),
        },
    }
    (output / "diagnostic.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    capture.release()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
