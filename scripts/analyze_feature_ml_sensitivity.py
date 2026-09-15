#!/usr/bin/env python3
"""Measure low-contrast proposal responses across kernels and frame lags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_feature_ml_frame import (  # noqa: E402
    read_frame,
    select_circle,
)
from scripts.extract_droplet_microplastic_features import (  # noqa: E402
    Rect,
    calibrate_dominant_circle,
    centered_crop,
)
from scripts.run_microplastic_one_droplet_hybrid import (  # noqa: E402
    robust_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=530)
    parser.add_argument("--processing-size", type=int, default=48)
    parser.add_argument("--core-radius-ratio", type=float, default=0.30)
    return parser.parse_args()


def component_areas(mask: np.ndarray) -> list[int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return sorted(
        [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)],
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    search_roi = Rect(560, 0, 320, 360)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    calibrated_x, calibrated_radius, _ = calibrate_dominant_circle(
        capture,
        frame_count=frame_count,
        search_roi=search_roi,
    )
    min_radius = max(18, int(round(calibrated_radius - 11)))
    max_radius = min(94, int(round(calibrated_radius + 11)))

    def patch_at(frame_index: int) -> tuple[np.ndarray, tuple[float, float, float]]:
        frame = read_frame(capture, frame_index)
        circle = select_circle(
            frame,
            search_roi=search_roi,
            calibrated_x=calibrated_x,
            calibrated_radius=calibrated_radius,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        if circle is None:
            raise RuntimeError(f"No droplet at frame {frame_index}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return (
            centered_crop(
                gray,
                circle[0],
                circle[1],
                args.processing_size,
            ),
            circle,
        )

    current, circle = patch_at(args.frame)
    core_radius = min(
        args.processing_size // 2 - 4,
        max(10, int(round(circle[2] * args.core_radius_ratio))),
    )
    core_mask = np.zeros_like(current)
    cv2.circle(
        core_mask,
        (args.processing_size // 2, args.processing_size // 2),
        core_radius,
        255,
        -1,
    )
    core = core_mask > 0
    background = cv2.GaussianBlur(current, (11, 11), 0)
    dark = cv2.subtract(background, current)
    dark_threshold = robust_threshold(dark, core_mask, 1.0, 1.0)
    rows: list[dict[str, object]] = []

    for kernel_size in (5, 7, 9, 11, 15, 21):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        blackhat = cv2.morphologyEx(current, cv2.MORPH_BLACKHAT, kernel)
        for sigma in (1.0, 1.5, 2.0, 2.5, 3.2):
            threshold = robust_threshold(blackhat, core_mask, sigma, 1.0)
            mask = (
                (blackhat >= threshold) & core
            ).astype(np.uint8) * 255
            rows.append(
                {
                    "branch": "blackhat",
                    "kernel_or_lag": kernel_size,
                    "sigma": sigma,
                    "threshold": threshold,
                    "maximum": int(np.max(blackhat[core])),
                    "pixels": int(np.count_nonzero(mask)),
                    "areas": component_areas(mask),
                }
            )

    for lag in (1, 2, 3, 4, 5, 7, 10):
        previous, _ = patch_at(args.frame - lag)
        temporal = cv2.absdiff(current, previous)
        for sigma in (1.0, 1.5, 2.0, 2.5):
            threshold = robust_threshold(temporal, core_mask, sigma, 1.0)
            mask = (
                (temporal >= threshold)
                & (dark >= dark_threshold)
                & core
            ).astype(np.uint8) * 255
            rows.append(
                {
                    "branch": "motion_dark",
                    "kernel_or_lag": lag,
                    "sigma": sigma,
                    "threshold": threshold,
                    "maximum": int(np.max(temporal[core])),
                    "pixels": int(np.count_nonzero(mask)),
                    "areas": component_areas(mask),
                }
            )

    report = {
        "frame": args.frame,
        "processing_size": args.processing_size,
        "core_radius": core_radius,
        "dark_threshold": dark_threshold,
        "dark_maximum": int(np.max(dark[core])),
        "measurements": rows,
    }
    (output / "sensitivity.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    capture.release()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
