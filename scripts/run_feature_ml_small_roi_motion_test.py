#!/usr/bin/env python3
"""Run the feature model with a small ROI and motion-fused proposals.

This keeps the original practical runner unchanged as a baseline. The
launcher injects a three-frame temporal proposal branch, reduces the runtime
ROIs, and renders only the actual 48x48 ML window on the source frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_feature_ml_video_test as baseline  # noqa: E402
from scripts.feature_ml_motion_candidates import (  # noqa: E402
    extract_motion_fused_candidates,
)


class LaggedMotionExtractor:
    def __init__(self, *, lag: int, dark_sigma: float) -> None:
        self.lag = max(1, lag)
        self.dark_sigma = dark_sigma
        self.history: deque[np.ndarray] = deque(maxlen=self.lag)
        self.source_counts: Counter[str] = Counter()
        self.frame_counts: Counter[int] = Counter()

    def __call__(
        self,
        gray_patch: np.ndarray,
        previous_patch: np.ndarray | None,
        **kwargs,
    ):
        if previous_patch is None:
            self.history.clear()
        reference = self.history[0] if len(self.history) == self.lag else None
        candidates, diagnostics = extract_motion_fused_candidates(
            gray_patch,
            reference,
            dark_sigma=self.dark_sigma,
            **kwargs,
        )
        self.history.append(gray_patch.copy())
        for candidate in candidates:
            source = str(candidate.row.get("proposal_source", "unknown"))
            self.source_counts[source] += 1
            self.frame_counts[int(candidate.row["frame_index"])] += 1
        return candidates, diagnostics


def add_dynamic_zoom_inset(
    annotated: np.ndarray,
    processing_patch: np.ndarray,
    *,
    candidates,
    assignments: list[int],
    active_tracks,
    threshold: float,
    minimum_hits: int,
    zoom_size: int,
) -> None:
    zoom = cv2.cvtColor(processing_patch, cv2.COLOR_GRAY2BGR)
    for candidate, track_id in zip(candidates, assignments):
        track = active_tracks[track_id]
        probability = baseline.track_probability(track)
        local_box = (
            int(candidate.row["bbox_x"]),
            int(candidate.row["bbox_y"]),
            int(candidate.row["bbox_width"]),
            int(candidate.row["bbox_height"]),
        )
        baseline.draw_candidate(
            zoom,
            bbox=local_box,
            track_id=track_id,
            probability=probability,
            hits=len(track.observations),
            threshold=threshold,
            minimum_hits=minimum_hits,
            label=True,
        )

    zoom = cv2.resize(
        zoom,
        (zoom_size, zoom_size),
        interpolation=cv2.INTER_NEAREST,
    )
    height, width = annotated.shape[:2]
    inset_x = max(8, width - zoom_size - 12)
    inset_y = max(54, height - zoom_size - 12)
    x2 = min(width, inset_x + zoom_size)
    y2 = min(height, inset_y + zoom_size)
    zoom = zoom[: y2 - inset_y, : x2 - inset_x]
    cv2.rectangle(
        annotated,
        (inset_x - 3, inset_y - 24),
        (x2 + 3, y2 + 3),
        (0, 0, 0),
        -1,
    )
    annotated[inset_y:y2, inset_x:x2] = zoom
    size = int(processing_patch.shape[0])
    baseline.draw_text(
        annotated,
        f"Processing ROI {size}x{size}",
        (inset_x, inset_y - 7),
        color=(255, 255, 255),
        scale=0.42,
    )


def parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-lag", type=int, default=3)
    parser.add_argument("--motion-dark-sigma", type=float, default=1.0)
    parser.add_argument("--search-roi-x", type=int, default=578)
    parser.add_argument("--search-roi-y", type=int, default=10)
    parser.add_argument("--search-roi-width", type=int, default=144)
    parser.add_argument("--search-roi-height", type=int, default=160)
    parser.add_argument("--processing-size", type=int, default=48)
    parser.add_argument("--core-radius-ratio", type=float, default=0.30)
    parser.add_argument("--blackhat-sigma", type=float, default=2.0)
    parser.add_argument("--temporal-sigma", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--minimum-track-hits", type=int, default=3)
    parser.add_argument("--max-kept-candidates", type=int, default=1)
    parser.add_argument("--track-max-distance", type=float, default=8.0)
    parser.add_argument("--zoom-size", type=int, default=240)
    parser.add_argument(
        "--debug-geometry",
        action="store_true",
        help="Show the old search and concentric-circle geometry.",
    )
    return parser.parse_known_args()


def baseline_argv(args: argparse.Namespace, remaining: list[str]) -> list[str]:
    return [
        sys.argv[0],
        "--source",
        str(args.source),
        "--output",
        str(args.output),
        "--search-roi-x",
        str(args.search_roi_x),
        "--search-roi-y",
        str(args.search_roi_y),
        "--search-roi-width",
        str(args.search_roi_width),
        "--search-roi-height",
        str(args.search_roi_height),
        "--processing-size",
        str(args.processing_size),
        "--core-radius-ratio",
        str(args.core_radius_ratio),
        "--blackhat-sigma",
        str(args.blackhat_sigma),
        "--temporal-sigma",
        str(args.temporal_sigma),
        "--threshold",
        str(args.threshold),
        "--minimum-track-hits",
        str(args.minimum_track_hits),
        "--max-kept-candidates",
        str(args.max_kept_candidates),
        "--track-max-distance",
        str(args.track_max_distance),
        "--zoom-size",
        str(args.zoom_size),
        *remaining,
    ]


def install_simple_geometry(processing_size: int):
    original_circle = cv2.circle
    original_rectangle = cv2.rectangle

    def simple_circle(image, center, radius, color, thickness=1, *args, **kwargs):
        if image.ndim == 3 and tuple(color) == (0, 220, 255):
            half = processing_size // 2
            original_rectangle(
                image,
                (int(center[0]) - half, int(center[1]) - half),
                (int(center[0]) + half, int(center[1]) + half),
                (40, 220, 40),
                2,
            )
            return image
        if image.ndim == 3 and tuple(color) == (255, 120, 0):
            return image
        return original_circle(
            image,
            center,
            radius,
            color,
            thickness,
            *args,
            **kwargs,
        )

    def simple_rectangle(image, pt1, pt2, color, thickness=1, *args, **kwargs):
        if (
            image.ndim == 3
            and tuple(color) == (255, 200, 0)
            and thickness == 1
        ):
            return image
        return original_rectangle(
            image,
            pt1,
            pt2,
            color,
            thickness,
            *args,
            **kwargs,
        )

    cv2.circle = simple_circle
    cv2.rectangle = simple_rectangle
    return original_circle, original_rectangle


def update_summary(
    output: Path,
    *,
    args: argparse.Namespace,
    extractor: LaggedMotionExtractor,
) -> None:
    summary_path = output.resolve() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = "feature_ml_small_roi_motion_fused"
    summary["runtime_roi"] = {
        "localization": {
            "x": args.search_roi_x,
            "y": args.search_roi_y,
            "width": args.search_roi_width,
            "height": args.search_roi_height,
            "pixels": args.search_roi_width * args.search_roi_height,
        },
        "feature_processing": {
            "width": args.processing_size,
            "height": args.processing_size,
            "pixels": args.processing_size * args.processing_size,
        },
        "pixel_reduction_vs_previous": {
            "localization_percent": (
                100.0
                * (1.0 - (args.search_roi_width * args.search_roi_height)
                   / (320.0 * 360.0))
            ),
            "feature_processing_percent": (
                100.0
                * (1.0 - (args.processing_size * args.processing_size)
                   / (128.0 * 128.0))
            ),
        },
    }
    summary["proposal_generation"] = {
        "branches": "blackhat OR (lagged temporal AND current-dark)",
        "temporal_lag_frames": args.motion_lag,
        "temporal_sigma": args.temporal_sigma,
        "dark_sigma": args.motion_dark_sigma,
        "candidate_source_counts": dict(extractor.source_counts),
        "frame_530_raw_candidates": int(extractor.frame_counts.get(530, 0)),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args, remaining = parse_launcher_args()
    extractor = LaggedMotionExtractor(
        lag=args.motion_lag,
        dark_sigma=args.motion_dark_sigma,
    )
    baseline.extract_candidates = extractor
    baseline.add_zoom_inset = add_dynamic_zoom_inset
    original_circle = None
    original_rectangle = None
    if not args.debug_geometry:
        original_circle, original_rectangle = install_simple_geometry(
            args.processing_size
        )
    original_argv = sys.argv
    try:
        sys.argv = baseline_argv(args, remaining)
        baseline.main()
    finally:
        sys.argv = original_argv
        if original_circle is not None:
            cv2.circle = original_circle
        if original_rectangle is not None:
            cv2.rectangle = original_rectangle
    update_summary(args.output, args=args, extractor=extractor)


if __name__ == "__main__":
    main()
