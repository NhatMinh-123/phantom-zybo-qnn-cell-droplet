#!/usr/bin/env python3
"""Compare temporal references for the reduced one-droplet ROI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_feature_ml_frame import read_frame, select_circle  # noqa: E402
from scripts.extract_droplet_microplastic_features import (  # noqa: E402
    Rect,
    calibrate_dominant_circle,
    centered_crop,
)
from scripts.feature_ml_motion_candidates import (  # noqa: E402
    extract_motion_fused_candidates,
)
from scripts.run_feature_ml_video_test import (  # noqa: E402
    DEFAULT_MODEL,
    candidate_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=530)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_roi = Rect(578, 10, 144, 160)
    capture = cv2.VideoCapture(str(args.source.resolve()))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    calibrated_x, calibrated_radius, _ = calibrate_dominant_circle(
        capture,
        frame_count=frame_count,
        search_roi=search_roi,
    )
    min_radius = max(18, int(round(calibrated_radius - 11)))
    max_radius = min(94, int(round(calibrated_radius + 11)))

    patches: dict[int, np.ndarray] = {}
    circles: dict[int, tuple[float, float, float]] = {}
    for frame_index in range(args.frame - 5, args.frame + 1):
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
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles[frame_index] = circle
        patches[frame_index] = centered_crop(
            gray,
            circle[0],
            circle[1],
            48,
        )
    current = patches[args.frame]
    history = [
        patches[index]
        for index in sorted(patches)
        if args.frame - 4 <= index < args.frame
    ]
    references: dict[str, np.ndarray] = {}
    for lag in range(1, 5):
        index = args.frame - lag
        if index in patches:
            references[f"lag_{lag}"] = patches[index]
    stack = np.stack(history).astype(np.float32)
    references["history_median"] = np.median(stack, axis=0).astype(np.uint8)
    references["history_max"] = np.max(stack, axis=0).astype(np.uint8)
    model = joblib.load(args.model.resolve())

    results: dict[str, object] = {}
    for name, reference in references.items():
        candidates, diagnostics = extract_motion_fused_candidates(
            current,
            reference,
            video_slug="diagnostic",
            video_name=args.source.name,
            frame_index=args.frame,
            timestamp_sec=0.0,
            sequence_id=1,
            droplet_radius_px=circles[args.frame][2],
            core_radius_px=18,
            patch_size=32,
            blackhat_kernel=7,
            blackhat_sigma=2.0,
            temporal_sigma=1.5,
            min_area=2,
            max_area=100,
            max_side=18,
            max_candidates=8,
            dark_sigma=1.0,
        )
        probabilities = candidate_probabilities(model, candidates)
        results[name] = {
            "candidate_count": len(candidates),
            "pixel_areas": [
                int(candidate.row["pixel_area"]) for candidate in candidates
            ],
            "sources": [
                str(candidate.row["proposal_source"])
                for candidate in candidates
            ],
            "probabilities": [float(value) for value in probabilities],
            "diagnostics": diagnostics,
        }
    capture.release()
    print(
        json.dumps(
            {
                "frame": args.frame,
                "localized_frames": sorted(patches),
                "references": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
