#!/usr/bin/env python3
"""Inspect all particle candidates inside one complete droplet frame."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=530)
    parser.add_argument("--core-ratio", type=float, default=0.84)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
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
    patches: list[np.ndarray] = []
    circle = None
    frame = None
    for frame_index in range(args.frame - 4, args.frame + 1):
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
        patches.append(
            centered_crop(gray, circle[0], circle[1], 128)
        )
    current = patches[-1]
    reference = np.max(np.stack(patches[:-1]), axis=0).astype(np.uint8)
    core_radius = min(
        60,
        max(10, int(round(circle[2] * args.core_ratio))),
    )
    candidates, diagnostics = extract_motion_fused_candidates(
        current,
        reference,
        video_slug="diagnostic",
        video_name=args.source.name,
        frame_index=args.frame,
        timestamp_sec=0.0,
        sequence_id=1,
        droplet_radius_px=circle[2],
        core_radius_px=core_radius,
        patch_size=32,
        blackhat_kernel=7,
        blackhat_sigma=2.0,
        temporal_sigma=1.5,
        min_area=2,
        max_area=100,
        max_side=18,
        max_candidates=32,
        dark_sigma=1.0,
    )
    model = joblib.load(args.model.resolve())
    probabilities = candidate_probabilities(model, candidates)
    rows: list[dict[str, object]] = []
    annotated = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
    for index, (candidate, probability) in enumerate(
        zip(candidates, probabilities),
        start=1,
    ):
        row = candidate.row
        rows.append(
            {
                "index": index,
                "probability": float(probability),
                "source": row["proposal_source"],
                "center_x": round(float(row["center_x_px"]), 2),
                "center_y": round(float(row["center_y_px"]), 2),
                "pixel_area": int(row["pixel_area"]),
                "bbox_width": int(row["bbox_width"]),
                "bbox_height": int(row["bbox_height"]),
                "local_contrast": round(
                    float(row["local_contrast_gray"]),
                    3,
                ),
                "blackhat_max": float(row["blackhat_max"]),
                "temporal_max": float(row["temporal_max"]),
                "radial_norm": round(
                    float(row["radial_distance_norm"]),
                    3,
                ),
            }
        )
        x = int(row["bbox_x"])
        y = int(row["bbox_y"])
        width = int(row["bbox_width"])
        height = int(row["bbox_height"])
        color = (0, 255, 0) if probability >= 0.5 else (130, 130, 130)
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            color,
            1,
        )
        cv2.putText(
            annotated,
            str(index),
            (x, max(9, y - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.circle(annotated, (64, 64), core_radius, (255, 120, 0), 1)
    cv2.imwrite(str(output / "full_droplet_candidates.jpg"), annotated)
    report = {
        "frame": args.frame,
        "core_ratio": args.core_ratio,
        "core_radius": core_radius,
        "candidate_count": len(rows),
        "accepted_count": sum(row["probability"] >= 0.5 for row in rows),
        "diagnostics": diagnostics,
        "candidates": rows,
    }
    (output / "candidates.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    capture.release()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
