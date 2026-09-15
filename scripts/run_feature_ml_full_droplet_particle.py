#!/usr/bin/env python3
"""Detect one complete droplet and particle candidates inside it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_feature_ml_small_roi_motion_test as v1  # noqa: E402
from scripts import run_feature_ml_small_roi_motion_v2 as v2  # noqa: E402


def draw_particle_candidate(
    image,
    *,
    bbox,
    track_id: int,
    probability: float,
    hits: int,
    threshold: float,
    minimum_hits: int,
    label: bool,
) -> None:
    accepted = probability >= threshold
    confirmed = accepted and hits >= minimum_hits
    if confirmed:
        color = (40, 220, 40)
        thickness = 2
    elif accepted:
        color = (0, 180, 255)
        thickness = 2
    else:
        color = (130, 130, 130)
        thickness = 1
    x, y, width, height = bbox
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        color,
        thickness,
    )
    if accepted:
        state = "confirmed" if confirmed else "pending"
        v1.baseline.draw_text(
            image,
            f"particle {probability:.2f} {state}",
            (max(0, x), max(13, y - 3)),
            color=color,
            scale=0.34,
        )


def install_droplet_particle_geometry(processing_size: int):
    original_circle = cv2.circle
    original_rectangle = cv2.rectangle

    def droplet_circle(
        image,
        center,
        radius,
        color,
        thickness=1,
        *args,
        **kwargs,
    ):
        if image.ndim == 3 and tuple(color) == (0, 220, 255):
            cx, cy = int(center[0]), int(center[1])
            radius_i = int(round(radius))
            original_rectangle(
                image,
                (cx - radius_i, cy - radius_i),
                (cx + radius_i, cy + radius_i),
                (255, 120, 0),
                2,
            )
            v1.baseline.draw_text(
                image,
                "droplet",
                (max(0, cx - radius_i), max(13, cy - radius_i - 4)),
                color=(255, 120, 0),
                scale=0.42,
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

    def hide_search_rectangle(
        image,
        pt1,
        pt2,
        color,
        thickness=1,
        *args,
        **kwargs,
    ):
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

    cv2.circle = droplet_circle
    cv2.rectangle = hide_search_rectangle
    return original_circle, original_rectangle


def ensure_argument(name: str, value: str) -> None:
    if name not in sys.argv:
        sys.argv.extend([name, value])


def argument_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def enrich_summary(output: Path) -> None:
    summary_path = output.resolve() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = "feature_ml_full_droplet_and_particle"
    summary["detection_outputs"] = {
        "droplet": {
            "method": "Hough circle localization",
            "rendered_as": "one blue bounding box labelled droplet",
            "approximately_full_diameter_px": 120,
        },
        "particle": {
            "method": (
                "motion-fused feature proposals followed by Decision Tree"
            ),
            "rendered_as": (
                "orange pending or green confirmed boxes labelled particle"
            ),
            "search_region": (
                "droplet interior core; core boundary is not rendered"
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    ensure_argument("--processing-size", "128")
    ensure_argument("--core-radius-ratio", "0.72")
    ensure_argument("--max-kept-candidates", "4")
    ensure_argument("--track-max-distance", "12")
    ensure_argument("--zoom-size", "260")
    output_value = argument_value("--output")
    if output_value is None:
        raise ValueError("--output is required")

    v1.install_simple_geometry = install_droplet_particle_geometry
    v1.baseline.draw_candidate = draw_particle_candidate
    v2.main()
    enrich_summary(Path(output_value))


if __name__ == "__main__":
    main()
