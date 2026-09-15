#!/usr/bin/env python3
"""Final full-droplet plus accepted-particle visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_feature_ml_full_droplet_particle as full  # noqa: E402
from scripts import run_feature_ml_full_droplet_particle_v2 as v2  # noqa: E402
from scripts import run_feature_ml_small_roi_motion_test as core  # noqa: E402


def draw_accepted_particle(
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
    if probability < threshold:
        return
    confirmed = hits >= minimum_hits
    color = (40, 220, 40) if confirmed else (0, 180, 255)
    thickness = 2
    x, y, width, height = bbox
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        color,
        thickness,
    )
    state = "confirmed" if confirmed else "pending"
    core.baseline.draw_text(
        image,
        f"particle {probability:.2f} {state}",
        (max(0, x), max(13, y - 3)),
        color=color,
        scale=0.34,
    )


def main() -> None:
    full.draw_particle_candidate = draw_accepted_particle
    v2.main()


if __name__ == "__main__":
    main()
