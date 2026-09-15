#!/usr/bin/env python3
"""Full-droplet detector with registered temporal particle proposals."""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_feature_ml_full_droplet_particle as full  # noqa: E402
from scripts import run_feature_ml_small_roi_motion_v2 as motion_v2  # noqa: E402
from scripts.feature_ml_motion_candidates import (  # noqa: E402
    extract_motion_fused_candidates,
)


def align_translation(
    historical: np.ndarray,
    current: np.ndarray,
    *,
    maximum_shift: float = 4.0,
) -> tuple[np.ndarray, tuple[float, float], float]:
    height, width = current.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    historical_float = cv2.GaussianBlur(historical, (3, 3), 0).astype(
        np.float32
    )
    current_float = cv2.GaussianBlur(current, (3, 3), 0).astype(np.float32)
    shift, response = cv2.phaseCorrelate(
        historical_float,
        current_float,
        window,
    )
    dx, dy = float(shift[0]), float(shift[1])
    if (
        response < 0.08
        or abs(dx) > maximum_shift
        or abs(dy) > maximum_shift
    ):
        return historical, (0.0, 0.0), float(response)
    transform = np.asarray(
        [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        dtype=np.float32,
    )
    aligned = cv2.warpAffine(
        historical,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, (dx, dy), float(response)


def physically_plausible(candidate) -> bool:
    row = candidate.row
    radial_norm = float(row["radial_distance_norm"])
    local_contrast = float(row["local_contrast_gray"])
    blackhat_max = float(row["blackhat_max"])
    temporal_max = float(row["temporal_max"])
    if radial_norm > 0.92:
        return False
    visible_dark_blob = local_contrast >= 1.5 or blackhat_max >= 4.0
    strong_motion_blob = temporal_max >= 10.0 and local_contrast >= 0.5
    return visible_dark_blob or strong_motion_blob


class AlignedMaxHistoryMotionExtractor(
    motion_v2.MaxHistoryMotionExtractor
):
    def __init__(self, *, lag: int, dark_sigma: float) -> None:
        super().__init__(lag=lag, dark_sigma=dark_sigma)
        self.history: deque[np.ndarray] = deque(maxlen=self.lag)
        self.alignment_samples = 0
        self.alignment_shift_sum = 0.0

    def __call__(
        self,
        gray_patch: np.ndarray,
        previous_patch: np.ndarray | None,
        **kwargs,
    ):
        if previous_patch is None:
            self.history.clear()
        reference = None
        if len(self.history) == self.lag:
            aligned_history = []
            for historical in self.history:
                aligned, shift, _ = align_translation(
                    historical,
                    gray_patch,
                )
                aligned_history.append(aligned)
                self.alignment_samples += 1
                self.alignment_shift_sum += float(
                    np.hypot(shift[0], shift[1])
                )
            reference = np.max(
                np.stack(aligned_history),
                axis=0,
            ).astype(np.uint8)

        candidates, diagnostics = extract_motion_fused_candidates(
            gray_patch,
            reference,
            dark_sigma=self.dark_sigma,
            **kwargs,
        )
        candidates = [
            candidate
            for candidate in candidates
            if physically_plausible(candidate)
        ]
        self.history.append(gray_patch.copy())
        for candidate in candidates:
            source = str(candidate.row.get("proposal_source", "unknown"))
            self.source_counts[source] += 1
            self.frame_counts[int(candidate.row["frame_index"])] += 1
        diagnostics["physically_plausible_candidates"] = float(
            len(candidates)
        )
        return candidates, diagnostics


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
    summary["experiment"] = (
        "feature_ml_full_droplet_particle_registered_v2"
    )
    summary["particle_candidate_filter"] = {
        "temporal_registration": "phase-correlation translation",
        "maximum_registration_shift_px": 4.0,
        "maximum_radial_norm": 0.92,
        "dark_blob_rule": "local_contrast >= 1.5 OR blackhat_max >= 4",
        "motion_rule": "temporal_max >= 10 AND local_contrast >= 0.5",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    ensure_argument("--processing-size", "128")
    ensure_argument("--core-radius-ratio", "0.84")
    ensure_argument("--blackhat-sigma", "3.2")
    ensure_argument("--max-candidates", "16")
    ensure_argument("--max-kept-candidates", "4")
    ensure_argument("--track-max-distance", "12")
    output_value = argument_value("--output")
    if output_value is None:
        raise ValueError("--output is required")

    motion_v2.MaxHistoryMotionExtractor = (
        AlignedMaxHistoryMotionExtractor
    )
    full.main()
    enrich_summary(Path(output_value))


if __name__ == "__main__":
    main()
