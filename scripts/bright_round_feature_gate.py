"""Strict feature gate for compact, high-response particle candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class BrightRoundGateConfig:
    min_blackhat_peak: float = 20.0
    min_blackhat_mean: float = 8.0
    min_peak_ratio: float = 2.5
    min_mean_ratio: float = 1.5
    min_area: float = 4.0
    max_area: float = 100.0
    min_axis_ratio: float = 0.55
    min_circularity: float = 0.42
    min_solidity: float = 0.55
    min_extent: float = 0.50
    max_radial_distance: float = 0.90


@dataclass(frozen=True)
class BrightRoundGateResult:
    passed: bool
    score: float
    reasons: tuple[str, ...]
    features: dict[str, float]


def evaluate_bright_round_candidate(
    row: Mapping[str, object],
    config: BrightRoundGateConfig = BrightRoundGateConfig(),
) -> BrightRoundGateResult:
    width = max(float(row.get("bbox_width", 0.0)), 0.0)
    height = max(float(row.get("bbox_height", 0.0)), 0.0)
    axis_ratio = min(width, height) / max(width, height, 1.0)
    threshold = max(float(row.get("blackhat_threshold", 0.0)), 1.0)
    peak = float(row.get("blackhat_max", 0.0))
    mean = float(row.get("blackhat_mean", 0.0))
    peak_ratio = peak / threshold
    mean_ratio = mean / threshold
    area = float(row.get("pixel_area", 0.0))
    circularity = float(row.get("circularity", 0.0))
    solidity = float(row.get("solidity", 0.0))
    extent = float(row.get("extent", 0.0))
    radial = float(row.get("radial_distance_norm", 0.0))

    checks = {
        "dim_peak": peak >= config.min_blackhat_peak,
        "dim_mean": mean >= config.min_blackhat_mean,
        "weak_peak_ratio": peak_ratio >= config.min_peak_ratio,
        "weak_mean_ratio": mean_ratio >= config.min_mean_ratio,
        "area": config.min_area <= area <= config.max_area,
        "elongated": axis_ratio >= config.min_axis_ratio,
        "not_circular": circularity >= config.min_circularity,
        "not_solid": solidity >= config.min_solidity,
        "diffuse": extent >= config.min_extent,
        "outside_core": radial <= config.max_radial_distance,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)

    brightness_score = float(
        np.mean(
            [
                np.clip((peak - 12.0) / 20.0, 0.0, 1.0),
                np.clip((mean - 5.0) / 9.0, 0.0, 1.0),
                np.clip((peak_ratio - 2.0) / 6.0, 0.0, 1.0),
            ]
        )
    )
    shape_score = float(
        np.mean(
            [
                np.clip(axis_ratio, 0.0, 1.0),
                np.clip(circularity, 0.0, 1.0),
                np.clip(solidity, 0.0, 1.0),
                np.clip(extent, 0.0, 1.0),
            ]
        )
    )
    score = 0.58 * brightness_score + 0.42 * shape_score
    return BrightRoundGateResult(
        passed=not reasons,
        score=float(np.clip(score, 0.0, 1.0)),
        reasons=reasons,
        features={
            "axis_ratio": axis_ratio,
            "peak_ratio": peak_ratio,
            "mean_ratio": mean_ratio,
            "brightness_score": brightness_score,
            "shape_score": shape_score,
        },
    )
