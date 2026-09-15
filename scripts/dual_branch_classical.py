#!/usr/bin/env python3
"""Hardware-oriented small-particle detector and QNN fusion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ClassicalDetection:
    """Normalized cell candidate emitted by the classical branch."""

    box: tuple[float, float, float, float]
    score: float
    center: tuple[float, float]
    response: int


@dataclass(frozen=True)
class DetectorConfig:
    """Integer-friendly detector parameters shared by Python and RTL."""

    response_threshold: int = 11
    local_contrast_threshold: int = 20
    minimum_distance: int = 5
    box_size: int = 8
    border: int = 4
    maximum_candidates: int = 24
    droplet_margin: float = 0.08


def _odd_box_mean(gray: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Box-filter size must be a positive odd number")
    return cv2.boxFilter(
        gray,
        ddepth=cv2.CV_16S,
        ksize=(size, size),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )


def radial_response(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return bright-center/dark-ring response and local dynamic range.

    The response uses only box sums, subtraction, min/max morphology and
    comparisons. This keeps the reference algorithm practical for RTL.
    """

    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected a uint8 grayscale image")
    center = _odd_box_mean(gray, 3)
    surround = _odd_box_mean(gray, 9)
    response = center - surround
    local_max = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    local_min = cv2.erode(gray, np.ones((7, 7), np.uint8))
    contrast = local_max.astype(np.int16) - local_min.astype(np.int16)
    return response, contrast


def _inside_any_droplet(
    x: float,
    y: float,
    droplets: Sequence[tuple[float, float, float, float]],
    margin: float,
) -> bool:
    if not droplets:
        return True
    for x1, y1, x2, y2 in droplets:
        width = x2 - x1
        height = y2 - y1
        if (
            x1 - margin * width <= x <= x2 + margin * width
            and y1 - margin * height <= y <= y2 + margin * height
        ):
            return True
    return False


def detect_small_particles(
    gray: np.ndarray,
    config: DetectorConfig,
    *,
    droplet_boxes: Sequence[tuple[float, float, float, float]] = (),
) -> list[ClassicalDetection]:
    """Detect small ring-like cells in a normalized grayscale ROI."""

    response, contrast = radial_response(gray)
    peak_map = cv2.dilate(response, np.ones((3, 3), np.uint8))
    valid = (
        (response >= config.response_threshold)
        & (contrast >= config.local_contrast_threshold)
        & (response == peak_map)
    )
    border = max(config.border, config.box_size // 2)
    valid[:border, :] = False
    valid[-border:, :] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    ys, xs = np.nonzero(valid)
    ranked = sorted(
        zip(xs.tolist(), ys.tolist()),
        key=lambda point: (
            int(response[point[1], point[0]]),
            int(contrast[point[1], point[0]]),
        ),
        reverse=True,
    )
    accepted: list[tuple[int, int]] = []
    detections: list[ClassicalDetection] = []
    height, width = gray.shape
    half = config.box_size / 2.0
    for x, y in ranked:
        nx = (x + 0.5) / width
        ny = (y + 0.5) / height
        if not _inside_any_droplet(nx, ny, droplet_boxes, config.droplet_margin):
            continue
        if any(
            (x - previous_x) ** 2 + (y - previous_y) ** 2
            < config.minimum_distance**2
            for previous_x, previous_y in accepted
        ):
            continue
        accepted.append((x, y))
        x1 = max(0.0, (x + 0.5 - half) / width)
        y1 = max(0.0, (y + 0.5 - half) / height)
        x2 = min(1.0, (x + 0.5 + half) / width)
        y2 = min(1.0, (y + 0.5 + half) / height)
        raw_response = int(response[y, x])
        score = min(1.0, max(0.0, raw_response / 32.0))
        detections.append(
            ClassicalDetection(
                box=(x1, y1, x2, y2),
                score=score,
                center=(nx, ny),
                response=raw_response,
            )
        )
        if len(detections) >= config.maximum_candidates:
            break
    return detections


def box_iou(
    left: Sequence[float], right: Sequence[float]
) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def center_distance(
    left: Sequence[float], right: Sequence[float]
) -> float:
    left_x = (float(left[0]) + float(left[2])) / 2.0
    left_y = (float(left[1]) + float(left[3])) / 2.0
    right_x = (float(right[0]) + float(right[2])) / 2.0
    right_y = (float(right[1]) + float(right[3])) / 2.0
    return float(np.hypot(left_x - right_x, left_y - right_y))


def fuse_cell_boxes(
    qnn_boxes: Iterable[Sequence[float]],
    classical: Sequence[ClassicalDetection],
    *,
    maximum_center_distance: float = 0.055,
    accept_classical_only: bool = True,
) -> list[tuple[float, float, float, float]]:
    """Merge duplicate QNN/classical cells without changing QNN geometry."""

    fused = [tuple(float(value) for value in box) for box in qnn_boxes]
    for candidate in classical:
        if any(
            center_distance(candidate.box, existing) <= maximum_center_distance
            or box_iou(candidate.box, existing) >= 0.15
            for existing in fused
        ):
            continue
        if accept_classical_only:
            fused.append(candidate.box)
    return fused
