#!/usr/bin/env python3
"""Eight-direction integer radial detector for 15 um cell candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class RadialConfig:
    response_threshold: int = 12
    local_contrast_threshold: int = 20
    minimum_distance: int = 5
    box_size: int = 8
    border: int = 4
    maximum_candidates: int = 24
    droplet_margin: float = 0.08


@dataclass(frozen=True)
class RadialDetection:
    box: tuple[float, float, float, float]
    score: float
    center: tuple[float, float]
    response: int


def radial_response(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute center minus eight radius-three samples using integers."""

    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected a uint8 grayscale image")
    kernel = np.zeros((7, 7), dtype=np.int16)
    kernel[3, 3] = 8
    for y, x in (
        (0, 3), (6, 3), (3, 0), (3, 6),
        (1, 1), (1, 5), (5, 1), (5, 5),
    ):
        kernel[y, x] = -1
    response_x8 = cv2.filter2D(
        gray,
        cv2.CV_16S,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )
    response = np.floor_divide(response_x8, 8).astype(np.int16)
    local_max = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    local_min = cv2.erode(gray, np.ones((7, 7), np.uint8))
    contrast = local_max.astype(np.int16) - local_min.astype(np.int16)
    return response, contrast


def _inside_droplet(
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


def detect_radial_cells(
    gray: np.ndarray,
    config: RadialConfig,
    *,
    droplet_boxes: Sequence[tuple[float, float, float, float]] = (),
) -> list[RadialDetection]:
    response, contrast = radial_response(gray)
    local_peak = cv2.dilate(response, np.ones((3, 3), np.uint8))
    valid = (
        (response >= config.response_threshold)
        & (contrast >= config.local_contrast_threshold)
        & (response == local_peak)
    )
    border = max(config.border, config.box_size // 2)
    valid[:border, :] = False
    valid[-border:, :] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    ys, xs = np.nonzero(valid)
    ranked = sorted(
        zip(xs.tolist(), ys.tolist()),
        key=lambda point: (response[point[1], point[0]], contrast[point[1], point[0]]),
        reverse=True,
    )
    accepted: list[tuple[int, int]] = []
    output: list[RadialDetection] = []
    height, width = gray.shape
    half = config.box_size / 2.0
    for x, y in ranked:
        nx = (x + 0.5) / width
        ny = (y + 0.5) / height
        if not _inside_droplet(nx, ny, droplet_boxes, config.droplet_margin):
            continue
        if any(
            (x - old_x) ** 2 + (y - old_y) ** 2 < config.minimum_distance**2
            for old_x, old_y in accepted
        ):
            continue
        accepted.append((x, y))
        box = (
            max(0.0, (x + 0.5 - half) / width),
            max(0.0, (y + 0.5 - half) / height),
            min(1.0, (x + 0.5 + half) / width),
            min(1.0, (y + 0.5 + half) / height),
        )
        value = int(response[y, x])
        output.append(
            RadialDetection(
                box=box,
                score=min(1.0, max(0.0, value / 48.0)),
                center=(nx, ny),
                response=value,
            )
        )
        if len(output) >= config.maximum_candidates:
            break
    return output
