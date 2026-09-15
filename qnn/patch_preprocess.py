"""Shared image transforms for patch-QNN training and inference."""

from __future__ import annotations

import cv2
import numpy as np


TRANSFORMS = ("raw", "dark_residual", "bright_residual", "contrast_residual")


def preprocess_patch(image: np.ndarray, mode: str) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Patch preprocessing expects one grayscale image")
    if mode == "raw":
        return image.astype(np.uint8, copy=True)
    if mode == "dark_residual":
        source = image.astype(np.uint8, copy=False)
        baseline = cv2.blur(
            source,
            (9, 9),
            borderType=cv2.BORDER_REFLECT_101,
        )
        residual = baseline.astype(np.int16) - source.astype(np.int16)
        return np.clip(residual * 8, 0, 255).astype(np.uint8)
    if mode == "bright_residual":
        source = image.astype(np.uint8, copy=False)
        baseline = cv2.blur(
            source,
            (9, 9),
            borderType=cv2.BORDER_REFLECT_101,
        )
        residual = source.astype(np.int16) - baseline.astype(np.int16)
        return np.clip(residual * 8, 0, 255).astype(np.uint8)
    if mode == "contrast_residual":
        source = image.astype(np.uint8, copy=False)
        baseline = cv2.blur(
            source,
            (9, 9),
            borderType=cv2.BORDER_REFLECT_101,
        )
        residual = np.abs(
            source.astype(np.int16) - baseline.astype(np.int16)
        )
        return np.clip(residual * 8, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown patch transform: {mode}")