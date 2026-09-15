#!/usr/bin/env python3
"""Motion-fused candidate extraction compatible with the feature-ML model."""

from __future__ import annotations

import math

import cv2
import numpy as np

from scripts.extract_droplet_microplastic_features import (
    CandidateData,
    entropy,
    local_patch,
)
from scripts.run_microplastic_one_droplet_hybrid import robust_threshold


def extract_motion_fused_candidates(
    gray_patch: np.ndarray,
    previous_patch: np.ndarray | None,
    *,
    video_slug: str,
    video_name: str,
    frame_index: int,
    timestamp_sec: float,
    sequence_id: int,
    droplet_radius_px: float,
    core_radius_px: int,
    patch_size: int,
    blackhat_kernel: int,
    blackhat_sigma: float,
    temporal_sigma: float,
    min_area: int,
    max_area: int,
    max_side: int,
    max_candidates: int,
    dark_sigma: float = 1.0,
) -> tuple[list[CandidateData], dict[str, float]]:
    """Create candidates from black-hat OR current-dark temporal motion.

    The temporal branch keeps only pixels that are dark in the current frame.
    This suppresses the bright "ghost" left at an object's previous position.
    """

    height, width = gray_patch.shape
    core_mask = np.zeros_like(gray_patch)
    cv2.circle(
        core_mask,
        (width // 2, height // 2),
        core_radius_px,
        255,
        -1,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (blackhat_kernel, blackhat_kernel),
    )
    blackhat = cv2.morphologyEx(gray_patch, cv2.MORPH_BLACKHAT, kernel)
    blackhat_threshold = robust_threshold(
        blackhat,
        core_mask,
        blackhat_sigma,
        3.0,
    )
    blackhat_mask = (
        (blackhat >= blackhat_threshold) & (core_mask > 0)
    ).astype(np.uint8) * 255

    if previous_patch is None:
        temporal = np.zeros_like(gray_patch)
        temporal_threshold = 0.0
        dark_threshold = 0.0
        motion_dark_mask = np.zeros_like(gray_patch)
    else:
        temporal = cv2.absdiff(gray_patch, previous_patch)
        temporal_threshold = robust_threshold(
            temporal,
            core_mask,
            temporal_sigma,
            1.0,
        )
        local_background = cv2.GaussianBlur(gray_patch, (11, 11), 0)
        dark_response = cv2.subtract(local_background, gray_patch)
        dark_threshold = robust_threshold(
            dark_response,
            core_mask,
            dark_sigma,
            1.0,
        )
        motion_dark_mask = (
            (temporal >= temporal_threshold)
            & (dark_response >= dark_threshold)
            & (core_mask > 0)
        ).astype(np.uint8) * 255

    candidate_mask = cv2.bitwise_or(blackhat_mask, motion_dark_mask)
    gradient_x = cv2.Sobel(gray_patch, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_patch, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )
    extracted: list[CandidateData] = []
    half = width / 2.0
    for component_id in range(1, component_count):
        x, y, box_width, box_height, pixel_area = stats[component_id]
        if pixel_area < min_area or pixel_area > max_area:
            continue
        if box_width > max_side or box_height > max_side:
            continue

        component_mask = np.zeros_like(gray_patch)
        component_mask[labels == component_id] = 255
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        circularity = (
            4.0 * math.pi * contour_area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        solidity = contour_area / hull_area if hull_area > 0 else 0.0
        extent = float(pixel_area) / max(box_width * box_height, 1)
        center_x, center_y = centroids[component_id]

        component_pixels = labels == component_id
        gray_values = gray_patch[component_pixels].astype(np.float32)
        blackhat_values = blackhat[component_pixels].astype(np.float32)
        temporal_values = temporal[component_pixels].astype(np.float32)
        gradient_values = gradient[component_pixels]

        surround = np.zeros_like(gray_patch)
        x1 = max(0, x - 4)
        y1 = max(0, y - 4)
        x2 = min(width, x + box_width + 4)
        y2 = min(height, y + box_height + 4)
        surround[y1:y2, x1:x2] = 255
        surround[component_pixels] = 0
        surround[core_mask == 0] = 0
        surround_values = gray_patch[surround > 0].astype(np.float32)
        local_background_mean = (
            float(np.mean(surround_values))
            if len(surround_values)
            else float(np.mean(gray_values))
        )
        object_mean = float(np.mean(gray_values))
        local_contrast = local_background_mean - object_mean

        texture_patch = local_patch(
            gray_patch,
            center_x,
            center_y,
            max(9, min(patch_size, 17)),
        )
        local_entropy = entropy(texture_patch.reshape(-1))
        local_laplacian_variance = float(
            cv2.Laplacian(texture_patch, cv2.CV_32F).var()
        )
        radial_distance = math.hypot(center_x - half, center_y - half)
        radial_norm = radial_distance / max(core_radius_px, 1)
        blackhat_mean = float(np.mean(blackhat_values))
        temporal_mean = float(np.mean(temporal_values))
        blackhat_strength = np.clip(
            (blackhat_mean - blackhat_threshold)
            / max(blackhat_threshold, 1.0),
            0.0,
            2.0,
        )
        temporal_strength = (
            np.clip(
                temporal_mean / max(temporal_threshold, 1.0),
                0.0,
                2.0,
            )
            if previous_patch is not None
            else 0.0
        )
        persistence_prior = 0.25 if previous_patch is None else 0.5
        proposal_score = float(
            np.clip(
                0.28 * min(blackhat_strength, 1.0)
                + 0.30 * min(temporal_strength, 1.0)
                + 0.16 * np.clip(local_contrast / 25.0, 0.0, 1.0)
                + 0.11 * np.clip(circularity, 0.0, 1.0)
                + 0.10 * (1.0 - np.clip(radial_norm, 0.0, 1.0))
                + 0.05 * persistence_prior,
                0.0,
                1.0,
            )
        )
        motion_pixels = int(
            np.count_nonzero(motion_dark_mask[component_pixels])
        )
        blackhat_pixels = int(
            np.count_nonzero(blackhat_mask[component_pixels])
        )
        if motion_pixels and blackhat_pixels:
            proposal_source = "blackhat+motion"
        elif motion_pixels:
            proposal_source = "motion_dark"
        else:
            proposal_source = "blackhat"

        candidate_id = (
            f"{video_slug}_f{frame_index:06d}_"
            f"c{component_id:02d}"
        )
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "video_slug": video_slug,
            "source_video": video_name,
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "droplet_sequence": sequence_id,
            "center_x_px": float(center_x),
            "center_y_px": float(center_y),
            "relative_x": float((center_x - half) / max(core_radius_px, 1)),
            "relative_y": float((center_y - half) / max(core_radius_px, 1)),
            "radial_distance_px": radial_distance,
            "radial_distance_norm": radial_norm,
            "droplet_radius_px": droplet_radius_px,
            "core_radius_px": core_radius_px,
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_width": int(box_width),
            "bbox_height": int(box_height),
            "pixel_area": int(pixel_area),
            "contour_area": contour_area,
            "perimeter": perimeter,
            "equivalent_diameter": math.sqrt(
                4.0 * max(contour_area, 0.0) / math.pi
            ),
            "aspect_ratio": float(box_width / max(box_height, 1)),
            "circularity": circularity,
            "solidity": solidity,
            "extent": extent,
            "object_mean_gray": object_mean,
            "object_std_gray": float(np.std(gray_values)),
            "object_min_gray": float(np.min(gray_values)),
            "object_max_gray": float(np.max(gray_values)),
            "local_background_mean_gray": local_background_mean,
            "local_contrast_gray": local_contrast,
            "blackhat_mean": blackhat_mean,
            "blackhat_max": float(np.max(blackhat_values)),
            "blackhat_threshold": blackhat_threshold,
            "temporal_mean": temporal_mean,
            "temporal_max": float(np.max(temporal_values)),
            "temporal_threshold": temporal_threshold,
            "gradient_mean": float(np.mean(gradient_values)),
            "gradient_max": float(np.max(gradient_values)),
            "local_entropy": local_entropy,
            "local_laplacian_variance": local_laplacian_variance,
            "proposal_score": proposal_score,
            "proposal_source": proposal_source,
            "motion_pixel_count": motion_pixels,
            "blackhat_pixel_count": blackhat_pixels,
            "ground_truth_class": "",
            "label_status": "unlabeled",
            "proposal_only": 1,
        }
        extracted.append(
            CandidateData(
                row=row,
                patch=local_patch(
                    gray_patch,
                    center_x,
                    center_y,
                    patch_size,
                ),
            )
        )

    extracted.sort(
        key=lambda item: float(item.row["proposal_score"]),
        reverse=True,
    )
    return extracted[:max_candidates], {
        "blackhat_threshold": blackhat_threshold,
        "temporal_threshold": temporal_threshold,
        "dark_threshold": dark_threshold,
        "blackhat_pixels": float(np.count_nonzero(blackhat_mask)),
        "motion_dark_pixels": float(np.count_nonzero(motion_dark_mask)),
    }
