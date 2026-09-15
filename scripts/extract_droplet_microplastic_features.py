#!/usr/bin/env python3
"""Extract unlabeled droplet, particle-candidate, and temporal-track features."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_one_droplet_frames_clustered_v6 import (  # noqa: E402
    calibrate_dominant_circle,
)
from scripts.extract_one_droplet_frames_autocalibrated_v4 import (  # noqa: E402
    hough_candidates,
)
from scripts.run_microplastic_one_droplet_hybrid import (  # noqa: E402
    DropletObservation,
    DropletSequenceTracker,
    Rect,
    centered_crop,
    crop_rect,
    robust_threshold,
)


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass
class CandidateData:
    row: dict[str, object]
    patch: np.ndarray


@dataclass
class CandidateTrack:
    track_id: int
    sequence_id: int
    first_frame: int
    last_frame: int
    x: float
    y: float
    missed: int = 0
    observations: list[dict[str, object]] = field(default_factory=list)
    positions: list[tuple[int, float, float]] = field(default_factory=list)
    best_score: float = -1.0
    best_patch: np.ndarray | None = field(default=None, repr=False)
    best_frame: int = -1


class CandidateFeatureTracker:
    def __init__(
        self,
        *,
        max_distance: float,
        max_missed: int,
    ) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_track_id = 1
        self.active: list[CandidateTrack] = []

    def update(
        self,
        candidates: list[CandidateData],
        *,
        frame_index: int,
        sequence_id: int,
    ) -> tuple[list[CandidateTrack], list[int]]:
        for track in self.active:
            track.missed += 1

        matches: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.active):
            if track.sequence_id != sequence_id:
                continue
            for candidate_index, candidate in enumerate(candidates):
                distance = math.hypot(
                    track.x - float(candidate.row["center_x_px"]),
                    track.y - float(candidate.row["center_y_px"]),
                )
                if distance <= self.max_distance:
                    matches.append((distance, track_index, candidate_index))
        matches.sort()

        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        assignments = [-1] * len(candidates)
        for _, track_index, candidate_index in matches:
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            track = self.active[track_index]
            candidate = candidates[candidate_index]
            center_x = float(candidate.row["center_x_px"])
            center_y = float(candidate.row["center_y_px"])
            track.x = center_x
            track.y = center_y
            track.last_frame = frame_index
            track.missed = 0
            track.observations.append(candidate.row)
            track.positions.append((frame_index, center_x, center_y))
            score = float(candidate.row["proposal_score"])
            if score > track.best_score:
                track.best_score = score
                track.best_patch = candidate.patch.copy()
                track.best_frame = frame_index
            assignments[candidate_index] = track.track_id
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidates:
                continue
            center_x = float(candidate.row["center_x_px"])
            center_y = float(candidate.row["center_y_px"])
            score = float(candidate.row["proposal_score"])
            track = CandidateTrack(
                track_id=self.next_track_id,
                sequence_id=sequence_id,
                first_frame=frame_index,
                last_frame=frame_index,
                x=center_x,
                y=center_y,
                observations=[candidate.row],
                positions=[(frame_index, center_x, center_y)],
                best_score=score,
                best_patch=candidate.patch.copy(),
                best_frame=frame_index,
            )
            self.active.append(track)
            assignments[candidate_index] = track.track_id
            self.next_track_id += 1

        finished = [
            track for track in self.active if track.missed > self.max_missed
        ]
        self.active = [
            track for track in self.active if track.missed <= self.max_missed
        ]
        return finished, assignments

    def reset(self) -> list[CandidateTrack]:
        finished = self.active
        self.active = []
        return finished


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-roi-x", type=int, default=560)
    parser.add_argument("--search-roi-y", type=int, default=0)
    parser.add_argument("--search-roi-width", type=int, default=320)
    parser.add_argument("--search-roi-height", type=int, default=360)
    parser.add_argument("--processing-size", type=int, default=128)
    parser.add_argument("--candidate-patch-size", type=int, default=32)
    parser.add_argument("--core-radius-ratio", type=float, default=0.72)
    parser.add_argument("--blackhat-kernel", type=int, default=7)
    parser.add_argument("--blackhat-sigma", type=float, default=3.2)
    parser.add_argument("--temporal-sigma", type=float, default=2.5)
    parser.add_argument("--particle-min-area", type=int, default=2)
    parser.add_argument("--particle-max-area", type=int, default=100)
    parser.add_argument("--particle-max-side", type=int, default=18)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--track-max-distance", type=float, default=10.0)
    parser.add_argument("--track-max-missed", type=int, default=2)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    return parser.parse_args()


def slugify(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")


def entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    histogram, _ = np.histogram(values, bins=64, range=(0, 256))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(probabilities.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def masked_stats(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = image[mask > 0].astype(np.float32)
    if not len(values):
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "contrast": 0.0,
            "entropy": 0.0,
        }
    p10, p90 = np.percentile(values, [10, 90])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p10": float(p10),
        "p90": float(p90),
        "contrast": float(p90 - p10),
        "entropy": entropy(values),
    }


def disk_feature_masks(
    shape: tuple[int, int],
    *,
    center_x: float,
    center_y: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interior = np.zeros(shape, dtype=np.uint8)
    core = np.zeros(shape, dtype=np.uint8)
    ring = np.zeros(shape, dtype=np.uint8)
    center = (int(round(center_x)), int(round(center_y)))
    cv2.circle(interior, center, max(1, int(round(radius * 0.86))), 255, -1)
    cv2.circle(core, center, max(1, int(round(radius * 0.55))), 255, -1)
    cv2.circle(ring, center, max(1, int(round(radius * 1.08))), 255, -1)
    cv2.circle(ring, center, max(1, int(round(radius * 0.88))), 0, -1)
    return interior, core, ring


def droplet_features(
    gray_frame: np.ndarray,
    *,
    video_slug: str,
    video_name: str,
    frame_index: int,
    timestamp_sec: float,
    sequence_id: int,
    center_x: float,
    center_y: float,
    radius: float,
    previous_center: tuple[float, float, float] | None,
) -> dict[str, object]:
    interior, core, ring = disk_feature_masks(
        gray_frame.shape,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )
    interior_stats = masked_stats(gray_frame, interior)
    core_stats = masked_stats(gray_frame, core)
    gradient_x = cv2.Sobel(gray_frame, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_frame, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    ring_values = gradient[ring > 0]
    ring_mean = float(np.mean(ring_values)) if len(ring_values) else 0.0
    ring_std = float(np.std(ring_values)) if len(ring_values) else 0.0
    ring_threshold = (
        float(np.percentile(ring_values, 65)) if len(ring_values) else 0.0
    )
    ring_coverage = (
        float(np.mean(ring_values >= ring_threshold))
        if len(ring_values)
        else 0.0
    )
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_32F)
    laplacian_values = laplacian[interior > 0]
    focus_variance = (
        float(np.var(laplacian_values)) if len(laplacian_values) else 0.0
    )

    displacement = 0.0
    radius_change = 0.0
    if previous_center is not None:
        displacement = math.hypot(
            center_x - previous_center[0],
            center_y - previous_center[1],
        )
        radius_change = radius - previous_center[2]

    return {
        "video_slug": video_slug,
        "source_video": video_name,
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "droplet_sequence": sequence_id,
        "center_x_px": center_x,
        "center_y_px": center_y,
        "radius_px": radius,
        "diameter_px": 2.0 * radius,
        "area_px2": math.pi * radius * radius,
        "circumference_px": 2.0 * math.pi * radius,
        "center_displacement_px": displacement,
        "radius_change_px": radius_change,
        "interior_mean_gray": interior_stats["mean"],
        "interior_std_gray": interior_stats["std"],
        "interior_median_gray": interior_stats["median"],
        "interior_p10_gray": interior_stats["p10"],
        "interior_p90_gray": interior_stats["p90"],
        "interior_contrast_gray": interior_stats["contrast"],
        "interior_entropy": interior_stats["entropy"],
        "core_mean_gray": core_stats["mean"],
        "core_std_gray": core_stats["std"],
        "core_contrast_gray": core_stats["contrast"],
        "core_entropy": core_stats["entropy"],
        "ring_gradient_mean": ring_mean,
        "ring_gradient_std": ring_std,
        "ring_gradient_coverage": ring_coverage,
        "focus_laplacian_variance": focus_variance,
        "candidate_count": 0,
        "ground_truth_class": "",
        "label_status": "unlabeled",
    }


def local_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    size: int,
) -> np.ndarray:
    return centered_crop(image, center_x, center_y, size)


def extract_candidates(
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
) -> tuple[list[CandidateData], dict[str, float]]:
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
    candidate_mask = np.zeros_like(gray_patch)
    candidate_mask[
        (blackhat >= blackhat_threshold) & (core_mask > 0)
    ] = 255

    if previous_patch is None:
        temporal = np.zeros_like(gray_patch)
        temporal_threshold = 0.0
    else:
        temporal = cv2.absdiff(gray_patch, previous_patch)
        temporal_threshold = robust_threshold(
            temporal,
            core_mask,
            temporal_sigma,
            2.0,
        )

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
                0.34 * min(blackhat_strength, 1.0)
                + 0.22 * min(temporal_strength, 1.0)
                + 0.16 * np.clip(local_contrast / 25.0, 0.0, 1.0)
                + 0.13 * np.clip(circularity, 0.0, 1.0)
                + 0.10 * (1.0 - np.clip(radial_norm, 0.0, 1.0))
                + 0.05 * persistence_prior,
                0.0,
                1.0,
            )
        )

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
    }


AGGREGATE_FEATURES = [
    "proposal_score",
    "pixel_area",
    "contour_area",
    "circularity",
    "solidity",
    "local_contrast_gray",
    "blackhat_mean",
    "blackhat_max",
    "temporal_mean",
    "temporal_max",
    "gradient_mean",
    "local_entropy",
    "local_laplacian_variance",
    "radial_distance_norm",
]


def summarize_track(
    track: CandidateTrack,
    *,
    video_slug: str,
    video_name: str,
    patch_dir: Path,
) -> dict[str, object]:
    positions = track.positions
    steps = [
        math.hypot(
            positions[index][1] - positions[index - 1][1],
            positions[index][2] - positions[index - 1][2],
        )
        for index in range(1, len(positions))
    ]
    displacement = (
        math.hypot(
            positions[-1][1] - positions[0][1],
            positions[-1][2] - positions[0][2],
        )
        if len(positions) > 1
        else 0.0
    )
    duration_frames = track.last_frame - track.first_frame + 1
    track_name = (
        f"{video_slug}_seq{track.sequence_id:04d}_"
        f"track{track.track_id:06d}"
    )
    patch_path = patch_dir / f"{track_name}.png"
    if track.best_patch is not None:
        cv2.imwrite(str(patch_path), track.best_patch)

    row: dict[str, object] = {
        "track_key": track_name,
        "track_id": track.track_id,
        "video_slug": video_slug,
        "source_video": video_name,
        "droplet_sequence": track.sequence_id,
        "first_frame": track.first_frame,
        "last_frame": track.last_frame,
        "best_frame": track.best_frame,
        "hits": len(track.observations),
        "duration_frames": duration_frames,
        "persistence_ratio": len(track.observations) / max(duration_frames, 1),
        "start_x_px": positions[0][1],
        "start_y_px": positions[0][2],
        "end_x_px": positions[-1][1],
        "end_y_px": positions[-1][2],
        "displacement_px": displacement,
        "path_length_px": float(sum(steps)),
        "mean_speed_px_per_frame": float(np.mean(steps)) if steps else 0.0,
        "max_speed_px_per_frame": max(steps) if steps else 0.0,
        "best_proposal_score": track.best_score,
        "temporal_status": (
            "persistent" if len(track.observations) >= 3 else "transient"
        ),
        "best_patch": str(patch_path.relative_to(patch_dir.parent)),
        "ground_truth_class": "",
        "label_status": "unlabeled",
        "proposal_only": 1,
    }
    for feature in AGGREGATE_FEATURES:
        values = np.asarray(
            [float(observation[feature]) for observation in track.observations],
            dtype=np.float64,
        )
        row[f"{feature}_mean"] = float(np.mean(values))
        row[f"{feature}_std"] = float(np.std(values))
        row[f"{feature}_max"] = float(np.max(values))
    return row


def aggregate_sequences(
    droplets: pd.DataFrame,
    candidates: pd.DataFrame,
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["video_slug", "source_video", "droplet_sequence"]
    for keys, group in droplets.groupby(group_columns, sort=True):
        video_slug, source_video, sequence_id = keys
        sequence_candidates = candidates[
            (candidates["video_slug"] == video_slug)
            & (candidates["droplet_sequence"] == sequence_id)
        ]
        sequence_tracks = tracks[
            (tracks["video_slug"] == video_slug)
            & (tracks["droplet_sequence"] == sequence_id)
        ]
        rows.append(
            {
                "video_slug": video_slug,
                "source_video": source_video,
                "droplet_sequence": sequence_id,
                "first_frame": int(group["frame_index"].min()),
                "last_frame": int(group["frame_index"].max()),
                "frames_observed": len(group),
                "radius_mean_px": float(group["radius_px"].mean()),
                "radius_std_px": float(group["radius_px"].std(ddof=0)),
                "radius_min_px": float(group["radius_px"].min()),
                "radius_max_px": float(group["radius_px"].max()),
                "interior_mean_gray": float(
                    group["interior_mean_gray"].mean()
                ),
                "interior_std_gray": float(
                    group["interior_std_gray"].mean()
                ),
                "interior_contrast_gray": float(
                    group["interior_contrast_gray"].mean()
                ),
                "interior_entropy": float(group["interior_entropy"].mean()),
                "focus_laplacian_variance": float(
                    group["focus_laplacian_variance"].mean()
                ),
                "candidate_observations": len(sequence_candidates),
                "candidate_tracks": len(sequence_tracks),
                "persistent_candidate_tracks": int(
                    (sequence_tracks["hits"] >= 3).sum()
                ),
                "max_candidate_score": (
                    float(sequence_tracks["best_proposal_score"].max())
                    if len(sequence_tracks)
                    else 0.0
                ),
                "ground_truth_particle_count": "",
                "label_status": "unlabeled",
            }
        )
    return pd.DataFrame(rows)


def feature_schema() -> dict[str, object]:
    return {
        "status": (
            "All particle rows are unlabeled proposals. They must not be used "
            "as ground truth before Roboflow matching or manual review."
        ),
        "tables": {
            "droplet_frame_features.csv": (
                "One row per accurately localized droplet frame."
            ),
            "particle_candidate_features.csv": (
                "One row per image-processing candidate observation."
            ),
            "particle_track_features.csv": (
                "Temporal aggregates and one best 32x32 patch per track."
            ),
            "droplet_sequence_features.csv": (
                "One row per droplet passage with candidate-track counts."
            ),
        },
        "key_fields": [
            "video_slug",
            "source_video",
            "frame_index",
            "droplet_sequence",
            "track_id",
        ],
        "fpga_friendly_features": [
            "pixel_area",
            "aspect_ratio",
            "circularity",
            "solidity",
            "local_contrast_gray",
            "blackhat_mean",
            "temporal_mean",
            "radial_distance_norm",
            "hits",
            "persistence_ratio",
            "mean_speed_px_per_frame",
        ],
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    patch_dir = output / "track_patches32"
    patch_dir.mkdir(parents=True, exist_ok=True)

    search_roi = Rect(
        args.search_roi_x,
        args.search_roi_y,
        args.search_roi_width,
        args.search_roi_height,
    )
    videos = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise RuntimeError(f"No videos found in {source_dir}")

    droplet_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    video_summaries: list[dict[str, object]] = []

    for video_index, video_path in enumerate(videos, start=1):
        video_slug = f"v{video_index:02d}_{slugify(video_path.stem)}"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        search_roi.validate(width, height)
        calibrated_x, calibrated_radius, cluster_size = calibrate_dominant_circle(
            capture,
            frame_count=frame_count,
            search_roi=search_roi,
        )

        sequence_tracker = DropletSequenceTracker(
            new_droplet_jump=55.0,
            max_missing=3,
        )
        candidate_tracker = CandidateFeatureTracker(
            max_distance=args.track_max_distance,
            max_missed=args.track_max_missed,
        )
        previous_patch: np.ndarray | None = None
        previous_sequence = 0
        previous_center: tuple[float, float, float] | None = None
        localized_frames = 0
        video_candidate_count = 0
        video_track_count = 0

        min_radius = max(22, int(round(calibrated_radius - 11)))
        max_radius = min(94, int(round(calibrated_radius + 11)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        while True:
            if (
                args.max_frames_per_video
                and frame_index >= args.max_frames_per_video
            ):
                break
            ok, frame = capture.read()
            if not ok:
                break

            roi = crop_rect(frame, search_roi)
            circles = hough_candidates(
                roi,
                min_radius=min_radius,
                max_radius=max_radius,
                accumulator_threshold=25,
            )
            matches = [
                circle
                for circle in circles
                if abs(circle[0] - calibrated_x) <= 14.0
                and abs(circle[2] - calibrated_radius) <= 10.0
            ]
            selected = (
                min(
                    matches,
                    key=lambda item: (
                        abs(item[0] - calibrated_x)
                        + 1.1 * abs(item[2] - calibrated_radius)
                    ),
                )
                if matches
                else None
            )
            raw_observation = None
            global_circle = None
            if selected is not None:
                local_x, local_y, radius = selected
                global_x = search_roi.x + local_x
                global_y = search_roi.y + local_y
                half = args.processing_size / 2.0
                if (
                    global_x >= half
                    and global_x < width - half
                    and global_y >= half
                    and global_y < height - half
                ):
                    global_circle = (global_x, global_y, radius)
                    raw_observation = DropletObservation(
                        center_x=local_x,
                        center_y=local_y,
                        radius=radius,
                        area=float(math.pi * radius * radius),
                        score=radius,
                        bbox=(
                            int(round(local_x - radius)),
                            int(round(local_y - radius)),
                            int(round(2 * radius)),
                            int(round(2 * radius)),
                        ),
                    )

            _, is_new_sequence = sequence_tracker.update(
                raw_observation,
                frame_index,
            )
            if is_new_sequence and previous_sequence:
                for finished in candidate_tracker.reset():
                    track_rows.append(
                        summarize_track(
                            finished,
                            video_slug=video_slug,
                            video_name=video_path.name,
                            patch_dir=patch_dir,
                        )
                    )
                    video_track_count += 1
                previous_patch = None
                previous_center = None

            current_sequence = sequence_tracker.sequence_id
            if global_circle is not None:
                global_x, global_y, radius = global_circle
                localized_frames += 1
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp_sec = frame_index / fps if fps > 0 else 0.0
                droplet_row = droplet_features(
                    gray_frame,
                    video_slug=video_slug,
                    video_name=video_path.name,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    sequence_id=current_sequence,
                    center_x=global_x,
                    center_y=global_y,
                    radius=radius,
                    previous_center=previous_center,
                )
                processing_patch = centered_crop(
                    gray_frame,
                    global_x,
                    global_y,
                    args.processing_size,
                )
                core_radius = min(
                    args.processing_size // 2 - 4,
                    max(10, int(round(radius * args.core_radius_ratio))),
                )
                candidates, thresholds = extract_candidates(
                    processing_patch,
                    previous_patch,
                    video_slug=video_slug,
                    video_name=video_path.name,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    sequence_id=current_sequence,
                    droplet_radius_px=radius,
                    core_radius_px=core_radius,
                    patch_size=args.candidate_patch_size,
                    blackhat_kernel=args.blackhat_kernel,
                    blackhat_sigma=args.blackhat_sigma,
                    temporal_sigma=args.temporal_sigma,
                    min_area=args.particle_min_area,
                    max_area=args.particle_max_area,
                    max_side=args.particle_max_side,
                    max_candidates=args.max_candidates,
                )
                finished_tracks, assignments = candidate_tracker.update(
                    candidates,
                    frame_index=frame_index,
                    sequence_id=current_sequence,
                )
                for candidate, track_id in zip(candidates, assignments):
                    candidate.row["track_id"] = track_id
                    candidate_rows.append(candidate.row)
                for finished in finished_tracks:
                    track_rows.append(
                        summarize_track(
                            finished,
                            video_slug=video_slug,
                            video_name=video_path.name,
                            patch_dir=patch_dir,
                        )
                    )
                    video_track_count += 1
                droplet_row["candidate_count"] = len(candidates)
                droplet_row["blackhat_threshold"] = thresholds[
                    "blackhat_threshold"
                ]
                droplet_row["temporal_threshold"] = thresholds[
                    "temporal_threshold"
                ]
                droplet_rows.append(droplet_row)
                video_candidate_count += len(candidates)
                previous_patch = processing_patch
                previous_center = (global_x, global_y, radius)
                previous_sequence = current_sequence
            else:
                previous_patch = None
                previous_center = None
                if frame_index - sequence_tracker.last_frame > 3:
                    for finished in candidate_tracker.reset():
                        track_rows.append(
                            summarize_track(
                                finished,
                                video_slug=video_slug,
                                video_name=video_path.name,
                                patch_dir=patch_dir,
                            )
                        )
                        video_track_count += 1
            frame_index += 1

        for finished in candidate_tracker.reset():
            track_rows.append(
                summarize_track(
                    finished,
                    video_slug=video_slug,
                    video_name=video_path.name,
                    patch_dir=patch_dir,
                )
            )
            video_track_count += 1
        capture.release()
        video_summaries.append(
            {
                "video_slug": video_slug,
                "source_video": str(video_path),
                "frames": frame_count,
                "fps": fps,
                "localized_droplet_frames": localized_frames,
                "droplet_sequences": sequence_tracker.sequence_id,
                "candidate_observations": video_candidate_count,
                "candidate_tracks": video_track_count,
                "calibrated_center_x_in_search_roi": calibrated_x,
                "calibrated_radius": calibrated_radius,
                "calibration_cluster_size": cluster_size,
            }
        )
        print(
            f"{video_path.name}: droplets={localized_frames} "
            f"candidates={video_candidate_count} tracks={video_track_count}"
        )

    droplets = pd.DataFrame(droplet_rows)
    candidates = pd.DataFrame(candidate_rows)
    tracks = pd.DataFrame(track_rows)
    if droplets.empty or candidates.empty or tracks.empty:
        raise RuntimeError("Feature extraction produced an empty table")
    sequences = aggregate_sequences(droplets, candidates, tracks)

    droplets.to_csv(output / "droplet_frame_features.csv", index=False)
    candidates.to_csv(output / "particle_candidate_features.csv", index=False)
    tracks.to_csv(output / "particle_track_features.csv", index=False)
    sequences.to_csv(output / "droplet_sequence_features.csv", index=False)
    (output / "feature_schema.json").write_text(
        json.dumps(feature_schema(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "source_directory": str(source_dir),
        "output_directory": str(output),
        "status": (
            "Unlabeled feature dataset. Candidate proposals are not "
            "microplastic ground truth."
        ),
        "configuration": {
            "search_roi": {
                "x": search_roi.x,
                "y": search_roi.y,
                "width": search_roi.width,
                "height": search_roi.height,
            },
            "processing_size": args.processing_size,
            "candidate_patch_size": args.candidate_patch_size,
            "core_radius_ratio": args.core_radius_ratio,
            "blackhat_kernel": args.blackhat_kernel,
            "blackhat_sigma": args.blackhat_sigma,
            "temporal_sigma": args.temporal_sigma,
            "max_candidates_per_frame": args.max_candidates,
        },
        "counts": {
            "droplet_frame_rows": len(droplets),
            "particle_candidate_rows": len(candidates),
            "particle_track_rows": len(tracks),
            "persistent_particle_track_rows": int((tracks["hits"] >= 3).sum()),
            "droplet_sequence_rows": len(sequences),
            "track_patch_files": len(list(patch_dir.glob("*.png"))),
        },
        "videos": video_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        (
            "# Unlabeled droplet and microplastic feature dataset\n\n"
            "This dataset can be used for clustering and feature inspection now. "
            "It must not be used for supervised particle/background training "
            "until Roboflow boxes or manual labels are matched.\n\n"
            "The key join fields are `source_video`, `frame_index`, "
            "`droplet_sequence`, and `track_id`.\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"], indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
