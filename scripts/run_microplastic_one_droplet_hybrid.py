"""Fast one-droplet microplastic candidate pipeline.

This is an independent research branch. It does not load, overwrite, or
retrain the existing YOLO/QNN detector checkpoints.

The pipeline is deliberately FPGA-friendly:

    fixed acquisition ROI -> background difference -> one-droplet tracker
    -> small stabilized patch -> black-hat candidate extraction
    -> temporal confirmation -> review patches

The temporal stage is important because the target particles are only a few
pixels wide. A single-frame response is treated as uncertain until it is seen
again at a nearby droplet-relative position.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def validate(self, frame_width: int, frame_height: int) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y must be non-negative")
        if self.x2 > frame_width or self.y2 > frame_height:
            raise ValueError(
                f"ROI {self} exceeds source frame {frame_width}x{frame_height}"
            )


@dataclass
class DropletObservation:
    center_x: float
    center_y: float
    radius: float
    area: float
    score: float
    bbox: tuple[int, int, int, int]


@dataclass
class ParticleCandidate:
    x: float
    y: float
    width: int
    height: int
    area: int
    contrast: float
    temporal: float
    score: float
    bbox: tuple[int, int, int, int]


@dataclass
class ParticleTrack:
    track_id: int
    x: float
    y: float
    first_frame: int
    last_frame: int
    hits: int = 1
    missed: int = 0
    score_sum: float = 0.0
    max_score: float = 0.0
    best_frame: int = -1
    best_x: float = 0.0
    best_y: float = 0.0
    best_patch: np.ndarray | None = field(default=None, repr=False)

    @property
    def mean_score(self) -> float:
        return self.score_sum / max(self.hits, 1)


@dataclass
class VisibleParticle:
    track_id: int
    x: float
    y: float
    hits: int
    confidence: float
    confirmed: bool
    bbox: tuple[int, int, int, int]


class TemporalParticleTracker:
    def __init__(
        self,
        *,
        max_distance: float,
        min_hits: int,
        max_missed: int,
        confidence_threshold: float,
        patch_size: int,
    ) -> None:
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.confidence_threshold = confidence_threshold
        self.patch_size = patch_size
        self.next_track_id = 1
        self.tracks: list[ParticleTrack] = []

    def reset(self) -> list[ParticleTrack]:
        finished = self.tracks
        self.tracks = []
        return finished

    def _track_confidence(self, track: ParticleTrack) -> float:
        persistence = min(track.hits / max(self.min_hits + 1, 1), 1.0)
        return float(np.clip(0.65 * track.mean_score + 0.35 * persistence, 0, 1))

    def _candidate_patch(
        self,
        gray_patch: np.ndarray,
        candidate: ParticleCandidate,
    ) -> np.ndarray:
        half = self.patch_size // 2
        cx = int(round(candidate.x))
        cy = int(round(candidate.y))
        padded = cv2.copyMakeBorder(
            gray_patch,
            half,
            half,
            half,
            half,
            cv2.BORDER_REFLECT_101,
        )
        crop = padded[cy : cy + self.patch_size, cx : cx + self.patch_size]
        if crop.shape != (self.patch_size, self.patch_size):
            return cv2.resize(
                crop,
                (self.patch_size, self.patch_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return crop.copy()

    def update(
        self,
        candidates: list[ParticleCandidate],
        gray_patch: np.ndarray,
        frame_index: int,
    ) -> tuple[list[VisibleParticle], list[ParticleTrack]]:
        for track in self.tracks:
            track.missed += 1

        possible_matches: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for candidate_index, candidate in enumerate(candidates):
                distance = math.hypot(track.x - candidate.x, track.y - candidate.y)
                if distance <= self.max_distance:
                    possible_matches.append(
                        (distance, track_index, candidate_index)
                    )
        possible_matches.sort()

        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        for _, track_index, candidate_index in possible_matches:
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_index]
            candidate = candidates[candidate_index]
            alpha = 0.7
            track.x = alpha * candidate.x + (1.0 - alpha) * track.x
            track.y = alpha * candidate.y + (1.0 - alpha) * track.y
            track.last_frame = frame_index
            track.hits += 1
            track.missed = 0
            track.score_sum += candidate.score
            if candidate.score > track.max_score:
                track.max_score = candidate.score
                track.best_frame = frame_index
                track.best_x = candidate.x
                track.best_y = candidate.y
                track.best_patch = self._candidate_patch(gray_patch, candidate)
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidates:
                continue
            self.tracks.append(
                ParticleTrack(
                    track_id=self.next_track_id,
                    x=candidate.x,
                    y=candidate.y,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    hits=1,
                    score_sum=candidate.score,
                    max_score=candidate.score,
                    best_frame=frame_index,
                    best_x=candidate.x,
                    best_y=candidate.y,
                    best_patch=self._candidate_patch(gray_patch, candidate),
                )
            )
            self.next_track_id += 1

        finished = [
            track for track in self.tracks if track.missed > self.max_missed
        ]
        self.tracks = [
            track for track in self.tracks if track.missed <= self.max_missed
        ]

        visible: list[VisibleParticle] = []
        for track in self.tracks:
            if track.missed != 0:
                continue
            confidence = self._track_confidence(track)
            half_box = 6
            visible.append(
                VisibleParticle(
                    track_id=track.track_id,
                    x=track.x,
                    y=track.y,
                    hits=track.hits,
                    confidence=confidence,
                    confirmed=(
                        track.hits >= self.min_hits
                        and confidence >= self.confidence_threshold
                    ),
                    bbox=(
                        int(round(track.x)) - half_box,
                        int(round(track.y)) - half_box,
                        half_box * 2,
                        half_box * 2,
                    ),
                )
            )
        return visible, finished

    def is_confirmed(self, track: ParticleTrack) -> bool:
        return (
            track.hits >= self.min_hits
            and self._track_confidence(track) >= self.confidence_threshold
        )


class DropletSequenceTracker:
    def __init__(
        self,
        *,
        new_droplet_jump: float,
        max_missing: int,
    ) -> None:
        self.new_droplet_jump = new_droplet_jump
        self.max_missing = max_missing
        self.sequence_id = 0
        self.last_frame = -1
        self.last_y: float | None = None
        self.filtered_x: float | None = None
        self.filtered_y: float | None = None
        self.velocity = 0.0

    def update(
        self,
        observation: DropletObservation | None,
        frame_index: int,
    ) -> tuple[DropletObservation | None, bool]:
        if observation is None:
            return None, False

        gap = (
            frame_index - self.last_frame
            if self.last_frame >= 0
            else self.max_missing + 1
        )
        is_new = (
            self.last_y is None
            or gap > self.max_missing
            or observation.center_y - self.last_y > self.new_droplet_jump
        )

        if is_new:
            self.sequence_id += 1
            self.filtered_x = observation.center_x
            self.filtered_y = observation.center_y
            self.velocity = 0.0
        else:
            assert self.filtered_x is not None
            assert self.filtered_y is not None
            self.filtered_x = 0.75 * observation.center_x + 0.25 * self.filtered_x
            predicted = self.filtered_y + self.velocity * gap
            filtered = 0.75 * observation.center_y + 0.25 * predicted
            measured_velocity = (filtered - self.filtered_y) / max(gap, 1)
            self.velocity = 0.65 * self.velocity + 0.35 * measured_velocity
            self.filtered_y = filtered

        self.last_frame = frame_index
        self.last_y = observation.center_y
        observation.center_x = float(self.filtered_x)
        observation.center_y = float(self.filtered_y)
        return observation, is_new


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track one droplet, detect tiny particle candidates in its core, "
            "and export timing plus review artifacts."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-x", type=int, default=570)
    parser.add_argument("--roi-y", type=int, default=0)
    parser.add_argument("--roi-width", type=int, default=310)
    parser.add_argument("--roi-height", type=int, default=350)
    parser.add_argument(
        "--show-search-roi",
        action="store_true",
        help="Overlay the fixed acquisition ROI in diagnostic output.",
    )
    parser.add_argument(
        "--channel-center-x",
        type=float,
        default=-1,
        help="Channel center inside acquisition ROI; negative uses ROI midpoint.",
    )
    parser.add_argument("--processing-size", type=int, default=144)
    parser.add_argument("--droplet-radius", type=float, default=64.0)
    parser.add_argument("--core-radius-ratio", type=float, default=0.58)
    parser.add_argument("--background-samples", type=int, default=80)
    parser.add_argument("--background-threshold", type=int, default=11)
    parser.add_argument("--droplet-min-area", type=float, default=350.0)
    parser.add_argument("--droplet-max-area", type=float, default=30000.0)
    parser.add_argument("--new-droplet-jump", type=float, default=55.0)
    parser.add_argument("--droplet-max-missing", type=int, default=3)
    parser.add_argument(
        "--feature-kernel",
        "--blackhat-kernel",
        dest="blackhat_kernel",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--particle-polarity",
        choices=("dark", "bright", "both"),
        default="dark",
        help=(
            "Candidate contrast polarity. Use both for high-recall proposals, "
            "or dark/bright for controlled polarity experiments."
        ),
    )
    parser.add_argument("--particle-sigma", type=float, default=3.0)
    parser.add_argument("--particle-min-area", type=int, default=2)
    parser.add_argument("--particle-max-area", type=int, default=90)
    parser.add_argument("--particle-max-side", type=int, default=18)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--particle-max-distance", type=float, default=11.0)
    parser.add_argument("--particle-min-hits", type=int, default=2)
    parser.add_argument("--particle-max-missed", type=int, default=1)
    parser.add_argument("--particle-confidence", type=float, default=0.50)
    parser.add_argument("--review-patch-size", type=int, default=32)
    parser.add_argument("--review-limit", type=int, default=250)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video encoding for a pure algorithm-speed benchmark.",
    )
    return parser.parse_args()


def ensure_odd(value: int, name: str) -> int:
    if value < 3 or value % 2 == 0:
        raise ValueError(f"{name} must be odd and at least 3")
    return value


def crop_rect(frame: np.ndarray, rect: Rect) -> np.ndarray:
    return frame[rect.y : rect.y2, rect.x : rect.x2]


def centered_crop(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    size: int,
) -> np.ndarray:
    half = size // 2
    cx = int(round(center_x))
    cy = int(round(center_y))
    padded = cv2.copyMakeBorder(
        image,
        half,
        half,
        half,
        half,
        cv2.BORDER_REFLECT_101,
    )
    crop = padded[cy : cy + size, cx : cx + size]
    if crop.shape[:2] != (size, size):
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop.copy()


def robust_threshold(
    image: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    floor: float,
) -> float:
    values = image[mask > 0].astype(np.float32)
    if not len(values):
        return floor
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, 1.0)
    return max(floor, median + sigma * robust_sigma)


def build_background(
    capture: cv2.VideoCapture,
    rect: Rect,
    *,
    total_frames: int,
    sample_count: int,
) -> np.ndarray:
    if sample_count < 3:
        raise ValueError("At least three background samples are required")
    upper = max(total_frames - 1, 0)
    positions = np.linspace(0, upper, min(sample_count, max(total_frames, 1)))
    samples: list[np.ndarray] = []
    for position in positions.astype(np.int64):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok:
            continue
        roi = crop_rect(frame, rect)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        samples.append(cv2.GaussianBlur(gray, (3, 3), 0))
    if len(samples) < 3:
        raise RuntimeError("Could not read enough frames to build a background")
    background = np.median(np.stack(samples), axis=0).astype(np.uint8)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return background


def find_droplet(
    gray_roi: np.ndarray,
    background: np.ndarray,
    *,
    threshold: int,
    min_area: float,
    max_area: float,
    channel_center_x: float,
    expected_radius: float,
) -> tuple[DropletObservation | None, np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    difference = cv2.absdiff(blurred, background)
    _, binary = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best: DropletObservation | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 20 or height < 20 or width > 230 or height > 230:
            continue
        center_y = y + height / 2.0
        measured_x = x + width / 2.0
        radius = max(width, height) / 2.0
        square_ratio = min(width, height) / max(width, height)
        x_penalty = abs(measured_x - channel_center_x) / max(expected_radius, 1)
        radius_penalty = abs(radius - expected_radius) / max(expected_radius, 1)
        score = (
            math.log1p(area)
            + 1.8 * square_ratio
            - 1.2 * x_penalty
            - 0.8 * radius_penalty
        )
        candidate = DropletObservation(
            center_x=measured_x,
            center_y=center_y,
            radius=radius,
            area=area,
            score=score,
            bbox=(x, y, width, height),
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best, difference, binary


def align_previous_patch(
    previous: np.ndarray | None,
    current: np.ndarray,
    hanning_window: np.ndarray,
) -> tuple[np.ndarray | None, tuple[float, float], float]:
    if previous is None:
        return None, (0.0, 0.0), 0.0
    shift, response = cv2.phaseCorrelate(
        previous.astype(np.float32),
        current.astype(np.float32),
        hanning_window,
    )
    shift_x = float(np.clip(shift[0], -7.0, 7.0))
    shift_y = float(np.clip(shift[1], -7.0, 7.0))
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned = cv2.warpAffine(
        previous,
        matrix,
        (current.shape[1], current.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, (shift_x, shift_y), float(response)


def find_particle_candidates(
    gray_patch: np.ndarray,
    aligned_previous: np.ndarray | None,
    *,
    core_mask: np.ndarray,
    blackhat_kernel: int,
    sigma: float,
    min_area: int,
    max_area: int,
    max_side: int,
    max_candidates: int,
    polarity: str = "dark",
) -> tuple[list[ParticleCandidate], np.ndarray, np.ndarray, float, float]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (blackhat_kernel, blackhat_kernel),
    )
    if polarity == "dark":
        feature_response = cv2.morphologyEx(
            gray_patch,
            cv2.MORPH_BLACKHAT,
            kernel,
        )
    elif polarity == "bright":
        feature_response = cv2.morphologyEx(
            gray_patch,
            cv2.MORPH_TOPHAT,
            kernel,
        )
    elif polarity == "both":
        dark_response = cv2.morphologyEx(
            gray_patch,
            cv2.MORPH_BLACKHAT,
            kernel,
        )
        bright_response = cv2.morphologyEx(
            gray_patch,
            cv2.MORPH_TOPHAT,
            kernel,
        )
        feature_response = np.maximum(dark_response, bright_response)
    else:
        raise ValueError(f"Unknown particle polarity: {polarity}")
    feature_threshold = robust_threshold(feature_response, core_mask, sigma, 3.0)
    candidate_mask = np.zeros_like(gray_patch)
    candidate_mask[
        (feature_response >= feature_threshold) & (core_mask > 0)
    ] = 255
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), dtype=np.uint8),
    )

    if aligned_previous is None:
        temporal_difference = np.zeros_like(gray_patch)
        temporal_threshold = 255.0
    else:
        temporal_difference = cv2.absdiff(gray_patch, aligned_previous)
        temporal_threshold = robust_threshold(
            temporal_difference,
            core_mask,
            sigma,
            3.0,
        )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )
    candidates: list[ParticleCandidate] = []
    for label_index in range(1, count):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if width > max_side or height > max_side:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        max_aspect = 4.5 if polarity == "dark" else 2.5
        if aspect > max_aspect:
            continue
        component_mask = labels == label_index
        peak = float(feature_response[component_mask].max())
        mean_temporal = float(temporal_difference[component_mask].mean())
        contrast_score = float(
            np.clip(
                (peak - feature_threshold)
                / max(feature_threshold * 1.5, 1.0),
                0,
                1,
            )
        )
        fill_score = float(np.clip(area / max(width * height, 1), 0, 1))
        if aligned_previous is None:
            temporal_score = 0.0
        else:
            temporal_score = float(
                np.clip(
                    mean_temporal / max(temporal_threshold * 1.5, 1.0),
                    0,
                    1,
                )
            )
        score = float(
            np.clip(
                0.60 * contrast_score
                + 0.25 * fill_score
                + 0.15 * temporal_score,
                0,
                1,
            )
        )
        candidates.append(
            ParticleCandidate(
                x=float(centroids[label_index][0]),
                y=float(centroids[label_index][1]),
                width=width,
                height=height,
                area=area,
                contrast=contrast_score,
                temporal=temporal_score,
                score=score,
                bbox=(x, y, width, height),
            )
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return (
        candidates[:max_candidates],
        feature_response,
        candidate_mask,
        feature_threshold,
        temporal_threshold,
    )


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def diagnostic_panel(
    gray_patch: np.ndarray,
    feature_response: np.ndarray,
    candidate_mask: np.ndarray,
    visible: list[VisibleParticle],
    *,
    sequence_id: int,
    processing_fps: float,
) -> np.ndarray:
    size = gray_patch.shape[0]
    raw = cv2.cvtColor(gray_patch, cv2.COLOR_GRAY2BGR)
    for particle in visible:
        x, y, width, height = particle.bbox
        color = (0, 220, 0) if particle.confirmed else (0, 180, 255)
        cv2.rectangle(raw, (x, y), (x + width, y + height), color, 1)
        draw_text(
            raw,
            f"{particle.track_id}:{particle.confidence:.2f}",
            (max(0, x), max(12, y - 2)),
            color=color,
            scale=0.35,
        )
    heat = cv2.applyColorMap(
        cv2.normalize(feature_response, None, 0, 255, cv2.NORM_MINMAX),
        cv2.COLORMAP_TURBO,
    )
    mask_bgr = cv2.cvtColor(candidate_mask, cv2.COLOR_GRAY2BGR)
    panel = np.concatenate([raw, heat, mask_bgr], axis=1)
    header = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
    draw_text(
        header,
        (
            f"one-droplet ROI | sequence={sequence_id} | "
            f"algorithm={processing_fps:.1f} FPS"
        ),
        (8, 26),
        scale=0.5,
    )
    return np.concatenate([header, panel], axis=0)


def open_writer(
    path: Path,
    *,
    codec: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_model_manifest() -> list[dict[str, object]]:
    paths = [
        ROOT
        / "models"
        / "qnn_cell_droplet_v2_w4a6_square192_grouped"
        / "best.pt",
        ROOT
        / "models"
        / "qnn_cell_droplet_v2_w4a6_square192_quality_aug_v1_rc1"
        / "best.pt",
    ]
    manifest: list[dict[str, object]] = []
    for path in paths:
        if path.exists():
            manifest.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "role": "preserved_baseline_not_modified",
                }
            )
    return manifest


def make_contact_sheet(
    samples: list[np.ndarray],
    output_path: Path,
    *,
    columns: int = 4,
) -> None:
    if not samples:
        return
    height, width = samples[0].shape[:2]
    rows = math.ceil(len(samples) / columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, sample in enumerate(samples):
        row = index // columns
        column = index % columns
        sheet[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = sample
    cv2.imwrite(str(output_path), sheet)


def write_track_patch(
    track: ParticleTrack,
    *,
    output_directory: Path,
    sequence_id: int,
    confidence: float,
) -> Path | None:
    if track.best_patch is None:
        return None
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / (
        f"drop{sequence_id:04d}_track{track.track_id:05d}"
        f"_hits{track.hits:02d}_conf{confidence:.3f}.png"
    )
    cv2.imwrite(str(path), track.best_patch)
    return path


def finalize_tracks(
    tracks: Iterable[ParticleTrack],
    *,
    tracker: TemporalParticleTracker,
    sequence_id: int,
    review_root: Path,
    review_limit: int,
    review_count: dict[str, int],
    track_rows: list[dict[str, object]],
) -> None:
    for track in tracks:
        confidence = tracker._track_confidence(track)
        confirmed = tracker.is_confirmed(track)
        category = "confirmed" if confirmed else "uncertain"
        patch_path: Path | None = None
        if review_count[category] < review_limit:
            patch_path = write_track_patch(
                track,
                output_directory=review_root / category,
                sequence_id=sequence_id,
                confidence=confidence,
            )
            if patch_path is not None:
                review_count[category] += 1
        track_rows.append(
            {
                "droplet_sequence": sequence_id,
                "particle_track": track.track_id,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "best_frame": track.best_frame,
                "best_x": track.best_x,
                "best_y": track.best_y,
                "hits": track.hits,
                "mean_score": track.mean_score,
                "max_score": track.max_score,
                "confidence": confidence,
                "confirmed": int(confirmed),
                "review_patch": str(patch_path.resolve()) if patch_path else "",
            }
        )


def main() -> None:
    args = parse_args()
    ensure_odd(args.blackhat_kernel, "feature-kernel")
    if args.processing_size < 64 or args.processing_size % 2:
        raise ValueError("processing-size must be even and at least 64")
    if not 0 <= args.particle_confidence <= 1:
        raise ValueError("particle-confidence must be between 0 and 1")

    source = args.source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    review_root = output / "review_patches"
    review_count = {"confirmed": 0, "uncertain": 0}

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0:
        source_fps = 30.0

    acquisition_roi = Rect(
        args.roi_x,
        args.roi_y,
        args.roi_width,
        args.roi_height,
    )
    acquisition_roi.validate(source_width, source_height)
    channel_center_x = (
        acquisition_roi.width / 2.0
        if args.channel_center_x < 0
        else args.channel_center_x
    )
    if not 0 <= channel_center_x < acquisition_roi.width:
        raise ValueError("channel-center-x must be inside the acquisition ROI")

    print(
        f"Source: {source_width}x{source_height}, "
        f"{source_fps:.3f} FPS, {total_frames} frames"
    )
    print(
        f"Acquisition ROI: {acquisition_roi}; "
        f"processing ROI: {args.processing_size}x{args.processing_size}"
    )
    background_start = time.perf_counter()
    background = build_background(
        capture,
        acquisition_roi,
        total_frames=total_frames,
        sample_count=args.background_samples,
    )
    background_build_ms = (time.perf_counter() - background_start) * 1000.0
    cv2.imwrite(str(output / "background.png"), background)

    sequence_tracker = DropletSequenceTracker(
        new_droplet_jump=args.new_droplet_jump,
        max_missing=args.droplet_max_missing,
    )
    particle_tracker = TemporalParticleTracker(
        max_distance=args.particle_max_distance,
        min_hits=args.particle_min_hits,
        max_missed=args.particle_max_missed,
        confidence_threshold=args.particle_confidence,
        patch_size=args.review_patch_size,
    )

    core_mask = np.zeros(
        (args.processing_size, args.processing_size),
        dtype=np.uint8,
    )
    core_radius = int(
        round(args.droplet_radius * args.core_radius_ratio)
    )
    patch_center = args.processing_size // 2
    cv2.circle(
        core_mask,
        (patch_center, patch_center),
        core_radius,
        255,
        -1,
    )
    hanning_window = cv2.createHanningWindow(
        (args.processing_size, args.processing_size),
        cv2.CV_32F,
    )

    annotated_writer: cv2.VideoWriter | None = None
    diagnostic_writer: cv2.VideoWriter | None = None
    if not args.no_video:
        annotated_writer = open_writer(
            output / "hybrid_one_droplet_annotated.mp4",
            codec=args.codec,
            fps=source_fps,
            width=source_width,
            height=source_height,
        )
        diagnostic_writer = open_writer(
            output / "hybrid_one_droplet_diagnostic.mp4",
            codec=args.codec,
            fps=source_fps,
            width=args.processing_size * 3,
            height=args.processing_size + 40,
        )

    frame_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    processing_times_ms: list[float] = []
    localization_times_ms: list[float] = []
    particle_times_ms: list[float] = []
    contact_samples: list[np.ndarray] = []
    previous_patch: np.ndarray | None = None
    previous_sequence_id = 0
    droplet_frames = 0
    complete_droplet_frames = 0
    confirmed_visible_frames = 0
    wall_start = time.perf_counter()
    frame_index = 0

    while True:
        if args.max_frames and frame_index >= args.max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break

        algorithm_start = time.perf_counter()
        roi_bgr = crop_rect(frame, acquisition_roi)
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        localization_start = time.perf_counter()
        observation, difference, droplet_mask = find_droplet(
            gray_roi,
            background,
            threshold=args.background_threshold,
            min_area=args.droplet_min_area,
            max_area=args.droplet_max_area,
            channel_center_x=channel_center_x,
            expected_radius=args.droplet_radius,
        )
        observation, is_new_droplet = sequence_tracker.update(
            observation,
            frame_index,
        )
        localization_ms = (time.perf_counter() - localization_start) * 1000.0
        localization_times_ms.append(localization_ms)

        if is_new_droplet and previous_sequence_id:
            finalize_tracks(
                particle_tracker.reset(),
                tracker=particle_tracker,
                sequence_id=previous_sequence_id,
                review_root=review_root,
                review_limit=args.review_limit,
                review_count=review_count,
                track_rows=track_rows,
            )
            previous_patch = None
        current_sequence_id = sequence_tracker.sequence_id
        if observation is not None:
            droplet_frames += 1

        visible_particles: list[VisibleParticle] = []
        candidates: list[ParticleCandidate] = []
        finished_tracks: list[ParticleTrack] = []
        patch: np.ndarray | None = None
        feature_response = np.zeros(
            (args.processing_size, args.processing_size),
            dtype=np.uint8,
        )
        candidate_mask = feature_response.copy()
        shift_x = 0.0
        shift_y = 0.0
        phase_response = 0.0
        feature_threshold = 0.0
        temporal_threshold = 0.0
        particle_ms = 0.0
        complete = False

        if observation is not None:
            half = args.processing_size / 2.0
            complete = (
                observation.center_y >= half
                and observation.center_y < acquisition_roi.height - half
            )
            if complete:
                complete_droplet_frames += 1
                particle_start = time.perf_counter()
                patch = centered_crop(
                    gray_roi,
                    observation.center_x,
                    observation.center_y,
                    args.processing_size,
                )
                aligned_previous, shift, phase_response = align_previous_patch(
                    previous_patch,
                    patch,
                    hanning_window,
                )
                shift_x, shift_y = shift
                (
                    candidates,
                    feature_response,
                    candidate_mask,
                    feature_threshold,
                    temporal_threshold,
                ) = find_particle_candidates(
                    patch,
                    aligned_previous,
                    core_mask=core_mask,
                    blackhat_kernel=args.blackhat_kernel,
                    sigma=args.particle_sigma,
                    min_area=args.particle_min_area,
                    max_area=args.particle_max_area,
                    max_side=args.particle_max_side,
                    max_candidates=args.max_candidates,
                    polarity=args.particle_polarity,
                )
                visible_particles, finished_tracks = particle_tracker.update(
                    candidates,
                    patch,
                    frame_index,
                )
                finalize_tracks(
                    finished_tracks,
                    tracker=particle_tracker,
                    sequence_id=current_sequence_id,
                    review_root=review_root,
                    review_limit=args.review_limit,
                    review_count=review_count,
                    track_rows=track_rows,
                )
                previous_patch = patch
                particle_ms = (time.perf_counter() - particle_start) * 1000.0
                particle_times_ms.append(particle_ms)
                if any(item.confirmed for item in visible_particles):
                    confirmed_visible_frames += 1
        elif frame_index - sequence_tracker.last_frame > args.droplet_max_missing:
            previous_patch = None

        processing_ms = (time.perf_counter() - algorithm_start) * 1000.0
        processing_times_ms.append(processing_ms)
        processing_fps = 1000.0 / max(processing_ms, 1e-9)

        annotated = frame.copy()
        if args.show_search_roi:
            cv2.rectangle(
                annotated,
                (acquisition_roi.x, acquisition_roi.y),
                (acquisition_roi.x2, acquisition_roi.y2),
                (0, 190, 0),
                2,
            )
        if observation is not None:
            global_cx = acquisition_roi.x + int(round(observation.center_x))
            global_cy = acquisition_roi.y + int(round(observation.center_y))
            cv2.circle(
                annotated,
                (global_cx, global_cy),
                int(round(args.droplet_radius)),
                (255, 180, 0),
                2,
            )
            half_int = args.processing_size // 2
            roi_color = (0, 255, 0) if complete else (128, 128, 128)
            cv2.rectangle(
                annotated,
                (global_cx - half_int, global_cy - half_int),
                (global_cx + half_int, global_cy + half_int),
                roi_color,
                2,
            )
            for particle in visible_particles:
                local_x, local_y, width, height = particle.bbox
                x1 = global_cx - half_int + local_x
                y1 = global_cy - half_int + local_y
                color = (
                    (0, 255, 0) if particle.confirmed else (0, 180, 255)
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x1 + width, y1 + height),
                    color,
                    2,
                )
                draw_text(
                    annotated,
                    f"P{particle.track_id} {particle.confidence:.2f}",
                    (x1, max(16, y1 - 4)),
                    color=color,
                    scale=0.42,
                )
        header = (
            f"Hybrid one-droplet | frame {frame_index} | "
            f"drop {current_sequence_id} | {processing_fps:.1f} FPS | "
            f"candidate {len(candidates)} | "
            f"confirmed {sum(item.confirmed for item in visible_particles)}"
        )
        draw_text(annotated, header, (12, 28), scale=0.58)

        if annotated_writer is not None:
            annotated_writer.write(annotated)
        if diagnostic_writer is not None and patch is not None:
            diagnostic_writer.write(
                diagnostic_panel(
                    patch,
                    feature_response,
                    candidate_mask,
                    visible_particles,
                    sequence_id=current_sequence_id,
                    processing_fps=processing_fps,
                )
            )

        if (
            args.sample_count > 0
            and patch is not None
            and len(contact_samples) < args.sample_count
        ):
            sample_period = max(
                1,
                (
                    args.max_frames
                    if args.max_frames
                    else max(total_frames, 1)
                )
                // args.sample_count,
            )
            if frame_index % sample_period == 0:
                contact_samples.append(
                    diagnostic_panel(
                        patch,
                        feature_response,
                        candidate_mask,
                        visible_particles,
                        sequence_id=current_sequence_id,
                        processing_fps=processing_fps,
                    )
                )

        frame_rows.append(
            {
                "frame": frame_index,
                "time_seconds": frame_index / source_fps,
                "droplet_sequence": current_sequence_id,
                "droplet_found": int(observation is not None),
                "droplet_complete": int(complete),
                "droplet_center_x": (
                    observation.center_x if observation is not None else ""
                ),
                "droplet_center_y": (
                    observation.center_y if observation is not None else ""
                ),
                "droplet_radius_measured": (
                    observation.radius if observation is not None else ""
                ),
                "droplet_area": (
                    observation.area if observation is not None else ""
                ),
                "candidate_count": len(candidates),
                "visible_track_count": len(visible_particles),
                "confirmed_visible_count": sum(
                    item.confirmed for item in visible_particles
                ),
                "feature_threshold": feature_threshold,
                "temporal_threshold": temporal_threshold,
                "alignment_shift_x": shift_x,
                "alignment_shift_y": shift_y,
                "alignment_response": phase_response,
                "localization_ms": localization_ms,
                "particle_ms": particle_ms,
                "processing_ms": processing_ms,
            }
        )
        previous_sequence_id = current_sequence_id
        frame_index += 1

    if previous_sequence_id:
        finalize_tracks(
            particle_tracker.reset(),
            tracker=particle_tracker,
            sequence_id=previous_sequence_id,
            review_root=review_root,
            review_limit=args.review_limit,
            review_count=review_count,
            track_rows=track_rows,
        )

    wall_seconds = time.perf_counter() - wall_start
    capture.release()
    if annotated_writer is not None:
        annotated_writer.release()
    if diagnostic_writer is not None:
        diagnostic_writer.release()

    with (output / "per_frame.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    with (output / "particle_tracks.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        fields = [
            "droplet_sequence",
            "particle_track",
            "first_frame",
            "last_frame",
            "best_frame",
            "best_x",
            "best_y",
            "hits",
            "mean_score",
            "max_score",
            "confidence",
            "confirmed",
            "review_patch",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(track_rows)

    confirmed_tracks = [
        row for row in track_rows if int(row["confirmed"]) == 1
    ]
    summary = {
        "pipeline": "one_droplet_hybrid_v1",
        "accuracy_status": (
            "Not a ground-truth accuracy measurement. Confirmed tracks are "
            "temporal candidates that require human review before CNN training."
        ),
        "source": {
            "path": str(source),
            "width": source_width,
            "height": source_height,
            "container_fps": source_fps,
            "total_frames_reported": total_frames,
            "frames_processed": frame_index,
        },
        "configuration": {
            "acquisition_roi": asdict(acquisition_roi),
            "channel_center_x_inside_roi": channel_center_x,
            "processing_size": args.processing_size,
            "processing_pixels": args.processing_size**2,
            "full_frame_pixels": source_width * source_height,
            "pixel_reduction_vs_full_frame": (
                1.0
                - args.processing_size**2
                / max(source_width * source_height, 1)
            ),
            "droplet_radius": args.droplet_radius,
            "core_radius": core_radius,
            "particle_confidence_threshold": args.particle_confidence,
            "particle_min_hits": args.particle_min_hits,
            "background_threshold": args.background_threshold,
            "blackhat_kernel": args.blackhat_kernel,
            "particle_polarity": args.particle_polarity,
        },
        "timing": {
            "background_build_ms": background_build_ms,
            "algorithm_mean_ms": statistics.fmean(processing_times_ms),
            "algorithm_median_ms": statistics.median(processing_times_ms),
            "algorithm_p95_ms": percentile(processing_times_ms, 95),
            "algorithm_p99_ms": percentile(processing_times_ms, 99),
            "algorithm_mean_fps": (
                1000.0 / statistics.fmean(processing_times_ms)
            ),
            "algorithm_p95_latency_fps": (
                1000.0 / max(percentile(processing_times_ms, 95), 1e-9)
            ),
            "localization_mean_ms": statistics.fmean(localization_times_ms),
            "particle_stage_mean_ms_when_active": (
                statistics.fmean(particle_times_ms)
                if particle_times_ms
                else 0.0
            ),
            "wall_seconds_including_io_and_encoding": wall_seconds,
            "wall_fps_including_io_and_encoding": (
                frame_index / max(wall_seconds, 1e-9)
            ),
            "meets_100_fps_algorithm_target": (
                percentile(processing_times_ms, 95) <= 10.0
            ),
            "meets_5_ms_stretch_target": (
                percentile(processing_times_ms, 95) <= 5.0
            ),
        },
        "observations": {
            "droplet_frames": droplet_frames,
            "complete_droplet_frames": complete_droplet_frames,
            "droplet_sequences": sequence_tracker.sequence_id,
            "particle_tracks_total": len(track_rows),
            "particle_tracks_confirmed": len(confirmed_tracks),
            "frames_with_confirmed_particle": confirmed_visible_frames,
            "review_patches": review_count,
            "temporal_confirmation_ratio": (
                len(confirmed_tracks) / max(len(track_rows), 1)
            ),
        },
        "preserved_existing_models": existing_model_manifest(),
        "artifacts": {
            "annotated_video": (
                str((output / "hybrid_one_droplet_annotated.mp4").resolve())
                if not args.no_video
                else None
            ),
            "diagnostic_video": (
                str((output / "hybrid_one_droplet_diagnostic.mp4").resolve())
                if not args.no_video
                else None
            ),
            "per_frame_csv": str((output / "per_frame.csv").resolve()),
            "particle_tracks_csv": str(
                (output / "particle_tracks.csv").resolve()
            ),
            "review_directory": str(review_root.resolve()),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_contact_sheet(
        contact_samples,
        output / "contact_sheet.jpg",
    )

    print(json.dumps(summary["timing"], indent=2))
    print(json.dumps(summary["observations"], indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
