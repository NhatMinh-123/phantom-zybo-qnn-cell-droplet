#!/usr/bin/env python3
"""Integer reference for the downstream ROI detector intended for RTL.

The detector evaluates only a fixed counting gate in a 96x96 ROI. Droplets
use a 16-direction dark-ring score. Cells use the same eight-direction radial
response as the existing FPGA QNN guard. Temporal state converts persistent
frame detections into one-shot passage events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from scripts.dual_branch_radial_v2 import radial_response


# Q10 unit-circle directions. These constants become a tiny ROM in RTL.
_DIRECTIONS_Q10 = (
    (1024, 0),
    (946, 392),
    (724, 724),
    (392, 946),
    (0, 1024),
    (-392, 946),
    (-724, 724),
    (-946, 392),
    (-1024, 0),
    (-946, -392),
    (-724, -724),
    (-392, -946),
    (0, -1024),
    (392, -946),
    (724, -724),
    (946, -392),
)


@dataclass(frozen=True)
class RtlGateConfig:
    image_size: int = 96
    gate_x: int = 48
    droplet_radii: tuple[int, ...] = tuple(range(18, 35, 2))
    droplet_radial_delta: int = 4
    droplet_support_threshold: int = 4
    droplet_support_bonus: int = 8
    droplet_score_threshold: int = 863
    droplet_refractory_frames: int = 7
    cell_gate_half_width: int = 5
    cell_response_threshold: int = 40
    cell_minimum_y_distance: int = 6
    cell_maximum_candidates: int = 2
    cell_required_hits: int = 2
    cell_maximum_y_motion: int = 7
    cell_maximum_missed_frames: int = 2


@dataclass(frozen=True)
class CellGateCandidate:
    x: int
    y: int
    response: int


@dataclass(frozen=True)
class RtlGateEvent:
    class_id: int
    class_name: str
    frame_index: int
    center_x: int
    center_y: int
    score: int
    confidence: float


@dataclass
class _CellTrack:
    y: int
    hits: int = 1
    missed: int = 0
    emitted: bool = False
    best_score: int = 0


def _scale_q10(value: int, direction: int) -> int:
    product = value * direction
    if product >= 0:
        return (product + 512) // 1024
    return -((-product + 512) // 1024)


def ensure_gray96(image: np.ndarray, image_size: int = 96) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Expected a grayscale or BGR image")
    if gray.dtype != np.uint8:
        raise ValueError("Expected uint8 image data")
    if gray.shape != (image_size, image_size):
        gray = cv2.resize(
            gray,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def pack_rgb332(image_bgr: np.ndarray, image_size: int = 96) -> np.ndarray:
    """Resize BGR input and pack each pixel as RRRGGGBB."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image")
    if image_bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 image data")
    if image_bgr.shape[:2] != (image_size, image_size):
        image_bgr = cv2.resize(
            image_bgr,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )
    blue, green, red = cv2.split(image_bgr)
    return np.ascontiguousarray(
        ((red >> 5) << 5) | ((green >> 5) << 2) | (blue >> 6)
    )


def rgb332_feature_planes(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the FPGA's integer luma and Cb-like feature transforms."""

    if packed.ndim != 2 or packed.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional uint8 RGB332 image")
    red = (packed >> 5).astype(np.int16)
    green = ((packed >> 2) & 0x07).astype(np.int16)
    blue = (packed & 0x03).astype(np.int16)
    luma = np.clip(9 * red + 18 * green + 21 * blue, 0, 255).astype(
        np.uint8
    )
    cb = np.clip(128 + 42 * blue - 6 * red - 12 * green, 0, 255).astype(
        np.uint8
    )
    return luma, cb


def droplet_ring_gate_score(
    gray: np.ndarray,
    config: RtlGateConfig | None = None,
) -> tuple[int, int, int]:
    """Return maximum dark-ring score, center y and radius at the gate."""

    cfg = config or RtlGateConfig()
    gray = ensure_gray96(gray, cfg.image_size)
    best_score = 0
    best_y = cfg.image_size // 2
    best_radius = cfg.droplet_radii[0]
    for radius in cfg.droplet_radii:
        margin = radius + cfg.droplet_radial_delta
        for center_y in range(margin, cfg.image_size - margin):
            score = 0
            support = 0
            for direction_x, direction_y in _DIRECTIONS_Q10:
                samples: list[int] = []
                for sample_radius in (
                    radius - cfg.droplet_radial_delta,
                    radius,
                    radius + cfg.droplet_radial_delta,
                ):
                    x = cfg.gate_x + _scale_q10(sample_radius, direction_x)
                    y = center_y + _scale_q10(sample_radius, direction_y)
                    samples.append(int(gray[y, x]))
                contrast = samples[0] + samples[2] - 2 * samples[1]
                if contrast > 0:
                    score += contrast
                if contrast > cfg.droplet_support_threshold:
                    support += 1
            score += cfg.droplet_support_bonus * support
            if score > best_score:
                best_score = score
                best_y = center_y
                best_radius = radius
    return best_score, best_y, best_radius


def cell_gate_candidates(
    gray: np.ndarray,
    config: RtlGateConfig | None = None,
) -> list[CellGateCandidate]:
    """Return at most two separated radial peaks in the fixed gate strip."""

    cfg = config or RtlGateConfig()
    gray = ensure_gray96(gray, cfg.image_size)
    response, _ = radial_response(gray)
    local_peak = cv2.dilate(response, np.ones((3, 3), np.uint8))
    x1 = max(3, cfg.gate_x - cfg.cell_gate_half_width)
    x2 = min(cfg.image_size - 4, cfg.gate_x + cfg.cell_gate_half_width)
    valid = np.zeros_like(response, dtype=bool)
    valid[3 : cfg.image_size - 3, x1 : x2 + 1] = (
        response[3 : cfg.image_size - 3, x1 : x2 + 1]
        >= cfg.cell_response_threshold
    ) & (
        response[3 : cfg.image_size - 3, x1 : x2 + 1]
        == local_peak[3 : cfg.image_size - 3, x1 : x2 + 1]
    )
    ys, xs = np.nonzero(valid)
    ranked = sorted(
        (
            CellGateCandidate(int(x), int(y), int(response[y, x]))
            for y, x in zip(ys.tolist(), xs.tolist())
        ),
        key=lambda item: item.response,
        reverse=True,
    )
    accepted: list[CellGateCandidate] = []
    for item in ranked:
        if any(
            abs(item.y - previous.y) < cfg.cell_minimum_y_distance
            for previous in accepted
        ):
            continue
        accepted.append(item)
        if len(accepted) >= cfg.cell_maximum_candidates:
            break
    return accepted


@dataclass
class RtlDownstreamGate:
    """Stateful one-shot event generator matching the planned FPGA FSM."""

    config: RtlGateConfig = field(default_factory=RtlGateConfig)
    _droplet_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    _last_droplet_event: int = -1_000_000
    _cell_tracks: list[_CellTrack] = field(default_factory=list)

    def _droplet_events(
        self,
        frame_index: int,
        score: int,
        center_y: int,
        radius: int,
    ) -> list[RtlGateEvent]:
        cfg = self.config
        self._droplet_history.append((frame_index, score, center_y, radius))
        if len(self._droplet_history) > 3:
            self._droplet_history.pop(0)
        if len(self._droplet_history) < 3:
            return []
        previous, peak, current = self._droplet_history
        peak_frame, peak_score, peak_y, _ = peak
        if (
            peak_score >= cfg.droplet_score_threshold
            and peak_score >= previous[1]
            and peak_score > current[1]
            and peak_frame - self._last_droplet_event
            >= cfg.droplet_refractory_frames
        ):
            self._last_droplet_event = peak_frame
            return [
                RtlGateEvent(
                    class_id=1,
                    class_name="droplet",
                    frame_index=peak_frame,
                    center_x=cfg.gate_x,
                    center_y=peak_y,
                    score=peak_score,
                    confidence=min(1.0, peak_score / 1726.0),
                )
            ]
        return []

    def _cell_events(
        self,
        frame_index: int,
        candidates: list[CellGateCandidate],
    ) -> list[RtlGateEvent]:
        cfg = self.config
        unmatched_tracks = set(range(len(self._cell_tracks)))
        unmatched_candidates = set(range(len(candidates)))
        choices: list[tuple[int, int, int]] = []
        for track_index, track in enumerate(self._cell_tracks):
            for candidate_index, candidate in enumerate(candidates):
                distance = abs(track.y - candidate.y)
                if distance <= cfg.cell_maximum_y_motion:
                    choices.append((distance, track_index, candidate_index))
        for _, track_index, candidate_index in sorted(choices):
            if (
                track_index not in unmatched_tracks
                or candidate_index not in unmatched_candidates
            ):
                continue
            track = self._cell_tracks[track_index]
            candidate = candidates[candidate_index]
            track.y = candidate.y
            track.hits += 1
            track.missed = 0
            track.best_score = max(track.best_score, candidate.response)
            unmatched_tracks.remove(track_index)
            unmatched_candidates.remove(candidate_index)

        for track_index in unmatched_tracks:
            self._cell_tracks[track_index].missed += 1
        self._cell_tracks = [
            track
            for track in self._cell_tracks
            if track.missed <= cfg.cell_maximum_missed_frames
        ]
        for candidate_index in sorted(unmatched_candidates):
            candidate = candidates[candidate_index]
            self._cell_tracks.append(
                _CellTrack(y=candidate.y, best_score=candidate.response)
            )

        events: list[RtlGateEvent] = []
        for track in self._cell_tracks:
            if track.emitted or track.hits < cfg.cell_required_hits:
                continue
            track.emitted = True
            events.append(
                RtlGateEvent(
                    class_id=0,
                    class_name="cell",
                    frame_index=frame_index,
                    center_x=cfg.gate_x,
                    center_y=track.y,
                    score=track.best_score,
                    confidence=min(1.0, track.best_score / 48.0),
                )
            )
        return events

    def process_frame(
        self,
        image: np.ndarray,
        frame_index: int,
    ) -> tuple[list[RtlGateEvent], dict[str, object]]:
        gray = ensure_gray96(image, self.config.image_size)
        droplet_score, droplet_y, droplet_radius = droplet_ring_gate_score(
            gray, self.config
        )
        cells = cell_gate_candidates(gray, self.config)
        events = self._droplet_events(
            frame_index, droplet_score, droplet_y, droplet_radius
        )
        events.extend(self._cell_events(frame_index, cells))
        return events, {
            "droplet_score": droplet_score,
            "droplet_center_y": droplet_y,
            "droplet_radius": droplet_radius,
            "cell_candidates": len(cells),
            "cell_responses": [item.response for item in cells],
            "active_cell_tracks": len(self._cell_tracks),
        }
