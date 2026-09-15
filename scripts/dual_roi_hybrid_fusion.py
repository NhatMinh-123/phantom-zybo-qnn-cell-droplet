#!/usr/bin/env python3
"""Independent downstream image processing and cross-ROI consensus helpers.

The upstream ROI is evaluated by the FPGA QNN. The downstream ROI is evaluated
by this classical reference branch. Only one-to-one, class-consistent temporal
matches are counted. The Hough droplet detector is a high-accuracy reference;
the radial cell response already has a compact integer RTL contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Mapping, Sequence

import cv2
import numpy as np

from scripts.dual_branch_radial_v2 import RadialConfig, detect_radial_cells
from scripts.dual_roi_temporal_counter import CrossingEvent, DetectionSample


@dataclass(frozen=True)
class ClassicalBranchConfig:
    working_size: int = 96
    hough_dp: float = 1.0
    hough_min_distance: int = 34
    hough_param1: float = 65.0
    hough_param2: float = 18.0
    droplet_min_radius: int = 20
    droplet_max_radius: int = 53
    droplet_min_ring_score: float = 0.34
    cell: RadialConfig = field(
        default_factory=lambda: RadialConfig(
            response_threshold=40,
            local_contrast_threshold=12,
            minimum_distance=4,
            box_size=8,
            border=2,
            maximum_candidates=24,
            droplet_margin=0.06,
        )
    )


@dataclass(frozen=True)
class HybridFusionConfig:
    minimum_delay_frames: int = 1
    maximum_delay_frames: int = 60
    maximum_cross_axis_distance: float = 36.0
    nominal_delay_frames: float = 12.0


@dataclass(frozen=True)
class HybridCountEvent:
    event_index: int
    class_id: int
    class_name: str
    qnn_track_id: int
    classical_track_id: int
    qnn_frame_index: int
    classical_frame_index: int
    delay_frames: int
    cross_axis_distance: float
    qnn_confidence: float
    classical_confidence: float
    fused_confidence: float
    cumulative_count: int


def _ring_support(
    magnitude: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> float:
    height, width = magnitude.shape
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    values: list[int] = []
    for angle in angles:
        x = int(round(center_x + radius * np.cos(angle)))
        y = int(round(center_y + radius * np.sin(angle)))
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            values.append(int(np.max(magnitude[y - 1 : y + 2, x - 1 : x + 2])))
    if len(values) < 24:
        return 0.0
    threshold = max(18.0, float(np.percentile(magnitude, 74)))
    return float(np.mean(np.asarray(values) >= threshold))


def _circle_iou(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> float:
    lx, ly, lr = left
    rx, ry, rr = right
    distance = hypot(lx - rx, ly - ry)
    if distance >= lr + rr:
        return 0.0
    if distance <= abs(lr - rr):
        small = min(lr, rr)
        large = max(lr, rr)
        return (small * small) / max(1.0, float(large * large))
    # A distance-only overlap proxy is sufficient for suppressing Hough duplicates.
    return max(0.0, 1.0 - distance / max(1.0, float(lr + rr)))


class DownstreamClassicalDetector:
    """Detect droplets and their small cell/particle blobs without using QNN."""

    def __init__(
        self,
        *,
        roi_geometry: tuple[int, int, int, int],
        class_ids: Mapping[str, int],
        config: ClassicalBranchConfig | None = None,
    ) -> None:
        self.roi_geometry = roi_geometry
        self.class_ids = dict(class_ids)
        self.config = config or ClassicalBranchConfig()
        for name in ("cell", "droplet"):
            if name not in self.class_ids:
                raise KeyError(f"Missing class id for {name!r}")

    def _droplets(
        self, gray: np.ndarray
    ) -> list[tuple[tuple[int, int, int], float]]:
        cfg = self.config
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.3)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=cfg.hough_dp,
            minDist=cfg.hough_min_distance,
            param1=cfg.hough_param1,
            param2=cfg.hough_param2,
            minRadius=cfg.droplet_min_radius,
            maxRadius=cfg.droplet_max_radius,
        )
        if circles is None:
            return []
        gx = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        magnitude = np.clip(np.abs(gx) + np.abs(gy), 0, 255).astype(np.uint8)
        ranked: list[tuple[tuple[int, int, int], float]] = []
        for raw_x, raw_y, raw_radius in np.round(circles[0]).astype(int):
            circle = (int(raw_x), int(raw_y), int(raw_radius))
            score = _ring_support(magnitude, *circle)
            if score >= cfg.droplet_min_ring_score:
                ranked.append((circle, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        accepted: list[tuple[tuple[int, int, int], float]] = []
        for circle, score in ranked:
            if any(_circle_iou(circle, previous) >= 0.52 for previous, _ in accepted):
                continue
            accepted.append((circle, score))
        return accepted

    def detect(self, roi_bgr: np.ndarray) -> tuple[list[DetectionSample], dict[str, object]]:
        if roi_bgr.size == 0:
            raise ValueError("Downstream ROI is empty")
        cfg = self.config
        gray_source = (
            roi_bgr
            if roi_bgr.ndim == 2
            else cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        )
        source_height, source_width = gray_source.shape
        droplets = self._droplets(gray_source)
        normalized_droplets: list[tuple[float, float, float, float]] = []
        detections: list[DetectionSample] = []
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_geometry
        for (center_x, center_y, radius), score in droplets:
            local_box = (
                max(0.0, center_x - radius),
                max(0.0, center_y - radius),
                min(float(source_width), center_x + radius),
                min(float(source_height), center_y + radius),
            )
            normalized = (
                local_box[0] / source_width,
                local_box[1] / source_height,
                local_box[2] / source_width,
                local_box[3] / source_height,
            )
            normalized_droplets.append(normalized)
            detections.append(
                DetectionSample(
                    class_id=self.class_ids["droplet"],
                    class_name="droplet",
                    confidence=score,
                    box=(
                        roi_x1 + local_box[0],
                        roi_y1 + local_box[1],
                        roi_x1 + local_box[2],
                        roi_y1 + local_box[3],
                    ),
                )
            )

        working = cv2.resize(
            gray_source,
            (cfg.working_size, cfg.working_size),
            interpolation=cv2.INTER_AREA,
        )
        cells = detect_radial_cells(
            working,
            cfg.cell,
            droplet_boxes=normalized_droplets,
        )
        for cell in cells:
            detections.append(
                DetectionSample(
                    class_id=self.class_ids["cell"],
                    class_name="cell",
                    confidence=cell.score,
                    box=(
                        roi_x1 + cell.box[0] * (roi_x2 - roi_x1),
                        roi_y1 + cell.box[1] * (roi_y2 - roi_y1),
                        roi_x1 + cell.box[2] * (roi_x2 - roi_x1),
                        roi_y1 + cell.box[3] * (roi_y2 - roi_y1),
                    ),
                )
            )
        diagnostics: dict[str, object] = {
            "droplet_candidates": len(droplets),
            "cell_candidates": len(cells),
            "droplet_ring_scores": [score for _, score in droplets],
            "cell_responses": [cell.response for cell in cells],
        }
        return detections, diagnostics


class CrossBranchConsensus:
    """One-to-one temporal AND between QNN and classical crossing events."""

    def __init__(self, config: HybridFusionConfig | None = None) -> None:
        self.config = config or HybridFusionConfig()
        self._qnn: list[CrossingEvent] = []
        self._classical: list[CrossingEvent] = []
        self.counts: dict[str, int] = {}
        self.event_index = 0
        self.rejected_qnn = 0
        self.rejected_classical = 0

    def _expire(self, current_frame: int) -> None:
        earliest = current_frame - self.config.maximum_delay_frames
        kept_qnn = [item for item in self._qnn if item.frame_index >= earliest]
        self.rejected_qnn += len(self._qnn) - len(kept_qnn)
        self._qnn = kept_qnn
        # Classical events may arrive at the host before a delayed FPGA response.
        classical_earliest = current_frame - 2 * self.config.maximum_delay_frames
        kept_classical = [
            item for item in self._classical if item.frame_index >= classical_earliest
        ]
        self.rejected_classical += len(self._classical) - len(kept_classical)
        self._classical = kept_classical

    def _match(self) -> list[HybridCountEvent]:
        cfg = self.config
        pairs: list[tuple[float, int, int, int, float]] = []
        for qnn_index, qnn in enumerate(self._qnn):
            for classical_index, classical in enumerate(self._classical):
                if qnn.class_id != classical.class_id:
                    continue
                delay = classical.frame_index - qnn.frame_index
                if not cfg.minimum_delay_frames <= delay <= cfg.maximum_delay_frames:
                    continue
                cross_axis = abs(classical.center_y - qnn.center_y)
                if cross_axis > cfg.maximum_cross_axis_distance:
                    continue
                time_error = abs(delay - cfg.nominal_delay_frames) / max(
                    1.0, cfg.maximum_delay_frames - cfg.minimum_delay_frames
                )
                cross_error = cross_axis / max(1.0, cfg.maximum_cross_axis_distance)
                confidence_bonus = 0.5 * (qnn.confidence + classical.confidence)
                cost = 0.58 * time_error + 0.42 * cross_error - 0.15 * confidence_bonus
                pairs.append((cost, qnn_index, classical_index, delay, cross_axis))
        pairs.sort()
        used_qnn: set[int] = set()
        used_classical: set[int] = set()
        matched: list[tuple[int, int, int, float]] = []
        for _, qnn_index, classical_index, delay, cross_axis in pairs:
            if qnn_index in used_qnn or classical_index in used_classical:
                continue
            used_qnn.add(qnn_index)
            used_classical.add(classical_index)
            matched.append((qnn_index, classical_index, delay, cross_axis))

        events: list[HybridCountEvent] = []
        for qnn_index, classical_index, delay, cross_axis in matched:
            qnn = self._qnn[qnn_index]
            classical = self._classical[classical_index]
            cumulative = self.counts.get(qnn.class_name, 0) + 1
            self.counts[qnn.class_name] = cumulative
            self.event_index += 1
            events.append(
                HybridCountEvent(
                    event_index=self.event_index,
                    class_id=qnn.class_id,
                    class_name=qnn.class_name,
                    qnn_track_id=qnn.track_id,
                    classical_track_id=classical.track_id,
                    qnn_frame_index=qnn.frame_index,
                    classical_frame_index=classical.frame_index,
                    delay_frames=delay,
                    cross_axis_distance=cross_axis,
                    qnn_confidence=qnn.confidence,
                    classical_confidence=classical.confidence,
                    fused_confidence=float(
                        np.sqrt(max(0.0, qnn.confidence * classical.confidence))
                    ),
                    cumulative_count=cumulative,
                )
            )
        self._qnn = [item for index, item in enumerate(self._qnn) if index not in used_qnn]
        self._classical = [
            item
            for index, item in enumerate(self._classical)
            if index not in used_classical
        ]
        return sorted(events, key=lambda item: item.classical_frame_index)

    def add_qnn(
        self, events: Sequence[CrossingEvent], current_frame: int
    ) -> list[HybridCountEvent]:
        self._qnn.extend(events)
        self._expire(current_frame)
        return self._match()

    def add_classical(
        self, events: Sequence[CrossingEvent], current_frame: int
    ) -> list[HybridCountEvent]:
        self._classical.extend(events)
        self._expire(current_frame)
        return self._match()

    @property
    def pending_qnn(self) -> int:
        return len(self._qnn)

    @property
    def pending_classical(self) -> int:
        return len(self._classical)
