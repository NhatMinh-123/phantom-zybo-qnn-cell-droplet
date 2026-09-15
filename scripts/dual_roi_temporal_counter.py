#!/usr/bin/env python3
"""Temporal tracking and one-shot line crossing counts for dual ROI detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, log
from typing import Mapping, Sequence


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionSample:
    class_id: int
    class_name: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: Box
    center: tuple[float, float]
    velocity: tuple[float, float]
    hits: int
    misses: int
    confirmed: bool
    counted: bool
    predicted: bool
    last_seen_frame: int


@dataclass(frozen=True)
class CrossingEvent:
    region_name: str
    track_id: int
    class_id: int
    class_name: str
    frame_index: int
    center_x: float
    center_y: float
    confidence: float
    hits: int
    velocity_x: float
    velocity_y: float
    cumulative_count: int


@dataclass(frozen=True)
class CrossRoiAssociation:
    downstream_event: CrossingEvent
    upstream_event: CrossingEvent | None
    delay_frames: int | None
    cross_axis_distance: float | None

    @property
    def verified(self) -> bool:
        return self.upstream_event is not None


@dataclass
class _Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float
    first_center_x: float
    first_center_y: float
    first_frame: int
    last_seen_frame: int
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    hits: int = 1
    misses: int = 0
    counted: bool = False
    armed: bool = False
    matched_this_update: bool = True
    history: list[tuple[int, float, float]] = field(default_factory=list)


def box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def box_size(box: Box) -> tuple[float, float]:
    return max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])


def box_iou(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


class ClassAwareLineTracker:
    """Track detections and count a confirmed track once across a line.

    A track must first be observed on the upstream side of the hysteresis band.
    It is counted only after it is confirmed and reaches the downstream side.
    Short detector gaps are coasted with a constant-velocity prediction.
    """

    def __init__(
        self,
        *,
        region_name: str,
        roi_geometry: tuple[int, int, int, int],
        line_fraction: float,
        direction: str,
        minimum_hits: Mapping[str, int] | int,
        max_misses: Mapping[str, int] | int,
        max_center_distance: Mapping[str, float] | float,
        count_hysteresis: float,
        reverse_tolerance: float = 0.04,
        edge_margin: float = 0.04,
    ) -> None:
        if direction not in {"left_to_right", "right_to_left"}:
            raise ValueError(f"Unsupported count direction: {direction}")
        if not 0.0 < line_fraction < 1.0:
            raise ValueError("line_fraction must be between zero and one")
        self.region_name = region_name
        self.roi_geometry = roi_geometry
        self.direction = direction
        self.minimum_hits = minimum_hits
        self.max_misses = max_misses
        self.max_center_distance = max_center_distance
        self.roi_width = float(roi_geometry[2] - roi_geometry[0])
        self.roi_height = float(roi_geometry[3] - roi_geometry[1])
        self.line_x = roi_geometry[0] + line_fraction * self.roi_width
        self.hysteresis_px = count_hysteresis * self.roi_width
        self.reverse_tolerance_px = reverse_tolerance * self.roi_width
        self.edge_margin_px = edge_margin * min(self.roi_width, self.roi_height)
        self.tracks: dict[int, _Track] = {}
        self.next_track_id = 1
        self.counts: dict[str, int] = {}
        self.edge_detections = 0
        self.coasted_track_updates = 0

    @staticmethod
    def _class_value(value: Mapping[str, int | float] | int | float, name: str) -> float:
        if isinstance(value, Mapping):
            if name not in value:
                raise KeyError(f"No tracker setting for class {name!r}")
            return float(value[name])
        return float(value)

    def _minimum_hits(self, class_name: str) -> int:
        return int(self._class_value(self.minimum_hits, class_name))

    def _max_misses(self, class_name: str) -> int:
        return int(self._class_value(self.max_misses, class_name))

    def _maximum_distance(self, class_name: str) -> float:
        return self._class_value(self.max_center_distance, class_name) * self.roi_width

    def _side(self, center_x: float) -> int:
        if center_x <= self.line_x - self.hysteresis_px:
            return -1
        if center_x >= self.line_x + self.hysteresis_px:
            return 1
        return 0

    def _is_upstream(self, side: int) -> bool:
        return side == (-1 if self.direction == "left_to_right" else 1)

    def _is_downstream(self, side: int) -> bool:
        return side == (1 if self.direction == "left_to_right" else -1)

    def _touches_edge(self, box: Box) -> bool:
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = self.roi_geometry
        margin = self.edge_margin_px
        return (
            x1 <= rx1 + margin
            or y1 <= ry1 + margin
            or x2 >= rx2 - margin
            or y2 >= ry2 - margin
        )

    @staticmethod
    def _track_box(track: _Track, center: tuple[float, float] | None = None) -> Box:
        cx, cy = center if center is not None else (track.center_x, track.center_y)
        return (
            cx - track.width * 0.5,
            cy - track.height * 0.5,
            cx + track.width * 0.5,
            cy + track.height * 0.5,
        )

    def _predicted_center(self, track: _Track, frame_index: int) -> tuple[float, float]:
        delta = max(0, frame_index - track.last_seen_frame)
        return (
            track.center_x + track.velocity_x * delta,
            track.center_y + track.velocity_y * delta,
        )

    def _pair_cost(
        self,
        track: _Track,
        detection: DetectionSample,
        frame_index: int,
    ) -> float | None:
        center_x, center_y = box_center(detection.box)
        predicted_x, predicted_y = self._predicted_center(track, frame_index)
        delta_frames = max(1, frame_index - track.last_seen_frame)
        distance = hypot(center_x - predicted_x, center_y - predicted_y)
        base_gate = self._maximum_distance(track.class_name)
        distance_gate = base_gate * (1.0 + 0.35 * (min(delta_frames, 4) - 1))
        if distance > distance_gate:
            return None
        if abs(center_y - predicted_y) > 0.38 * self.roi_height:
            return None
        if self.direction == "left_to_right":
            reverse_distance = track.center_x - center_x
        else:
            reverse_distance = center_x - track.center_x
        if reverse_distance > self.reverse_tolerance_px * max(1, delta_frames):
            return None

        width, height = box_size(detection.box)
        size_penalty = abs(log(width / track.width)) + abs(log(height / track.height))
        if size_penalty > 2.25:
            return None
        overlap = box_iou(self._track_box(track, (predicted_x, predicted_y)), detection.box)
        normalized_distance = distance / max(distance_gate, 1.0)
        normalized_size = min(1.0, size_penalty / 2.25)
        return 0.62 * normalized_distance + 0.23 * (1.0 - overlap) + 0.15 * normalized_size

    def _new_track(self, detection: DetectionSample, frame_index: int) -> _Track:
        center_x, center_y = box_center(detection.box)
        width, height = box_size(detection.box)
        side = self._side(center_x)
        track = _Track(
            track_id=self.next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            first_center_x=center_x,
            first_center_y=center_y,
            first_frame=frame_index,
            last_seen_frame=frame_index,
            armed=self._is_upstream(side),
            history=[(frame_index, center_x, center_y)],
        )
        self.next_track_id += 1
        return track

    def _maybe_count(
        self,
        track: _Track,
        frame_index: int,
        center_x: float,
        center_y: float,
    ) -> CrossingEvent | None:
        side = self._side(center_x)
        if self._is_upstream(side):
            track.armed = True
        if (
            track.counted
            or not track.armed
            or track.hits < self._minimum_hits(track.class_name)
            or not self._is_downstream(side)
        ):
            return None
        forward_displacement = (
            center_x - track.first_center_x
            if self.direction == "left_to_right"
            else track.first_center_x - center_x
        )
        if forward_displacement < 2.0 * self.hysteresis_px:
            return None

        track.counted = True
        cumulative = self.counts.get(track.class_name, 0) + 1
        self.counts[track.class_name] = cumulative
        return CrossingEvent(
            region_name=self.region_name,
            track_id=track.track_id,
            class_id=track.class_id,
            class_name=track.class_name,
            frame_index=frame_index,
            center_x=center_x,
            center_y=center_y,
            confidence=track.confidence,
            hits=track.hits,
            velocity_x=track.velocity_x,
            velocity_y=track.velocity_y,
            cumulative_count=cumulative,
        )

    def _update_track(
        self,
        track: _Track,
        detection: DetectionSample,
        frame_index: int,
    ) -> CrossingEvent | None:
        center_x, center_y = box_center(detection.box)
        width, height = box_size(detection.box)
        delta_frames = max(1, frame_index - track.last_seen_frame)
        measured_velocity_x = (center_x - track.center_x) / delta_frames
        measured_velocity_y = (center_y - track.center_y) / delta_frames
        if track.hits <= 1:
            velocity_alpha = 1.0
        else:
            velocity_alpha = 0.42
        track.velocity_x = (
            (1.0 - velocity_alpha) * track.velocity_x
            + velocity_alpha * measured_velocity_x
        )
        track.velocity_y = (
            (1.0 - velocity_alpha) * track.velocity_y
            + velocity_alpha * measured_velocity_y
        )
        if self._touches_edge(detection.box):
            # A clipped edge box must not shrink an already stable object extent.
            track.width = max(track.width, width)
            track.height = max(track.height, height)
        else:
            track.width = 0.60 * track.width + 0.40 * width
            track.height = 0.60 * track.height + 0.40 * height
        track.center_x = center_x
        track.center_y = center_y
        track.confidence = detection.confidence
        track.last_seen_frame = frame_index
        track.hits += 1
        track.misses = 0
        track.matched_this_update = True
        track.history.append((frame_index, center_x, center_y))
        track.history = track.history[-32:]

        return self._maybe_count(track, frame_index, center_x, center_y)

    def update(
        self,
        detections: Sequence[DetectionSample],
        frame_index: int,
    ) -> list[CrossingEvent]:
        for track in self.tracks.values():
            track.matched_this_update = False

        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                cost = self._pair_cost(track, detection, frame_index)
                if cost is not None:
                    pairs.append((cost, track_id, detection_index))
        pairs.sort()

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        events: list[CrossingEvent] = []
        for cost, track_id, detection_index in pairs:
            if cost >= 1.0:
                continue
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            event = self._update_track(
                self.tracks[track_id], detections[detection_index], frame_index
            )
            if event is not None:
                events.append(event)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track.misses += 1
                if track.hits >= self._minimum_hits(track.class_name):
                    self.coasted_track_updates += 1
                    predicted_x, predicted_y = self._predicted_center(
                        track, frame_index
                    )
                    event = self._maybe_count(
                        track, frame_index, predicted_x, predicted_y
                    )
                    if event is not None:
                        events.append(event)

        for detection_index, detection in enumerate(detections):
            if self._touches_edge(detection.box):
                self.edge_detections += 1
            if detection_index in matched_detections:
                continue
            track = self._new_track(detection, frame_index)
            self.tracks[track.track_id] = track

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.misses > self._max_misses(track.class_name)
        ]
        for track_id in expired:
            del self.tracks[track_id]
        return events

    def observations(
        self,
        frame_index: int,
        *,
        maximum_prediction_frames: int = 4,
    ) -> list[TrackObservation]:
        output: list[TrackObservation] = []
        for track in self.tracks.values():
            confirmed = track.hits >= self._minimum_hits(track.class_name)
            if not confirmed and not track.matched_this_update:
                continue
            prediction_frames = min(
                max(0, frame_index - track.last_seen_frame), maximum_prediction_frames
            )
            center = (
                track.center_x + track.velocity_x * prediction_frames,
                track.center_y + track.velocity_y * prediction_frames,
            )
            output.append(
                TrackObservation(
                    track_id=track.track_id,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    box=self._track_box(track, center),
                    center=center,
                    velocity=(track.velocity_x, track.velocity_y),
                    hits=track.hits,
                    misses=track.misses,
                    confirmed=confirmed,
                    counted=track.counted,
                    predicted=(
                        not track.matched_this_update
                        or frame_index > track.last_seen_frame
                    ),
                    last_seen_frame=track.last_seen_frame,
                )
            )
        return sorted(output, key=lambda item: (item.class_id, item.track_id))


class CrossRoiAssociator:
    """Associate downstream crossing events with earlier upstream events."""

    def __init__(
        self,
        *,
        minimum_delay_frames: int,
        maximum_delay_frames: int,
        maximum_cross_axis_distance: float,
        line_distance: float,
        preserve_order: bool = False,
    ) -> None:
        self.minimum_delay_frames = minimum_delay_frames
        self.maximum_delay_frames = maximum_delay_frames
        self.maximum_cross_axis_distance = maximum_cross_axis_distance
        self.line_distance = line_distance
        self.preserve_order = preserve_order
        self.pending: list[CrossingEvent] = []

    def add_upstream(self, events: Sequence[CrossingEvent]) -> None:
        self.pending.extend(events)

    def associate(self, downstream: CrossingEvent) -> CrossRoiAssociation:
        self.pending = [
            event
            for event in self.pending
            if downstream.frame_index - event.frame_index <= self.maximum_delay_frames
        ]
        candidates: list[tuple[float, int, int, float]] = []
        for index, upstream in enumerate(self.pending):
            if upstream.class_id != downstream.class_id:
                continue
            delay = downstream.frame_index - upstream.frame_index
            if not self.minimum_delay_frames <= delay <= self.maximum_delay_frames:
                continue
            cross_axis_distance = abs(downstream.center_y - upstream.center_y)
            if cross_axis_distance > self.maximum_cross_axis_distance:
                continue
            speed = abs(upstream.velocity_x)
            expected_delay = self.line_distance / speed if speed >= 0.25 else float(delay)
            time_error = abs(delay - expected_delay) / max(expected_delay, 1.0)
            score = time_error + cross_axis_distance / max(
                self.maximum_cross_axis_distance, 1.0
            )
            candidates.append((score, index, delay, cross_axis_distance))
        if not candidates:
            return CrossRoiAssociation(downstream, None, None, None)
        if self.preserve_order:
            _, index, delay, cross_axis_distance = min(
                candidates,
                key=lambda item: (self.pending[item[1]].frame_index, item[0]),
            )
        else:
            _, index, delay, cross_axis_distance = min(candidates)
        upstream = self.pending.pop(index)
        return CrossRoiAssociation(
            downstream,
            upstream,
            delay,
            cross_axis_distance,
        )
