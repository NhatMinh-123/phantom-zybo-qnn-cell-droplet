#!/usr/bin/env python3
"""Run one-ROI cell/droplet detection, tracking, counting, and CSV export."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


@dataclass
class ObjectDetection:
    class_id: int
    confidence: float
    roi_box: np.ndarray
    frame_box: np.ndarray
    track_id: int = -1
    crossed: bool = False


@dataclass
class Track:
    track_id: int
    class_id: int
    roi_box: np.ndarray
    center: np.ndarray
    previous_center: np.ndarray
    confidence: float
    hits: int = 1
    misses: int = 0
    counted: bool = False
    origin_side: int = 0
    last_frame: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect, track, and count cells and droplets in one fixed ROI."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Video path or camera index such as 0")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-track-misses", type=int, default=8)
    parser.add_argument("--max-center-distance", type=float, default=0.22)
    parser.add_argument("--inset-size", type=int, default=320)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def resolve_source(value: str) -> int | str:
    path = Path(value)
    if value.isdigit() and not path.exists():
        return int(value)
    return str(path.resolve())


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection_x1 = max(first[0], second[0])
    intersection_y1 = max(first[1], second[1])
    intersection_x2 = min(first[2], second[2])
    intersection_y2 = min(first[3], second[3])
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


class ClassAwareTracker:
    def __init__(
        self,
        model_width: int,
        max_misses: int,
        max_center_distance: float,
        count_direction: str,
        count_hysteresis: float,
        minimum_hits: int,
    ) -> None:
        self.max_misses = max_misses
        self.max_center_distance = max_center_distance * model_width
        self.count_direction = count_direction
        self.count_hysteresis = count_hysteresis * model_width
        self.minimum_hits = minimum_hits
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1

    def _cost(self, track: Track, detection: ObjectDetection) -> float:
        center = box_center(detection.roi_box)
        velocity = track.center - track.previous_center
        predicted_center = track.center + velocity
        distance = float(np.linalg.norm(center - predicted_center))
        overlap = box_iou(track.roi_box, detection.roi_box)
        if distance > self.max_center_distance and overlap < 0.05:
            return 1e6
        normalized_distance = distance / max(1.0, self.max_center_distance)
        return 0.65 * normalized_distance + 0.35 * (1.0 - overlap)

    def _side(self, center_x: float, count_line_x: float) -> int:
        if center_x <= count_line_x - self.count_hysteresis:
            return -1
        if center_x >= count_line_x + self.count_hysteresis:
            return 1
        return 0

    def _crossed(self, track: Track, current_side: int) -> bool:
        if current_side == 0 or track.origin_side == 0:
            return False
        if self.count_direction == "left_to_right":
            return track.origin_side == -1 and current_side == 1
        if self.count_direction == "right_to_left":
            return track.origin_side == 1 and current_side == -1
        return current_side != track.origin_side

    def update(
        self,
        detections: list[ObjectDetection],
        frame_index: int,
        count_line_x: float,
    ) -> list[ObjectDetection]:
        active_ids = list(self.tracks)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        if active_ids and detections:
            cost_matrix = np.full((len(active_ids), len(detections)), 1e6, dtype=float)
            for track_index, track_id in enumerate(active_ids):
                track = self.tracks[track_id]
                for detection_index, detection in enumerate(detections):
                    if track.class_id == detection.class_id:
                        cost_matrix[track_index, detection_index] = self._cost(
                            track, detection
                        )
            row_indices, column_indices = linear_sum_assignment(cost_matrix)
            for row_index, column_index in zip(row_indices, column_indices):
                if cost_matrix[row_index, column_index] >= 1.0:
                    continue
                track_id = active_ids[row_index]
                detection = detections[column_index]
                track = self.tracks[track_id]
                new_center = box_center(detection.roi_box)
                track.previous_center = track.center.copy()
                track.center = new_center
                track.roi_box = detection.roi_box.copy()
                track.confidence = detection.confidence
                track.hits += 1
                track.misses = 0
                track.last_frame = frame_index
                track.history.append((float(new_center[0]), float(new_center[1])))
                track.history = track.history[-20:]
                current_side = self._side(float(new_center[0]), count_line_x)
                if track.origin_side == 0 and current_side != 0:
                    track.origin_side = current_side
                if (
                    not track.counted
                    and track.hits >= self.minimum_hits
                    and self._crossed(track, current_side)
                ):
                    track.counted = True
                    detection.crossed = True
                detection.track_id = track_id
                matched_tracks.add(track_id)
                matched_detections.add(column_index)

        for track_id in active_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id].misses += 1

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            center = box_center(detection.roi_box)
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                class_id=detection.class_id,
                roi_box=detection.roi_box.copy(),
                center=center.copy(),
                previous_center=center.copy(),
                confidence=detection.confidence,
                origin_side=self._side(float(center[0]), count_line_x),
                last_frame=frame_index,
                history=[(float(center[0]), float(center[1]))],
            )
            detection.track_id = track_id

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.misses > self.max_misses
        ]
        for track_id in expired:
            del self.tracks[track_id]
        return detections


def scaled_roi(
    frame_width: int,
    frame_height: int,
    reference_width: int,
    reference_height: int,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
) -> tuple[int, int, int, int]:
    scale_x = frame_width / reference_width
    scale_y = frame_height / reference_height
    return (
        int(round(roi_x * scale_x)),
        int(round(roi_y * scale_y)),
        int(round((roi_x + roi_width) * scale_x)),
        int(round((roi_y + roi_height) * scale_y)),
    )


@dataclass(frozen=True)
class InputTransform:
    canvas_width: int
    canvas_height: int
    content_width: int
    content_height: int
    offset_x: int
    offset_y: int


def prepare_model_input(
    roi: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    content_width: int,
    content_height: int,
) -> tuple[np.ndarray, InputTransform]:
    resized = cv2.resize(
        roi, (content_width, content_height), interpolation=cv2.INTER_CUBIC
    )
    median_color = np.median(roi.reshape(-1, roi.shape[2]), axis=0).astype(np.uint8)
    model_input = np.empty((canvas_height, canvas_width, 3), dtype=np.uint8)
    model_input[:] = median_color
    offset_x = (canvas_width - content_width) // 2
    offset_y = (canvas_height - content_height) // 2
    model_input[
        offset_y : offset_y + content_height,
        offset_x : offset_x + content_width,
    ] = resized
    return model_input, InputTransform(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        content_width=content_width,
        content_height=content_height,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def map_box_to_frame(
    roi_box: np.ndarray,
    roi_geometry: tuple[int, int, int, int],
    transform: InputTransform,
) -> np.ndarray:
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry
    scale_x = (roi_x2 - roi_x1) / transform.content_width
    scale_y = (roi_y2 - roi_y1) / transform.content_height
    content_box = roi_box.copy()
    content_box[[0, 2]] = np.clip(
        content_box[[0, 2]] - transform.offset_x, 0, transform.content_width
    )
    content_box[[1, 3]] = np.clip(
        content_box[[1, 3]] - transform.offset_y, 0, transform.content_height
    )
    return np.array(
        [
            roi_x1 + content_box[0] * scale_x,
            roi_y1 + content_box[1] * scale_y,
            roi_x1 + content_box[2] * scale_x,
            roi_y1 + content_box[3] * scale_y,
        ],
        dtype=float,
    )


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )
    label_y1 = max(0, y1 - text_height - baseline - 5)
    cv2.rectangle(
        image,
        (x1, label_y1),
        (x1 + text_width + 6, label_y1 + text_height + baseline + 5),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 3, label_y1 + text_height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def make_contact_sheet(images: list[np.ndarray], output: Path) -> None:
    if not images:
        return
    tile_width, tile_height = 480, 300
    columns = 3
    tiles = [
        cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    rows = (len(tiles) + columns - 1) // columns
    tiles.extend([np.full_like(tiles[0], 245)] * (rows * columns - len(tiles)))
    sheet = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    cv2.imwrite(str(output), sheet)


def add_roi_inset(
    frame: np.ndarray,
    model_input: np.ndarray,
    detections: list[ObjectDetection],
    names: dict[int, str],
    count_line_x: float,
    inset_size: int,
) -> None:
    roi_view = model_input.copy()
    cv2.line(
        roi_view,
        (int(round(count_line_x)), 0),
        (int(round(count_line_x)), roi_view.shape[0] - 1),
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    for detection in detections:
        color = (30, 205, 40) if names[detection.class_id] == "cell" else (205, 80, 20)
        draw_box(
            roi_view,
            detection.roi_box,
            f"{names[detection.class_id]} #{detection.track_id} {detection.confidence:.2f}",
            color,
        )
    inset = cv2.resize(
        roi_view, (inset_size, inset_size), interpolation=cv2.INTER_AREA
    )
    header_height = 28
    panel = np.zeros((inset_size + header_height, inset_size, 3), dtype=np.uint8)
    panel[header_height:] = inset
    cv2.putText(
        panel,
        f"ROI {model_input.shape[1]}x{model_input.shape[0]} model input",
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    margin = 12
    x1 = max(0, frame.shape[1] - inset_size - margin)
    y1 = margin
    x2 = min(frame.shape[1], x1 + inset_size)
    y2 = min(frame.shape[0], y1 + panel.shape[0])
    frame[y1:y2, x1:x2] = panel[: y2 - y1, : x2 - x1]
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (230, 230, 230), 1)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="ascii"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_size = config.get("image_size", 640)
    if isinstance(image_size, int):
        model_height = model_width = image_size
    else:
        model_height, model_width = map(int, image_size)
    preprocess_config = config.get("preprocess", {})
    content_width = int(preprocess_config.get("content_width", model_width))
    content_height = int(preprocess_config.get("content_height", model_height))
    if content_width > model_width or content_height > model_height:
        raise ValueError("Preprocess content must fit inside the model canvas")
    static_transform = InputTransform(
        canvas_width=model_width,
        canvas_height=model_height,
        content_width=content_width,
        content_height=content_height,
        offset_x=(model_width - content_width) // 2,
        offset_y=(model_height - content_height) // 2,
    )
    model_imgsz: int | tuple[int, int] = (
        model_width if model_width == model_height else (model_height, model_width)
    )
    nms_iou = float(config.get("nms_iou", 0.5))
    roi_config = config["roi"]
    tracking_config = config.get("tracking", {})

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    model = YOLO(str(args.model.resolve()))
    names = {int(index): str(name) for index, name in model.names.items()}
    class_ids = {name: class_id for class_id, name in names.items()}
    thresholds = {
        class_id: float(config["confidence"].get(class_name, 0.5))
        for class_id, class_name in names.items()
    }
    minimum_confidence = min(thresholds.values())

    source = resolve_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    if args.start_frame > 0 and not isinstance(source, int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    available_frames = max(0, source_frames - args.start_frame)
    planned_frames = (
        min(args.max_frames, available_frames)
        if args.max_frames > 0 and available_frames > 0
        else available_frames
    )
    roi_width = int(roi_config.get("width", roi_config.get("size", 256)))
    roi_height = int(roi_config.get("height", roi_config.get("size", 256)))
    roi_geometry = scaled_roi(
        frame_width,
        frame_height,
        int(roi_config["reference_width"]),
        int(roi_config["reference_height"]),
        int(roi_config["x"]),
        int(roi_config["y"]),
        roi_width,
        roi_height,
    )
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry
    count_line_fraction = float(roi_config.get("count_line_fraction", 0.60))
    count_line_model_x = (
        static_transform.offset_x + count_line_fraction * static_transform.content_width
    )
    count_line_frame_x = roi_x1 + count_line_fraction * (roi_x2 - roi_x1)

    tracker = ClassAwareTracker(
        model_width,
        int(tracking_config.get("max_misses", args.max_track_misses)),
        float(tracking_config.get("max_center_distance", args.max_center_distance)),
        str(tracking_config.get("direction", "left_to_right")),
        float(tracking_config.get("count_hysteresis", 0.04)),
        int(tracking_config.get("minimum_hits", 3)),
    )
    cumulative_counts = {class_id: 0 for class_id in names}
    processed = 0
    processing_times: list[float] = []
    sheet_frames: list[np.ndarray] = []
    sample_interval = max(1, planned_frames // 12) if planned_frames else max(1, int(source_fps * 2))

    warmup_input = np.zeros((model_height, model_width, 3), dtype=np.uint8)
    for _ in range(int(config.get("warmup_frames", 3))):
        model.predict(
            warmup_input,
            imgsz=model_imgsz,
            conf=minimum_confidence,
            iou=nms_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )

    video_writer = None
    video_path = output / "realtime_result.mp4"
    if not args.no_video:
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (frame_width, frame_height),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not create output video: {video_path}")

    detections_path = output / "detections.csv"
    frames_path = output / "frame_summary.csv"
    with detections_path.open("w", newline="", encoding="ascii") as detections_file, frames_path.open(
        "w", newline="", encoding="ascii"
    ) as frames_file:
        detection_fields = [
            "frame",
            "time_s",
            "track_id",
            "class_id",
            "class",
            "confidence",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "frame_x1",
            "frame_y1",
            "frame_x2",
            "frame_y2",
            "crossed",
        ]
        detection_writer = csv.DictWriter(detections_file, fieldnames=detection_fields)
        detection_writer.writeheader()
        frame_fields = [
            "frame",
            "time_s",
            "detected_cell",
            "detected_droplet",
            "counted_cell",
            "counted_droplet",
            "processing_fps",
        ]
        frame_writer = csv.DictWriter(frames_file, fieldnames=frame_fields)
        frame_writer.writeheader()

        while True:
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            source_frame_index = args.start_frame + processed
            started = time.perf_counter()
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                raise RuntimeError(f"ROI is outside frame: {roi_geometry}")
            model_input, input_transform = prepare_model_input(
                roi,
                model_width,
                model_height,
                content_width,
                content_height,
            )
            result = model.predict(
                model_input,
                imgsz=model_imgsz,
                conf=minimum_confidence,
                iou=nms_iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )[0]

            detections: list[ObjectDetection] = []
            if result.boxes is not None:
                for class_id, confidence, roi_box in zip(
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.xyxy.cpu().numpy(),
                ):
                    class_id = int(class_id)
                    confidence = float(confidence)
                    if confidence < thresholds[class_id]:
                        continue
                    center = box_center(roi_box)
                    if not (
                        input_transform.offset_x <= center[0]
                        < input_transform.offset_x + input_transform.content_width
                        and input_transform.offset_y <= center[1]
                        < input_transform.offset_y + input_transform.content_height
                    ):
                        continue
                    detections.append(
                        ObjectDetection(
                            class_id=class_id,
                            confidence=confidence,
                            roi_box=roi_box.copy(),
                            frame_box=map_box_to_frame(
                                roi_box, roi_geometry, input_transform
                            ),
                        )
                    )
            tracker.update(detections, source_frame_index, count_line_model_x)
            for detection in detections:
                if detection.crossed:
                    cumulative_counts[detection.class_id] += 1

            elapsed = time.perf_counter() - started
            processing_times.append(elapsed)
            processing_fps = 1.0 / max(elapsed, 1e-9)
            annotated = frame.copy()
            cv2.rectangle(
                annotated,
                (roi_x1, roi_y1),
                (roi_x2, roi_y2),
                (240, 190, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                annotated,
                (int(round(count_line_frame_x)), roi_y1),
                (int(round(count_line_frame_x)), roi_y2),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            detected_counts = {class_id: 0 for class_id in names}
            for detection in detections:
                detected_counts[detection.class_id] += 1
                color = (30, 205, 40) if names[detection.class_id] == "cell" else (205, 80, 20)
                short_name = "C" if names[detection.class_id] == "cell" else "D"
                draw_box(
                    annotated,
                    detection.frame_box,
                    f"{short_name}{detection.track_id} {detection.confidence:.2f}",
                    color,
                )
                detection_writer.writerow(
                    {
                        "frame": source_frame_index,
                        "time_s": f"{source_frame_index / source_fps:.6f}",
                        "track_id": detection.track_id,
                        "class_id": detection.class_id,
                        "class": names[detection.class_id],
                        "confidence": f"{detection.confidence:.6f}",
                        "roi_x1": int(round(detection.roi_box[0])),
                        "roi_y1": int(round(detection.roi_box[1])),
                        "roi_x2": int(round(detection.roi_box[2])),
                        "roi_y2": int(round(detection.roi_box[3])),
                        "frame_x1": int(round(detection.frame_box[0])),
                        "frame_y1": int(round(detection.frame_box[1])),
                        "frame_x2": int(round(detection.frame_box[2])),
                        "frame_y2": int(round(detection.frame_box[3])),
                        "crossed": int(detection.crossed),
                    }
                )

            add_roi_inset(
                annotated,
                model_input,
                detections,
                names,
                count_line_model_x,
                args.inset_size,
            )

            cell_id = class_ids.get("cell", -1)
            droplet_id = class_ids.get("droplet", -1)
            hud_lines = [
                f"ROI detector | frame {source_frame_index:06d} | {processing_fps:.1f} FPS",
                (
                    f"Visible: cell {detected_counts.get(cell_id, 0)} | "
                    f"droplet {detected_counts.get(droplet_id, 0)}"
                ),
                (
                    f"Crossed: cell {cumulative_counts.get(cell_id, 0)} | "
                    f"droplet {cumulative_counts.get(droplet_id, 0)}"
                ),
            ]
            for line_index, line in enumerate(hud_lines):
                y_position = 28 + line_index * 26
                cv2.putText(
                    annotated,
                    line,
                    (12, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (20, 20, 20),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    line,
                    (12, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )

            frame_writer.writerow(
                {
                    "frame": source_frame_index,
                    "time_s": f"{source_frame_index / source_fps:.6f}",
                    "detected_cell": detected_counts.get(cell_id, 0),
                    "detected_droplet": detected_counts.get(droplet_id, 0),
                    "counted_cell": cumulative_counts.get(cell_id, 0),
                    "counted_droplet": cumulative_counts.get(droplet_id, 0),
                    "processing_fps": f"{processing_fps:.3f}",
                }
            )
            if video_writer is not None:
                video_writer.write(annotated)
            if processed % sample_interval == 0 and len(sheet_frames) < 12:
                sheet_frames.append(annotated.copy())
            if args.show:
                cv2.imshow("Cell and Droplet ROI Detector", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            processed += 1

    capture.release()
    if video_writer is not None:
        video_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    steady_times = processing_times[min(5, len(processing_times)) :]
    timing_values = np.asarray(steady_times or processing_times or [0.0], dtype=float)
    average_time = float(np.mean(timing_values))
    median_time = float(np.median(timing_values))
    p95_time = float(np.percentile(timing_values, 95))
    summary = {
        "source": args.source,
        "processed_frames": processed,
        "source_fps": source_fps,
        "average_processing_ms": average_time * 1000.0,
        "average_processing_fps": 1.0 / average_time if average_time > 0 else 0.0,
        "median_processing_ms": median_time * 1000.0,
        "median_processing_fps": 1.0 / median_time if median_time > 0 else 0.0,
        "p95_processing_ms": p95_time * 1000.0,
        "roi_frame_coordinates": list(roi_geometry),
        "model_input": {
            "canvas": [model_width, model_height],
            "content": [content_width, content_height],
            "padding": [static_transform.offset_x, static_transform.offset_y],
        },
        "tracking": {
            "direction": tracker.count_direction,
            "count_hysteresis_px": tracker.count_hysteresis,
            "minimum_hits": tracker.minimum_hits,
            "max_misses": tracker.max_misses,
        },
        "thresholds": {
            names[class_id]: threshold for class_id, threshold in thresholds.items()
        },
        "crossed_counts": {
            names[class_id]: count for class_id, count in cumulative_counts.items()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    make_contact_sheet(sheet_frames, output / "realtime_preview_sheet.jpg")
    print(json.dumps(summary, indent=2))
    print(f"Video: {video_path if video_writer is not None else 'disabled'}")
    print(f"Detections: {detections_path}")
    print(f"Frames: {frames_path}")


if __name__ == "__main__":
    main()
