#!/usr/bin/env python3
"""Run the 15 um dual-ROI QNN on live Phantom frames through Zybo Ethernet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import queue
import statistics
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyphantom import Phantom, utils


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import (  # noqa: E402
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    prepare_image,
)
from scripts.dual_roi_temporal_counter import (  # noqa: E402
    ClassAwareLineTracker,
    CrossRoiAssociator,
    DetectionSample,
)
from scripts.zybo_qnn_udp_protocol import ZyboQnnUdpClient  # noqa: E402
from scripts.latest_frame_capture import LatestFrameCapture  # noqa: E402
from scripts.phantom_native_stream import NativePhantomReader  # noqa: E402


RECORD_BYTES = 13
OBJECT_CHANNELS = (0, 5, 10)
ROI_COLORS = ((40, 210, 40), (235, 220, 20))
CLASS_COLORS = {"cell": (30, 60, 245), "droplet": (245, 90, 20)}


@dataclass(frozen=True)
class GlobalDetection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class FPGAResult:
    source_frame: int
    captured_at: float
    completed_at: float
    detections: tuple[tuple[GlobalDetection, ...], tuple[GlobalDetection, ...]]
    qnn_us: tuple[int, int]
    round_trip_seconds: tuple[float, float]
    records: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-ip", default="100.100.100.2")
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--display-fps", type=float, default=60.0)
    parser.add_argument("--camera-timeout", type=float, default=30.0)
    parser.add_argument("--camera-serial", type=int, default=25225)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--enable-counting", action="store_true")
    parser.add_argument("--acquisition-mode", choices=("sequential", "latest"), default="latest")
    parser.add_argument(
        "--reader",
        choices=("python", "native", "native-fast", "native-crop-fast"),
        default="native-crop-fast",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/phantom_zybo_qnn_live",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "exports/15micro_qnn_w4a6_96_roi120_v1/fpga_manifest_raw_core.json",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=ROOT / "configs/phantom_live_dual_roi_20260910.json",
    )
    return parser.parse_args()


def scaled_roi(
    width: int, height: int, config: dict[str, Any]
) -> tuple[int, int, int, int]:
    scale_x = width / int(config["reference_width"])
    scale_y = height / int(config["reference_height"])
    x1 = int(round(float(config["x"]) * scale_x))
    y1 = int(round(float(config["y"]) * scale_y))
    x2 = int(round((float(config["x"]) + float(config["width"])) * scale_x))
    y2 = int(round((float(config["y"]) + float(config["height"])) * scale_y))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ROI {(x1, y1, x2, y2)} is outside {width}x{height}")
    return x1, y1, x2, y2


def sparse_tensor(records: bytes, manifest: dict[str, Any]) -> np.ndarray:
    if len(records) % RECORD_BYTES:
        raise ValueError("Sparse FPGA result has an incomplete record")
    shape = tuple(
        int(value)
        for value in manifest["fpga_core"]["output_stream"]["shape_nhwc"][1:]
    )
    tensor = np.zeros(shape, dtype=np.int16)
    tensor[..., list(OBJECT_CHANNELS)] = np.int16(-32768)
    for offset in range(0, len(records), RECORD_BYTES):
        record = records[offset : offset + RECORD_BYTES]
        grid_index = int.from_bytes(record[0:2], "little")
        slot = int(record[2])
        if grid_index >= shape[0] * shape[1] or slot >= shape[2] // 5:
            raise ValueError(f"Invalid sparse record grid={grid_index} slot={slot}")
        grid_y, grid_x = divmod(grid_index, shape[1])
        tensor[grid_y, grid_x, slot * 5 : slot * 5 + 5] = np.frombuffer(
            record[3:13], dtype="<i2"
        )
    return tensor


def map_detections(
    decoded: list[Any], class_names: list[str], roi: tuple[int, int, int, int],
    rotate_roi_ccw_degrees: int = 0,
) -> tuple[GlobalDetection, ...]:
    if rotate_roi_ccw_degrees == 90:
        decoded = [replace(item, box=(1-item.box[3], item.box[0], 1-item.box[1], item.box[2])) for item in decoded]
    elif rotate_roi_ccw_degrees == 270:
        decoded = [replace(item, box=(item.box[1], 1-item.box[2], item.box[3], 1-item.box[0])) for item in decoded]
    elif rotate_roi_ccw_degrees != 0:
        raise ValueError("Only 0, 90 or 270-degree ROI rotation is supported")
    x1, y1, x2, y2 = roi
    width = x2 - x1
    height = y2 - y1
    return tuple(
        GlobalDetection(
            class_id=item.class_id,
            class_name=class_names[item.class_id],
            confidence=float(item.confidence),
            box=(
                int(round(x1 + item.box[0] * width)),
                int(round(y1 + item.box[1] * height)),
                int(round(x1 + item.box[2] * width)),
                int(round(y1 + item.box[3] * height)),
            ),
        )
        for item in decoded
    )


class FPGAWorker(threading.Thread):
    def __init__(
        self,
        board_ip: str,
        manifest: dict[str, Any],
        class_names: list[str],
        rois: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
        rotate_roi_ccw_degrees: int = 0,
    ) -> None:
        super().__init__(daemon=True)
        self.board_ip = board_ip
        self.manifest = manifest
        self.class_names = class_names
        self.rois = rois
        self.rotate_roi_ccw_degrees = rotate_roi_ccw_degrees
        self.jobs: queue.Queue[
            tuple[int, float, tuple[np.ndarray, np.ndarray]] | None
        ] = queue.Queue(maxsize=1)
        self.results: queue.Queue[FPGAResult] = queue.Queue(maxsize=2)
        self.ready_message: str | None = None
        self.error: BaseException | None = None
        self.dropped_jobs = 0

    def submit(
        self,
        frame_id: int,
        captured_at: float,
        crops: tuple[np.ndarray, np.ndarray],
    ) -> None:
        job = (frame_id, captured_at, crops)
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            try:
                self.jobs.get_nowait()
                self.dropped_jobs += 1
            except queue.Empty:
                pass
            self.jobs.put_nowait(job)

    def stop(self) -> None:
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            self.jobs.put_nowait(None)

    def _publish(self, result: FPGAResult) -> None:
        try:
            self.results.put_nowait(result)
        except queue.Full:
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            self.results.put_nowait(result)

    def run(self) -> None:
        try:
            with ZyboQnnUdpClient(
                board_ip=self.board_ip, timeout=0.35, retries=3
            ) as client:
                self.ready_message = client.hello()
                while True:
                    job = self.jobs.get()
                    if job is None:
                        break
                    frame_id, captured_at, crops = job
                    detections: list[tuple[GlobalDetection, ...]] = []
                    qnn_times: list[int] = []
                    round_trips: list[float] = []
                    record_counts: list[int] = []
                    for roi_id, (crop, roi) in enumerate(zip(crops, self.rois)):
                        if self.rotate_roi_ccw_degrees == 90:
                            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif self.rotate_roi_ccw_degrees == 270:
                            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                        codes = prepare_image(
                            crop, self.manifest, array_is_bgr=True
                        )
                        payload = pack_input_axis(codes, self.manifest)
                        response = client.infer(frame_id, roi_id, payload)
                        tensor = sparse_tensor(response.records, self.manifest)
                        decoded = decode_output_tensor(tensor, self.manifest)[0]
                        detections.append(
                            map_detections(decoded, self.class_names, roi, self.rotate_roi_ccw_degrees)
                        )
                        qnn_times.append(response.qnn_us)
                        round_trips.append(response.round_trip_seconds)
                        record_counts.append(response.record_count)
                    self._publish(
                        FPGAResult(
                            source_frame=frame_id,
                            captured_at=captured_at,
                            completed_at=time.perf_counter(),
                            detections=(detections[0], detections[1]),
                            qnn_us=(qnn_times[0], qnn_times[1]),
                            round_trip_seconds=(round_trips[0], round_trips[1]),
                            records=(record_counts[0], record_counts[1]),
                        )
                    )
        except BaseException as exc:
            self.error = exc


def connect_camera(timeout: float) -> tuple[Phantom, Any]:
    phantom = Phantom()
    deadline = time.perf_counter() + timeout
    while phantom.camera_count < 1 and time.perf_counter() < deadline:
        time.sleep(0.25)
    if phantom.camera_count < 1:
        phantom.close()
        raise RuntimeError("Phantom SDK did not discover a physical camera")
    return phantom, phantom.Camera(0)


def read_reduce8_bgr(camera: Any) -> np.ndarray:
    # SDK option 1 performs the same sensitivity-aware 16-to-8-bit reduction
    # used by the vendor examples. Per-frame percentile stretching is unstable.
    image = np.asarray(
        camera._live_cine.get_images(utils.FrameRange(0, 0), Option=1)[0]
    )
    if image.dtype != np.uint8:
        raise TypeError(f"Phantom Reduce8 returned {image.dtype}, expected uint8")
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unexpected Phantom image shape: {image.shape}")


def draw_tracker(
    frame: np.ndarray,
    tracker: ClassAwareLineTracker,
    label: str,
    color: tuple[int, int, int],
    observations: tuple[Any, ...],
    object_ids: dict[int, str],
) -> None:
    x1, y1, x2, y2 = tracker.roi_geometry
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.line(
        frame,
        (int(round(tracker.line_x)), y1),
        (int(round(tracker.line_x)), y2),
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x1, max(22, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )
    for item in observations:
        bx1, by1, bx2, by2 = (int(round(value)) for value in item.box)
        bx1 = max(x1, min(x2 - 1, bx1))
        bx2 = max(x1 + 1, min(x2, bx2))
        by1 = max(y1, min(y2 - 1, by1))
        by2 = max(y1 + 1, min(y2, by2))
        detection_color = CLASS_COLORS.get(item.class_name, color)
        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            detection_color,
            2 if item.confirmed else 1,
            cv2.LINE_AA,
        )
        object_id = object_ids.get(item.track_id, f"T{item.track_id}")
        predicted = "~" if item.predicted else ""
        cv2.putText(
            frame,
            f"{object_id}{predicted} {item.class_name} {item.confidence:.2f}",
            (bx1, max(18, by1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            detection_color,
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    if args.display_fps <= 0.0:
        raise ValueError("--display-fps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    runtime = json.loads(args.roi_config.read_text(encoding="utf-8"))
    if args.enable_counting and not runtime.get("counting_enabled", True):
        raise ValueError("Counting is not calibrated for this live ROI configuration")
    if args.enable_counting and args.acquisition_mode == "latest":
        raise ValueError("Counting with frame skipping requires time-based calibration; use sequential mode")
    tracking = runtime["tracking"]
    verification = runtime["cross_roi_verification"]

    phantom, camera = connect_camera(args.camera_timeout)
    if int(camera.serial) != args.camera_serial:
        camera.close()
        phantom.close()
        raise RuntimeError("SDK selected a different camera; refusing simulated or unexpected input")
    worker: FPGAWorker | None = None
    capture: LatestFrameCapture | None = None
    native_reader: NativePhantomReader | None = None
    writer: cv2.VideoWriter | None = None
    events: list[dict[str, object]] = []
    captured_hashes: set[str] = set()
    qnn_us_values: list[int] = []
    round_trip_values: list[float] = []
    latency_values: list[float] = []
    update_times: list[float] = []
    frame_count = 0
    display_count = 0
    result_count = 0
    saved_frame_count = 0
    frame_save_seconds: list[float] = []
    latest: FPGAResult | None = None
    observations: tuple[tuple[Any, ...], tuple[Any, ...]] = ((), ())
    confirmed_counts = {name: 0 for name in class_names}
    candidate_serials = {name: 0 for name in class_names}
    roi_1_ids: dict[int, str] = {}
    roi_2_ids: dict[int, str] = {}
    started = time.perf_counter()
    try:
        source_sequence = 0
        roi_config = runtime["rois"]
        sensor_frame = read_reduce8_bgr(camera)
        sensor_height, sensor_width = sensor_frame.shape[:2]
        global_rois = (
            scaled_roi(sensor_width, sensor_height, roi_config["left"]),
            scaled_roi(sensor_width, sensor_height, roi_config["right"]),
        )
        sdk_crop = None
        if args.reader in ("native", "native-fast", "native-crop-fast"):
            crop_request = None
            if args.reader == "native-crop-fast":
                crop_request = (
                    min(roi[0] for roi in global_rois),
                    min(roi[1] for roi in global_rois),
                    max(roi[2] for roi in global_rois),
                    max(roi[3] for roi in global_rois),
                )
            native_reader = NativePhantomReader(
                camera,
                fast_demosaic=args.reader in ("native-fast", "native-crop-fast"),
                crop_rect=crop_request,
            )
            read_frame = native_reader.read
        else:
            read_frame = lambda: read_reduce8_bgr(camera)
        if args.acquisition_mode == "latest":
            capture = LatestFrameCapture(read_frame)
            capture.start()
            first_item = capture.next_after(0)
            first_frame = first_item.image
        else:
            first_frame = read_frame()
        height, width = first_frame.shape[:2]
        if args.reader == "native-crop-fast":
            if native_reader is None or native_reader.effective_crop_rect is None:
                raise RuntimeError("Phantom SDK did not report the active crop rectangle")
            left, top, _, _ = native_reader.effective_crop_rect
            sdk_crop = (left, top, left + width, top + height)
            rois = tuple(
                (x1 - left, y1 - top, x2 - left, y2 - top)
                for x1, y1, x2, y2 in global_rois
            )
            for roi in rois:
                if not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
                    raise ValueError(
                        f"ROI {roi} does not fit SDK crop {sdk_crop} ({width}x{height})"
                    )
        else:
            rois = global_rois
        trackers = tuple(
            ClassAwareLineTracker(
                region_name=name,
                roi_geometry=roi,
                line_fraction=float(config["line_fraction"]),
                direction=str(tracking["direction"]),
                minimum_hits=tracking["minimum_hits"],
                max_misses=tracking["max_misses"],
                max_center_distance=tracking["max_center_distance"],
                count_hysteresis=float(tracking["count_hysteresis"]),
                reverse_tolerance=float(tracking["reverse_tolerance"]),
                edge_margin=float(tracking["edge_margin"]),
            )
            for name, roi, config in zip(
                ("candidate", "confirm"),
                rois,
                (roi_config["left"], roi_config["right"]),
            )
        )
        associator = CrossRoiAssociator(
            minimum_delay_frames=int(verification["minimum_delay_frames"]),
            maximum_delay_frames=int(verification["maximum_delay_frames"]),
            maximum_cross_axis_distance=(
                float(verification["maximum_cross_axis_distance_fraction"])
                * (rois[1][3] - rois[1][1])
            ),
            line_distance=trackers[1].line_x - trackers[0].line_x,
            preserve_order=bool(verification.get("preserve_order", False)),
        )

        output_video = args.output / "phantom_zybo_qnn_live.mp4"
        if args.record:
            writer = cv2.VideoWriter(
                str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                args.display_fps, (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create {output_video}")
        worker = FPGAWorker(args.board_ip, manifest, class_names, rois,
                            int(runtime.get("rotate_roi_ccw_degrees", 0)))
        worker.start()
        frame = first_frame
        cv2.imwrite(str(args.output / "source_first.png"), first_frame)
        recording_started = time.perf_counter()
        encoded_frames = 0
        next_deadline = time.perf_counter()
        while args.duration_sec <= 0.0 or time.perf_counter() - started < args.duration_sec:
            captured_at = time.perf_counter()
            frame_count += 1
            if capture is not None:
                item = first_item if frame_count == 1 else capture.next_after(source_sequence)
                frame = item.image
                source_sequence = item.sequence
                captured_at = item.started_at
            elif frame_count > 1:
                frame = read_frame()
            if capture is None:
                source_sequence = frame_count
            if frame_count <= 120:
                captured_hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
            crops = tuple(
                frame[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in rois
            )
            worker.submit(source_sequence, captured_at, crops)  # type: ignore[arg-type]

            # Stop-and-wait keeps each overlay attached to its own source frame.
            # Acquisition is included in the end-to-end benchmark below.
            try:
                result = worker.results.get(timeout=4.0)
            except queue.Empty:
                if worker.error is not None:
                    raise worker.error
                raise TimeoutError("No complete dual-ROI FPGA result")
            if result.source_frame != source_sequence:
                raise RuntimeError("FPGA result/source frame mismatch")
            worker.results.put_nowait(result)
            while True:
                try:
                    latest = worker.results.get_nowait()
                except queue.Empty:
                    break
                result_count += 1
                update_times.append(time.perf_counter())
                qnn_us_values.extend(latest.qnn_us)
                round_trip_values.extend(latest.round_trip_seconds)
                latency_values.append(latest.completed_at - latest.captured_at)
                samples = tuple(
                    tuple(
                        DetectionSample(
                            class_id=item.class_id,
                            class_name=item.class_name,
                            confidence=item.confidence,
                            box=tuple(float(value) for value in item.box),
                        )
                        for item in detections
                    )
                    for detections in latest.detections
                )
                if not args.enable_counting:
                    break
                upstream_events = trackers[0].update(samples[0], result_count)
                current_1 = tuple(
                    trackers[0].observations(
                        result_count,
                        maximum_prediction_frames=int(
                            tracking["maximum_prediction_frames"]
                        ),
                    )
                )
                for observation in current_1:
                    if not observation.confirmed or observation.track_id in roi_1_ids:
                        continue
                    candidate_serials[observation.class_name] += 1
                    prefix = "C" if observation.class_name == "cell" else "D"
                    roi_1_ids[observation.track_id] = (
                        f"{prefix}{candidate_serials[observation.class_name]:04d}"
                    )
                associator.add_upstream(upstream_events)
                associator.pending = [
                    event for event in associator.pending
                    if result_count - event.frame_index <= associator.maximum_delay_frames
                ]
                downstream_events = trackers[1].update(samples[1], result_count)
                for downstream in downstream_events:
                    association = associator.associate(downstream)
                    if not association.verified or association.upstream_event is None:
                        continue
                    upstream = association.upstream_event
                    object_id = roi_1_ids.get(upstream.track_id)
                    if object_id is None:
                        candidate_serials[upstream.class_name] += 1
                        prefix = "C" if upstream.class_name == "cell" else "D"
                        object_id = (
                            f"{prefix}{candidate_serials[upstream.class_name]:04d}"
                        )
                        roi_1_ids[upstream.track_id] = object_id
                    roi_2_ids[downstream.track_id] = object_id
                    confirmed_counts[downstream.class_name] += 1
                    events.append(
                        {
                            "event_id": len(events) + 1,
                            "object_id": object_id,
                            "class_name": downstream.class_name,
                            "source_frame": latest.source_frame,
                            "elapsed_seconds": time.perf_counter() - started,
                            "roi_1_result_index": upstream.frame_index,
                            "roi_2_result_index": downstream.frame_index,
                            "delay_updates": association.delay_frames,
                            "cross_axis_distance_px": association.cross_axis_distance,
                            "roi_1_confidence": upstream.confidence,
                            "roi_2_confidence": downstream.confidence,
                            "cumulative_class_count": confirmed_counts[
                                downstream.class_name
                            ],
                        }
                    )
                current_2 = tuple(
                    trackers[1].observations(
                        result_count,
                        maximum_prediction_frames=int(
                            tracking["maximum_prediction_frames"]
                        ),
                    )
                )
                observations = (current_1, current_2)

            if worker.error is not None:
                raise worker.error
            output = frame.copy()
            if args.enable_counting:
                draw_tracker(
                    output, trackers[0], "ROI 1: FPGA QNN candidate",
                    ROI_COLORS[0], observations[0], roi_1_ids,
                )
                draw_tracker(
                    output, trackers[1], "ROI 2: FPGA QNN confirm",
                    ROI_COLORS[1], observations[1], roi_2_ids,
                )
            else:
                for roi_id, (roi, detections) in enumerate(zip(rois, latest.detections)):
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(output, (x1,y1), (x2,y2), ROI_COLORS[roi_id], 1)
                    cv2.putText(output, f'ROI {roi_id + 1}', (max(0,x1-65),y1+18),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, ROI_COLORS[roi_id], 1)
                    for detection in detections:
                        a,b,c,d = detection.box
                        color = CLASS_COLORS[detection.class_name]
                        cv2.rectangle(output, (a,b), (c,d), color, 1)
                        cv2.putText(output, f'{detection.class_name} {detection.confidence:.2f}',
                                    (a,max(12,b-3)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
            elapsed = max(time.perf_counter() - started, 1e-9)
            update_fps = result_count / elapsed
            core_ms = (
                statistics.fmean(qnn_us_values[-30:]) / 1000.0
                if qnn_us_values
                else 0.0
            )
            acquisition_stats = capture.statistics() if capture is not None else {}
            if sdk_crop is None:
                banner_y = height - 62
                cv2.rectangle(output, (0, banner_y), (width, height), (10, 13, 17), -1)
                cv2.putText(
                    output,
                    f"Camera {acquisition_stats.get('stream_fps',update_fps):.1f} FPS -> PC SDK -> Ethernet -> Zybo QNN | detection {update_fps:.1f} FPS | {core_ms:.2f} ms/ROI",
                    (12, banner_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    (f"confirmed A->B once | cell={confirmed_counts.get('cell', 0)} droplet={confirmed_counts.get('droplet', 0)} | camera frame {frame_count}"
                     if args.enable_counting else f"Raw QNN detections | counting disabled pending scene/flow validation | frame {frame_count}"),
                    (12, banner_y + 49),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (80, 230, 80),
                    1,
                    cv2.LINE_AA,
                )
            # Preserve wall-clock playback duration; duplicate encoded frames
            # are never counted as new camera frames or FPGA inferences.
            target_frames = max(1, round((time.perf_counter() - recording_started) * args.display_fps))
            while writer is not None and encoded_frames < target_frames:
                writer.write(output)
                encoded_frames += 1
            if args.save_frames:
                frame_dir = args.output / "annotated_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                save_started = time.perf_counter()
                saved = cv2.imwrite(
                    str(frame_dir / f"frame_{source_sequence:06d}.jpg"),
                    output,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                frame_save_seconds.append(time.perf_counter() - save_started)
                if not saved:
                    raise RuntimeError(f"Could not save camera frame {source_sequence}")
                saved_frame_count += 1
            if result_count == 1 or result_count % 30 == 0:
                cv2.imwrite(str(args.output / "detection_latest.png"), output)
                heartbeat = dict(
                    source_frame=source_sequence, result_frame=latest.source_frame,
                    acquisition=acquisition_stats,
                    measured_dual_roi_fps=update_fps,
                    mean_qnn_ms_per_roi=core_ms,
                    mean_udp_ms_per_roi=statistics.fmean(round_trip_values[-30:])*1000,
                    flow_direction=runtime.get("flow_direction"),
                    rois=rois, counting_enabled=args.enable_counting,
                    detection_accuracy="Not validated",
                )
                (args.output / "live_status.json").write_text(json.dumps(heartbeat, indent=2))
            monitor = output
            if sdk_crop is not None:
                scaled = cv2.resize(
                    output, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST
                )
                panel = np.full((scaled.shape[0], 430, 3), (10, 13, 17), dtype=np.uint8)
                lines = (
                    "Phantom ROI stream -> Zybo FPGA QNN",
                    f"Camera stream: {acquisition_stats.get('stream_fps',update_fps):.1f} FPS",
                    f"Dual-ROI detection: {update_fps:.1f} FPS",
                    f"QNN core: {core_ms:.2f} ms / ROI",
                    f"UDP: {statistics.fmean(round_trip_values[-30:])*1000:.2f} ms / ROI",
                    f"Source frame: {source_sequence}",
                    "Counting: disabled until live calibration",
                )
                for line_index, line in enumerate(lines):
                    cv2.putText(
                        panel, line, (18, 38 + 42 * line_index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (80, 230, 80) if line_index in (1, 2) else (240, 240, 240),
                        1, cv2.LINE_AA,
                    )
                monitor = np.hstack((scaled, panel))
            if not args.headless:
                cv2.imshow("Phantom live - Zybo FPGA QNN", monitor)
            display_count += 1
            if not args.headless and cv2.waitKey(1) & 0xFF == 27:
                break
            next_deadline += 1.0 / args.display_fps
            delay = next_deadline - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)

        elapsed = time.perf_counter() - started
        if capture is not None:
            capture.stop()
            capture.join(timeout=5.0)
            if capture.is_alive():
                raise TimeoutError('Camera SDK did not finish its read during shutdown')
        report = {
            "status": "TRANSPORT_MEASURED" if result_count else "NO_INFERENCE",
            "detection_accuracy": "Not measured: requires visible droplets and reviewed ground truth",
            "roi_calibration": runtime.get("calibration_status", "Unverified"),
            "rotate_roi_ccw_degrees": runtime.get("rotate_roi_ccw_degrees", 0),
            "flow_direction": runtime.get("flow_direction", tracking["direction"]),
            "elapsed_seconds": elapsed,
            "source_frames": frame_count,
            "acquisition_mode": args.acquisition_mode,
            "reader": args.reader,
            "acquisition": capture.statistics() if capture is not None else None,
            "source_sequence_last": source_sequence,
            "skipped_source_frames": source_sequence-frame_count,
            "recording_encoded_frames": encoded_frames,
            "recording_duplicate_frames_are_not_inference": True,
            "saved_annotated_frames": saved_frame_count,
            "mean_frame_save_ms": (
                statistics.fmean(frame_save_seconds) * 1000.0
                if frame_save_seconds else None
            ),
            "counting_enabled": args.enable_counting,
            "camera": {
                "model": camera.model,
                "serial": int(camera.serial),
                "ip": camera.get_selector_string(utils.CamSelector.gsIPAddress),
                "sensor_resolution": [sensor_width, sensor_height],
                "stream_resolution": [width, height],
                "sdk_crop": list(sdk_crop) if sdk_crop is not None else None,
                "configured_fps": float(camera.frame_rate),
                "sdk_mode": "Reduce8",
                "unique_first_120_frames": len(captured_hashes),
            },
            "fpga": {
                "board": "Digilent Zybo Z7-10 XC7Z010-1CLG400C",
                "transport": f"UDP {args.board_ip}:50123",
                "ready_message": worker.ready_message,
                "inference_location": "FINN QNN in programmable logic via AXI DMA",
                "qnn_inferences": len(qnn_us_values),
                "mean_qnn_ms_per_roi": (
                    statistics.fmean(qnn_us_values) / 1000.0
                    if qnn_us_values
                    else None
                ),
                "mean_udp_round_trip_ms_per_roi": (
                    statistics.fmean(round_trip_values) * 1000.0
                    if round_trip_values
                    else None
                ),
                "mean_capture_to_dual_result_ms": (
                    statistics.fmean(latency_values) * 1000.0
                    if latency_values
                    else None
                ),
                "dual_roi_update_fps": result_count / max(elapsed, 1e-9),
                "dropped_jobs_for_freshness": worker.dropped_jobs,
            },
            "display": {
                "frames": display_count,
                "fps": display_count / max(elapsed, 1e-9),
            },
            "rois": {
                "candidate_stream": list(rois[0]),
                "confirm_stream": list(rois[1]),
                "candidate_sensor": list(global_rois[0]),
                "confirm_sensor": list(global_rois[1]),
            },
            "confirmed_counts": confirmed_counts,
            "confirmed_events": len(events),
            "outputs": {
                "video": str(output_video.resolve()) if args.record else None,
                "events_csv": str((args.output / "confirmed_events.csv").resolve()),
            },
            "host_tasks": [
                "Phantom SDK acquisition",
                "ROI crop and grayscale/resize/quantization",
                "sparse-head decode",
                "temporal tracking, A-to-B association, overlay and recording",
            ],
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        event_columns = [
            "event_id",
            "object_id",
            "class_name",
            "source_frame",
            "elapsed_seconds",
            "roi_1_result_index",
            "roi_2_result_index",
            "delay_updates",
            "cross_axis_distance_px",
            "roi_1_confidence",
            "roi_2_confidence",
            "cumulative_class_count",
        ]
        with (args.output / "confirmed_events.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer_csv = csv.DictWriter(file, fieldnames=event_columns)
            writer_csv.writeheader()
            writer_csv.writerows(events)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    finally:
        if capture is not None:
            capture.stop()
            capture.join(timeout=5.0)
        if worker is not None:
            worker.stop()
            worker.join(timeout=3.0)
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if capture is None or not capture.is_alive():
            if native_reader is not None:
                native_reader.close()
            camera.close()
            phantom.close()


if __name__ == "__main__":
    raise SystemExit(main())
