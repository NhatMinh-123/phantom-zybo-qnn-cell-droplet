#!/usr/bin/env python3
"""Run two downstream 15 um ROIs through Arty S7 and count verified crossings."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import (  # noqa: E402
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    prepare_image,
    unpack_sparse_detection_axis,
)
from scripts.dual_roi_temporal_counter import (  # noqa: E402
    ClassAwareLineTracker,
    CrossRoiAssociation,
    CrossRoiAssociator,
    DetectionSample,
    TrackObservation,
)
from scripts.run_video_finn_uart import (  # noqa: E402
    FrameDetection,
    InputTransform,
    map_roi_detections,
    scaled_roi,
)
from scripts.send_frame_finn_uart import detect_serial_port  # noqa: E402
from scripts.send_frame_finn_uart_sparse import transact_sparse  # noqa: E402


DEFAULT_VIDEO = (
    ROOT / "data" / "raw" / "09_07_2026" / "09_07_2026" / "3.5.mp4"
)
DEFAULT_MANIFEST = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "23_fpga_qnn96_dual_guard_v3"
    / "config"
    / "fpga_manifest_sparse_uart.json"
)
DEFAULT_CONFIG = ROOT / "configs" / "15micro_qnn_dual_roi120_downstream_v1.json"


@dataclass(frozen=True)
class FrameJob:
    frame_index: int
    submitted_at: float
    frame: np.ndarray


@dataclass(frozen=True)
class RegionHardwareResult:
    name: str
    detections: tuple[FrameDetection, ...]
    stream_cycles: int
    stream_fps: float
    round_trip_seconds: float
    sparse_bytes: int


@dataclass(frozen=True)
class DualHardwareResult:
    frame_index: int
    completed_at: float
    regions: tuple[RegionHardwareResult, ...]
    pair_round_trip_seconds: float

    def region(self, name: str) -> RegionHardwareResult:
        for item in self.regions:
            if item.name == name:
                return item
        raise KeyError(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_VIDEO))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=12_000_000)
    parser.add_argument("--clock-hz", type=int, default=108_000_000)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "15micro_fpga_dual_roi_counting",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-pace", action="store_true")
    return parser.parse_args()


def resolve_source(value: str) -> int | str:
    path = Path(value)
    if value.isdigit() and not path.exists():
        return int(value)
    return str(path.resolve())


def replace_pending_job(
    jobs: queue.Queue[FrameJob | None],
    job: FrameJob,
) -> bool:
    try:
        jobs.put_nowait(job)
        return False
    except queue.Full:
        pass
    try:
        jobs.get_nowait()
        jobs.task_done()
    except queue.Empty:
        pass
    jobs.put_nowait(job)
    return True


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_tracker(
    name: str,
    geometry: tuple[int, int, int, int],
    region_config: dict[str, Any],
    tracking: dict[str, Any],
) -> ClassAwareLineTracker:
    line_fraction = float(
        region_config.get("count_line_fraction", region_config.get("line_fraction", 0.65))
    )
    return ClassAwareLineTracker(
        region_name=name,
        roi_geometry=geometry,
        line_fraction=line_fraction,
        direction=str(tracking["direction"]),
        minimum_hits=tracking["minimum_hits"],
        max_misses=tracking["max_misses"],
        max_center_distance=tracking["max_center_distance"],
        count_hysteresis=float(tracking["count_hysteresis"]),
        reverse_tolerance=float(tracking.get("reverse_tolerance", 0.04)),
        edge_margin=float(tracking.get("edge_margin", 0.04)),
    )


def detection_samples(items: Iterable[FrameDetection]) -> list[DetectionSample]:
    return [
        DetectionSample(item.class_id, item.class_name, item.confidence, item.box)
        for item in items
    ]


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(box[0]))))
    y1 = max(0, min(height - 1, int(round(box[1]))))
    x2 = max(x1 + 1, min(width, int(round(box[2]))))
    y2 = max(y1 + 1, min(height, int(round(box[3]))))
    return x1, y1, x2, y2


def draw_dashed_rectangle(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    segment: int = 6,
) -> None:
    x1, y1, x2, y2 = box
    for start in range(x1, x2, segment * 2):
        cv2.line(image, (start, y1), (min(start + segment, x2), y1), color, thickness)
        cv2.line(image, (start, y2), (min(start + segment, x2), y2), color, thickness)
    for start in range(y1, y2, segment * 2):
        cv2.line(image, (x1, start), (x1, min(start + segment, y2)), color, thickness)
        cv2.line(image, (x2, start), (x2, min(start + segment, y2)), color, thickness)


def draw_track(
    image: np.ndarray,
    item: TrackObservation,
    *,
    downstream: bool,
) -> None:
    colors = {"cell": (35, 55, 235), "droplet": (235, 125, 25)}
    color = colors.get(item.class_name, (80, 210, 80))
    if not downstream:
        color = tuple(int(channel * 0.70) for channel in color)
    if item.predicted:
        color = (155, 155, 155)
    box = clamp_box(item.box, image.shape[1], image.shape[0])
    if item.predicted:
        draw_dashed_rectangle(image, box, color, 1)
    else:
        cv2.rectangle(image, box[:2], box[2:], color, 2 if downstream else 1, cv2.LINE_AA)
    center = (int(round(item.center[0])), int(round(item.center[1])))
    cv2.circle(image, center, 2, color, -1, cv2.LINE_AA)
    prefix = "C" if item.class_name == "cell" else "D"
    state = "*" if item.counted else ""
    cv2.putText(
        image,
        f"{prefix}{item.track_id}{state} {item.confidence:.2f}",
        (box[0] + 2, max(16, box[1] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    *,
    geometries: dict[str, tuple[int, int, int, int]],
    trackers: dict[str, ClassAwareLineTracker],
    current_frame_index: int,
    latest_result: DualHardwareResult | None,
    display_fps: float,
    pair_update_fps: float,
    maximum_prediction_frames: int,
    verified_counts: dict[str, int],
) -> np.ndarray:
    output = frame.copy()
    region_colors = {"detection": (45, 205, 45), "verification": (230, 175, 25)}
    labels = {"detection": "ROI A - DETECT", "verification": "ROI B - VERIFY / COUNT"}
    for name, geometry in geometries.items():
        x1, y1, x2, y2 = geometry
        color = region_colors[name]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            labels[name],
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        tracker = trackers[name]
        line_x = int(round(tracker.line_x))
        line_color = (0, 50, 245) if name == "verification" else (0, 190, 245)
        line_thickness = 3 if name == "verification" else 1
        cv2.line(output, (line_x, y1), (line_x, y2), line_color, line_thickness, cv2.LINE_AA)
        for item in tracker.observations(
            current_frame_index,
            maximum_prediction_frames=maximum_prediction_frames,
        ):
            draw_track(output, item, downstream=name == "verification")

    lag_frames = (
        max(0, current_frame_index - latest_result.frame_index)
        if latest_result is not None
        else 0
    )
    counts = trackers["verification"].counts
    cells = counts.get("cell", 0)
    droplets = counts.get("droplet", 0)
    verified_cell = verified_counts.get("cell", 0)
    verified_droplet = verified_counts.get("droplet", 0)
    cv2.rectangle(output, (0, 0), (output.shape[1], 78), (13, 13, 13), -1)
    if latest_result is None:
        first_line = "Arty S7-25 dual ROI | FPGA warming up"
        second_line = "Waiting for the first two sparse ROI responses"
    else:
        first_line = (
            f"Arty S7-25 QNN | display {display_fps:4.1f} FPS | "
            f"dual-ROI updates {pair_update_fps:4.1f} FPS | lag {lag_frames} frame"
        )
        second_line = (
            f"ROI-B crossing count: cell {cells} | droplet {droplets} | "
            f"A->B verified: cell {verified_cell} | droplet {verified_droplet}"
        )
    cv2.putText(
        output,
        first_line,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        second_line,
        (12, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 225, 90) if latest_result is not None else (40, 190, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "red line = one-shot count; dashed gray = coasted edge/missed detection",
        (12, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    region_configs = config["rois"]
    region_order = ("detection", "verification")
    tracking_config = config["tracking"]
    verification_config = config["cross_roi_verification"]

    source = resolve_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    geometries = {
        name: scaled_roi(width, height, region_configs[name]) for name in region_order
    }
    if geometries["verification"][0] <= geometries["detection"][0]:
        raise ValueError("Verification ROI must be downstream of detection ROI")

    resize = manifest["preprocessing"]["resize"]
    canvas_width = int(resize["width"])
    canvas_height = int(resize["height"])
    transform = InputTransform(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        content_width=canvas_width,
        content_height=canvas_height,
        offset_x=0,
        offset_y=0,
    )
    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])

    trackers = {
        name: make_tracker(
            name,
            geometries[name],
            region_configs[name],
            tracking_config,
        )
        for name in region_order
    }
    line_distance = trackers["verification"].line_x - trackers["detection"].line_x
    associator = CrossRoiAssociator(
        minimum_delay_frames=int(verification_config["minimum_delay_frames"]),
        maximum_delay_frames=int(verification_config["maximum_delay_frames"]),
        maximum_cross_axis_distance=(
            float(verification_config["maximum_cross_axis_distance_fraction"])
            * (geometries["verification"][3] - geometries["verification"][1])
        ),
        line_distance=line_distance,
    )

    if args.start_sec > 0.0 and not isinstance(source, int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(args.start_sec * source_fps))
    start_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    maximum_frames = (
        int(round(args.duration_sec * source_fps)) if args.duration_sec > 0.0 else 0
    )

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "fpga_dual_roi_counting.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    jobs: queue.Queue[FrameJob | None] = queue.Queue(maxsize=1)
    result_lock = threading.Lock()
    hardware_results: list[DualHardwareResult] = []
    worker_error: BaseException | None = None
    stop_event = threading.Event()
    port_name = args.port or detect_serial_port()

    def worker() -> None:
        nonlocal worker_error
        import serial

        try:
            with serial.Serial(
                port_name,
                args.baud,
                timeout=0.01,
                write_timeout=3.0,
            ) as port:
                try:
                    port.set_buffer_size(rx_size=1 << 20, tx_size=1 << 20)
                except (AttributeError, OSError):
                    pass
                port.reset_input_buffer()
                port.reset_output_buffer()
                transaction_id = 0
                while not stop_event.is_set():
                    try:
                        job = jobs.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if job is None:
                        jobs.task_done()
                        break
                    pair_started = time.perf_counter()
                    region_results: list[RegionHardwareResult] = []
                    try:
                        for name in region_order:
                            x1, y1, x2, y2 = geometries[name]
                            roi = job.frame[y1:y2, x1:x2]
                            codes = prepare_image(roi, manifest, array_is_bgr=True)
                            payload = pack_input_axis(codes, manifest)
                            transaction_id = (transaction_id + 1) & 0xFFFF
                            response, cycles, elapsed = transact_sparse(
                                port,
                                payload,
                                frame_id=transaction_id,
                                timeout=args.timeout,
                            )
                            tensor = unpack_sparse_detection_axis(response, manifest)
                            decoded = decode_output_tensor(tensor, manifest)[0]
                            mapped = tuple(
                                map_roi_detections(
                                    decoded,
                                    class_names,
                                    geometries[name],
                                    transform,
                                )
                            )
                            region_results.append(
                                RegionHardwareResult(
                                    name=name,
                                    detections=mapped,
                                    stream_cycles=cycles,
                                    stream_fps=args.clock_hz / cycles,
                                    round_trip_seconds=elapsed,
                                    sparse_bytes=len(response),
                                )
                            )
                        result = DualHardwareResult(
                            frame_index=job.frame_index,
                            completed_at=time.perf_counter(),
                            regions=tuple(region_results),
                            pair_round_trip_seconds=time.perf_counter() - pair_started,
                        )
                        with result_lock:
                            hardware_results.append(result)
                    finally:
                        jobs.task_done()
        except BaseException as error:
            worker_error = error
            stop_event.set()

    worker_thread = threading.Thread(
        target=worker,
        name="arty-s7-dual-roi-uart",
        daemon=True,
    )
    worker_thread.start()

    displayed = 0
    submitted = 0
    dropped_jobs = 0
    processed_result_count = 0
    display_started = time.perf_counter()
    display_lag_frames: list[int] = []
    detection_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    verified_counts = {name: 0 for name in class_names}
    upstream_event_count = {name: 0 for name in class_names}
    try:
        while maximum_frames <= 0 or displayed < maximum_frames:
            if stop_event.is_set():
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = start_frame + displayed
            submitted += 1
            dropped_jobs += int(
                replace_pending_job(
                    jobs,
                    FrameJob(frame_index, time.perf_counter(), frame.copy()),
                )
            )

            with result_lock:
                new_results = list(hardware_results[processed_result_count:])
                all_results = list(hardware_results)
            for result in new_results:
                upstream = result.region("detection")
                downstream = result.region("verification")
                upstream_events = trackers["detection"].update(
                    detection_samples(upstream.detections), result.frame_index
                )
                associator.add_upstream(upstream_events)
                for event in upstream_events:
                    upstream_event_count[event.class_name] += 1
                downstream_events = trackers["verification"].update(
                    detection_samples(downstream.detections), result.frame_index
                )
                associations: list[CrossRoiAssociation] = [
                    associator.associate(event) for event in downstream_events
                ]
                for association in associations:
                    event = association.downstream_event
                    if association.verified:
                        verified_counts[event.class_name] += 1
                    event_rows.append(
                        {
                            "event_index": len(event_rows) + 1,
                            "frame_index": event.frame_index,
                            "time_seconds": event.frame_index / source_fps,
                            "class_id": event.class_id,
                            "class_name": event.class_name,
                            "track_id": event.track_id,
                            "cumulative_count": event.cumulative_count,
                            "center_x": event.center_x,
                            "center_y": event.center_y,
                            "confidence": event.confidence,
                            "hits": event.hits,
                            "velocity_x": event.velocity_x,
                            "velocity_y": event.velocity_y,
                            "roi_a_verified": association.verified,
                            "roi_a_track_id": (
                                association.upstream_event.track_id
                                if association.upstream_event is not None
                                else ""
                            ),
                            "roi_a_delay_frames": association.delay_frames or "",
                            "roi_a_cross_axis_distance": (
                                association.cross_axis_distance
                                if association.cross_axis_distance is not None
                                else ""
                            ),
                        }
                    )
                for region in result.regions:
                    geometry = geometries[region.name]
                    edge_margin = float(tracking_config.get("edge_margin", 0.04)) * min(
                        geometry[2] - geometry[0], geometry[3] - geometry[1]
                    )
                    for item in region.detections:
                        x1, y1, x2, y2 = item.box
                        touches_edge = (
                            x1 <= geometry[0] + edge_margin
                            or y1 <= geometry[1] + edge_margin
                            or x2 >= geometry[2] - edge_margin
                            or y2 >= geometry[3] - edge_margin
                        )
                        detection_rows.append(
                            {
                                "frame_index": result.frame_index,
                                "time_seconds": result.frame_index / source_fps,
                                "region": region.name,
                                "class_id": item.class_id,
                                "class_name": item.class_name,
                                "confidence": item.confidence,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "touches_roi_edge": touches_edge,
                            }
                        )
                    for observation in trackers[region.name].observations(
                        result.frame_index,
                        maximum_prediction_frames=int(
                            tracking_config.get("maximum_prediction_frames", 4)
                        ),
                    ):
                        track_rows.append(
                            {
                                "frame_index": result.frame_index,
                                "time_seconds": result.frame_index / source_fps,
                                "region": region.name,
                                **asdict(observation),
                            }
                        )
            processed_result_count += len(new_results)

            latest_result = all_results[-1] if all_results else None
            recent_pair_times = [
                item.pair_round_trip_seconds for item in all_results[-30:]
            ]
            pair_update_fps = (
                1.0 / float(np.mean(recent_pair_times)) if recent_pair_times else 0.0
            )
            elapsed_display = max(time.perf_counter() - display_started, 1e-9)
            display_fps = displayed / elapsed_display if displayed else source_fps
            annotated = draw_overlay(
                frame,
                geometries=geometries,
                trackers=trackers,
                current_frame_index=frame_index,
                latest_result=latest_result,
                display_fps=display_fps,
                pair_update_fps=pair_update_fps,
                maximum_prediction_frames=int(
                    tracking_config.get("maximum_prediction_frames", 4)
                ),
                verified_counts=verified_counts,
            )
            writer.write(annotated)
            if args.show:
                cv2.imshow("Arty S7 Dual ROI Counting", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            displayed += 1
            lag = (
                max(0, frame_index - latest_result.frame_index)
                if latest_result is not None
                else 0
            )
            if latest_result is not None:
                display_lag_frames.append(lag)
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / source_fps,
                    "latest_fpga_frame": (
                        latest_result.frame_index if latest_result is not None else ""
                    ),
                    "display_lag_frames": lag,
                    "counted_cell": trackers["verification"].counts.get("cell", 0),
                    "counted_droplet": trackers["verification"].counts.get(
                        "droplet", 0
                    ),
                    "verified_cell": verified_counts.get("cell", 0),
                    "verified_droplet": verified_counts.get("droplet", 0),
                    "display_fps": display_fps,
                    "dual_roi_update_fps": pair_update_fps,
                }
            )

            if not args.no_pace and not isinstance(source, int):
                target = display_started + displayed / source_fps
                delay = target - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
            if displayed % max(1, round(source_fps)) == 0:
                counts = trackers["verification"].counts
                print(
                    f"display={displayed} source={display_fps:.1f}FPS "
                    f"dual_roi={pair_update_fps:.1f}FPS fpga_pairs={len(all_results)} "
                    f"drop={counts.get('droplet', 0)} cell={counts.get('cell', 0)} "
                    f"queue_drop={dropped_jobs}",
                    flush=True,
                )
    finally:
        stop_event.set()
        try:
            jobs.put_nowait(None)
        except queue.Full:
            try:
                jobs.get_nowait()
                jobs.task_done()
            except queue.Empty:
                pass
            jobs.put_nowait(None)
        worker_thread.join(timeout=args.timeout + 5.0)
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if worker_error is not None:
        raise RuntimeError("Dual-ROI sparse UART worker failed") from worker_error
    total_elapsed = time.perf_counter() - display_started
    with result_lock:
        results = list(hardware_results)
    pair_round_trips = [item.pair_round_trip_seconds for item in results]
    transactions = [region for result in results for region in result.regions]

    write_csv(
        output_dir / "hardware_detections.csv",
        detection_rows,
        [
            "frame_index",
            "time_seconds",
            "region",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "touches_roi_edge",
        ],
    )
    write_csv(
        output_dir / "track_observations.csv",
        track_rows,
        [
            "frame_index",
            "time_seconds",
            "region",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "box",
            "center",
            "velocity",
            "hits",
            "misses",
            "confirmed",
            "counted",
            "predicted",
            "last_seen_frame",
        ],
    )
    write_csv(
        output_dir / "count_events.csv",
        event_rows,
        [
            "event_index",
            "frame_index",
            "time_seconds",
            "class_id",
            "class_name",
            "track_id",
            "cumulative_count",
            "center_x",
            "center_y",
            "confidence",
            "hits",
            "velocity_x",
            "velocity_y",
            "roi_a_verified",
            "roi_a_track_id",
            "roi_a_delay_frames",
            "roi_a_cross_axis_distance",
        ],
    )
    write_csv(
        output_dir / "frame_summary.csv",
        frame_rows,
        [
            "frame_index",
            "time_seconds",
            "latest_fpga_frame",
            "display_lag_frames",
            "counted_cell",
            "counted_droplet",
            "verified_cell",
            "verified_droplet",
            "display_fps",
            "dual_roi_update_fps",
        ],
    )

    report: dict[str, Any] = {
        "source": str(args.source),
        "source_fps": source_fps,
        "displayed_frames": displayed,
        "display_elapsed_seconds": total_elapsed,
        "display_fps": displayed / total_elapsed if total_elapsed else 0.0,
        "fpga_frame_pairs": len(results),
        "fpga_roi_transactions": len(transactions),
        "dual_roi_update_fps": (
            1.0 / float(np.mean(pair_round_trips)) if pair_round_trips else 0.0
        ),
        "single_roi_transaction_fps": (
            1.0
            / float(np.mean([item.round_trip_seconds for item in transactions]))
            if transactions
            else 0.0
        ),
        "jobs_submitted": submitted,
        "stale_jobs_dropped": dropped_jobs,
        "port": port_name,
        "baud": args.baud,
        "clock_hz": args.clock_hz,
        "rois": {
            name: {
                "x": geometries[name][0],
                "y": geometries[name][1],
                "width": geometries[name][2] - geometries[name][0],
                "height": geometries[name][3] - geometries[name][1],
                "line_x": trackers[name].line_x,
                "role": (
                    "upstream detection and temporal association"
                    if name == "detection"
                    else "downstream verification and one-shot count"
                ),
            }
            for name in region_order
        },
        "counts": {
            "roi_b_crossings": {
                name: trackers["verification"].counts.get(name, 0)
                for name in class_names
            },
            "roi_a_to_b_verified": verified_counts,
            "roi_a_crossings": upstream_event_count,
        },
        "edge_recovery": {
            name: {
                "raw_detections_touching_roi_edge": trackers[name].edge_detections,
                "coasted_confirmed_track_updates": trackers[name].coasted_track_updates,
            }
            for name in region_order
        },
        "display_lag_frames": {
            "mean": float(np.mean(display_lag_frames)) if display_lag_frames else 0.0,
            "p95": (
                float(np.percentile(display_lag_frames, 95))
                if display_lag_frames
                else 0.0
            ),
        },
        "round_trip_ms": {
            "dual_roi_mean": (
                float(np.mean(pair_round_trips) * 1000.0)
                if pair_round_trips
                else 0.0
            ),
            "dual_roi_p95": (
                float(np.percentile(pair_round_trips, 95) * 1000.0)
                if pair_round_trips
                else 0.0
            ),
            "single_roi_mean": (
                float(np.mean([item.round_trip_seconds for item in transactions]) * 1000.0)
                if transactions
                else 0.0
            ),
        },
        "sparse_payload_bytes": {
            "mean": (
                float(np.mean([item.sparse_bytes for item in transactions]))
                if transactions
                else 0.0
            ),
            "maximum": int(max((item.sparse_bytes for item in transactions), default=0)),
        },
        "output_video": str(output_video),
        "artifacts": {
            "hardware_detections": str(output_dir / "hardware_detections.csv"),
            "track_observations": str(output_dir / "track_observations.csv"),
            "count_events": str(output_dir / "count_events.csv"),
            "frame_summary": str(output_dir / "frame_summary.csv"),
        },
        "truth_boundary": {
            "fpga": (
                "Both 120x120 ROI crops are time-multiplexed through the programmed "
                "Arty S7-25 QNN, output requantization, radial guard, thresholding, "
                "and sparse serialization."
            ),
            "host_validation_stage": (
                "MP4 decode, ROI crop/transport, temporal association, line crossing "
                "state, CSV logging, and rendering. The validated tracker/count rules "
                "are the reference for the subsequent RTL FSM implementation."
            ),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"FPGA_DUAL_ROI_COUNTING_PASS: {output_video}")


if __name__ == "__main__":
    main()
