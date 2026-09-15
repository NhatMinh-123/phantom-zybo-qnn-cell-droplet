#!/usr/bin/env python3
"""Play video smoothly while Arty S7 processes the newest ROI asynchronously."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import (
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    prepare_image,
    unpack_sparse_detection_axis,
)
from scripts.run_video_finn_uart import (
    FrameDetection,
    InputTransform,
    map_roi_detections,
    prepare_roi_input,
    scaled_roi,
)
from scripts.send_frame_finn_uart import detect_serial_port
from scripts.send_frame_finn_uart_sparse import transact_sparse


DEFAULT_VIDEO = ROOT / "data" / "raw" / "3.4.mp4"
DEFAULT_MANIFEST = (
    ROOT
    / "final_results"
    / "arty_s7_25_qnn_detection"
    / "07_50fps_optimized"
    / "fpga_manifest_60fps.json"
)
DEFAULT_ROI_CONFIG = (
    ROOT / "models" / "cell_droplet_yolo11n" / "roi384_baseline_config.json"
)


@dataclass(frozen=True)
class FrameJob:
    frame_index: int
    submitted_at: float
    frame: np.ndarray


@dataclass(frozen=True)
class HardwareResult:
    frame_index: int
    completed_at: float
    detections: tuple[FrameDetection, ...]
    stream_cycles: int
    stream_fps: float
    round_trip_seconds: float
    sparse_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_VIDEO))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=12_000_000)
    parser.add_argument("--clock-hz", type=int, default=108_000_000)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "fpga_sparse_realtime_video",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--no-pace",
        action="store_true",
        help="Do not pace video files to their source frame rate",
    )
    parser.add_argument(
        "--no-motion-compensation",
        action="store_true",
        help="Draw the latest FPGA boxes without short-horizon motion prediction",
    )
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
    dropped = False
    try:
        jobs.put_nowait(job)
        return dropped
    except queue.Full:
        pass
    try:
        jobs.get_nowait()
        jobs.task_done()
        dropped = True
    except queue.Empty:
        pass
    jobs.put_nowait(job)
    return dropped


def detection_center(item: FrameDetection) -> tuple[float, float]:
    x1, y1, x2, y2 = item.box
    return (x1 + x2) / 2, (y1 + y2) / 2


def estimate_detection_velocity(
    previous: HardwareResult | None,
    current: HardwareResult | None,
    *,
    maximum_distance: float = 80.0,
    maximum_vertical_distance: float = 30.0,
) -> tuple[float, float, int]:
    if previous is None or current is None:
        return 0.0, 0.0, 0
    frame_delta = current.frame_index - previous.frame_index
    if frame_delta <= 0:
        return 0.0, 0.0, 0
    previous_items = [
        item for item in previous.detections if item.class_name == "droplet"
    ]
    current_items = [
        item for item in current.detections if item.class_name == "droplet"
    ]
    if not previous_items or not current_items:
        return 0.0, 0.0, 0

    available = set(range(len(previous_items)))
    displacements: list[tuple[float, float]] = []
    for item in sorted(
        current_items,
        key=lambda candidate: candidate.confidence,
        reverse=True,
    ):
        current_x, current_y = detection_center(item)
        choices: list[tuple[float, int, float, float]] = []
        for index in available:
            previous_x, previous_y = detection_center(previous_items[index])
            dx = current_x - previous_x
            dy = current_y - previous_y
            distance = float(np.hypot(dx, dy))
            if (
                distance <= maximum_distance
                and abs(dy) <= maximum_vertical_distance
            ):
                choices.append((distance, index, dx, dy))
        if not choices:
            continue
        _, matched_index, dx, dy = min(choices)
        available.remove(matched_index)
        displacements.append((dx / frame_delta, dy / frame_delta))
    if not displacements:
        return 0.0, 0.0, 0
    velocity_x = float(np.median([item[0] for item in displacements]))
    velocity_y = float(np.median([item[1] for item in displacements]))
    return velocity_x, velocity_y, len(displacements)

def draw_overlay(
    frame: np.ndarray,
    *,
    roi_geometry: tuple[int, int, int, int],
    result: HardwareResult | None,
    motion_per_frame: tuple[float, float, int],
    motion_compensation: bool,
    current_frame_index: int,
    source_fps: float,
    update_fps: float,
    display_fps: float,
) -> np.ndarray:
    output = frame.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry
    cv2.rectangle(output, (roi_x1, roi_y1), (roi_x2, roi_y2), (30, 210, 40), 2)
    colors = {"cell": (35, 45, 230), "droplet": (235, 125, 25)}
    detections = result.detections if result is not None else ()
    lag_frames = (
        max(0, current_frame_index - result.frame_index)
        if result is not None
        else 0
    )
    velocity_x, velocity_y, motion_matches = motion_per_frame
    prediction_horizon = min(lag_frames, 4) if motion_compensation else 0
    shift_x = velocity_x * prediction_horizon
    shift_y = velocity_y * prediction_horizon
    for item in detections:
        x1, y1, x2, y2 = (
            int(round(value))
            for value in (
                item.box[0] + shift_x,
                item.box[1] + shift_y,
                item.box[2] + shift_x,
                item.box[3] + shift_y,
            )
        )
        x1 = max(roi_x1, min(roi_x2 - 1, x1))
        x2 = max(roi_x1 + 1, min(roi_x2, x2))
        y1 = max(roi_y1, min(roi_y2 - 1, y1))
        y2 = max(roi_y1 + 1, min(roi_y2, y2))
        color = colors.get(item.class_name, (40, 200, 40))
        thickness = 1 if item.class_name == "cell" else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if item.class_name == "droplet":
            cv2.putText(
                output,
                f"D {item.confidence:.2f}",
                (x1 + 2, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )

    cells = sum(item.class_name == "cell" for item in detections)
    droplets = sum(item.class_name == "droplet" for item in detections)
    if result is None:
        status = "FPGA warming up"
        detail = "waiting for first sparse response"
    else:
        lag_ms = lag_frames / max(source_fps, 1e-9) * 1000.0
        status = (
            f"display {display_fps:4.1f} FPS | FPGA updates {update_fps:4.1f} FPS | "
            f"stream {result.stream_fps:5.2f} FPS"
        )
        detail = (
            f"12 Mbaud sparse {result.sparse_bytes} B | lag {lag_frames} frames/"
            f"{lag_ms:.0f} ms | motion {motion_matches} | "
            f"cell {cells} droplet {droplets}"
        )
    cv2.rectangle(output, (0, 0), (output.shape[1], 58), (14, 14, 14), -1)
    cv2.putText(
        output,
        status,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        detail,
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 230, 80) if result is not None else (40, 190, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    runtime_config = json.loads(args.roi_config.read_text(encoding="utf-8"))
    roi_config = runtime_config["roi"]
    preprocess_config = runtime_config["preprocess"]
    resize = manifest["preprocessing"]["resize"]
    canvas_width = int(resize["width"])
    canvas_height = int(resize["height"])
    source_image_size = int(runtime_config["image_size"])
    content_width = int(
        round(
            canvas_width
            * int(preprocess_config["content_width"])
            / source_image_size
        )
    )
    content_height = int(
        round(
            canvas_height
            * int(preprocess_config["content_height"])
            / source_image_size
        )
    )
    transform = InputTransform(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        content_width=content_width,
        content_height=content_height,
        offset_x=(canvas_width - content_width) // 2,
        offset_y=(canvas_height - content_height) // 2,
    )
    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])

    source = resolve_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_geometry = scaled_roi(width, height, roi_config)
    if args.start_sec > 0.0 and not isinstance(source, int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(args.start_sec * source_fps))
    start_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    maximum_frames = (
        int(round(args.duration_sec * source_fps))
        if args.duration_sec > 0.0
        else 0
    )

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "fpga_sparse_realtime.mp4"
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
    latest_result: HardwareResult | None = None
    hardware_results: list[HardwareResult] = []
    worker_error: BaseException | None = None
    stop_event = threading.Event()
    port_name = args.port or detect_serial_port()

    def worker() -> None:
        nonlocal latest_result, worker_error
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
                    roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry
                    roi = job.frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    model_input, _ = prepare_roi_input(
                        roi,
                        canvas_width,
                        canvas_height,
                        content_width,
                        content_height,
                    )
                    codes = prepare_image(model_input, manifest, array_is_bgr=True)
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
                    detections = tuple(
                        map_roi_detections(
                            decoded,
                            class_names,
                            roi_geometry,
                            transform,
                        )
                    )
                    result = HardwareResult(
                        frame_index=job.frame_index,
                        completed_at=time.perf_counter(),
                        detections=detections,
                        stream_cycles=cycles,
                        stream_fps=args.clock_hz / cycles,
                        round_trip_seconds=elapsed,
                        sparse_bytes=len(response),
                    )
                    with result_lock:
                        latest_result = result
                        hardware_results.append(result)
                    jobs.task_done()
        except BaseException as error:
            worker_error = error
            stop_event.set()

    worker_thread = threading.Thread(
        target=worker,
        name="arty-s7-sparse-uart",
        daemon=True,
    )
    worker_thread.start()

    displayed = 0
    dropped_jobs = 0
    submitted = 0
    display_started = time.perf_counter()
    display_tick_times: list[float] = []
    display_lag_frames: list[int] = []
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
                result = latest_result
                completed = len(hardware_results)
                previous_result = (
                    hardware_results[-2]
                    if len(hardware_results) >= 2
                    else None
                )
                recent_round_trips = [
                    item.round_trip_seconds for item in hardware_results[-30:]
                ]
            motion_per_frame = estimate_detection_velocity(
                previous_result,
                result,
            )
            update_fps = (
                1.0 / float(np.mean(recent_round_trips))
                if recent_round_trips
                else 0.0
            )
            now = time.perf_counter()
            elapsed_display = max(now - display_started, 1e-9)
            display_fps = displayed / elapsed_display if displayed else source_fps
            annotated = draw_overlay(
                frame,
                roi_geometry=roi_geometry,
                result=result,
                motion_per_frame=motion_per_frame,
                motion_compensation=not args.no_motion_compensation,
                current_frame_index=frame_index,
                source_fps=source_fps,
                update_fps=update_fps,
                display_fps=display_fps,
            )
            writer.write(annotated)
            if args.show:
                cv2.imshow("Arty S7 Sparse Realtime", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            displayed += 1
            display_tick_times.append(time.perf_counter())
            if result is not None:
                display_lag_frames.append(
                    max(0, frame_index - result.frame_index)
                )

            if not args.no_pace and not isinstance(source, int):
                target = display_started + displayed / source_fps
                delay = target - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
            if displayed % max(1, round(source_fps)) == 0:
                print(
                    f"display={displayed} source={display_fps:.1f}FPS "
                    f"fpga_results={completed} update={update_fps:.1f}FPS "
                    f"dropped_queue={dropped_jobs}",
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
        worker_thread.join(timeout=args.timeout + 2.0)
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if worker_error is not None:
        raise RuntimeError("Sparse UART worker failed") from worker_error
    total_elapsed = time.perf_counter() - display_started
    with result_lock:
        results = list(hardware_results)
    round_trips = [item.round_trip_seconds for item in results]
    report: dict[str, Any] = {
        "source": args.source,
        "source_fps": source_fps,
        "displayed_frames": displayed,
        "display_elapsed_seconds": total_elapsed,
        "display_fps": displayed / total_elapsed if total_elapsed else 0.0,
        "fpga_results": len(results),
        "fpga_update_fps": (
            1.0 / float(np.mean(round_trips)) if round_trips else 0.0
        ),
        "jobs_submitted": submitted,
        "stale_jobs_dropped": dropped_jobs,
        "motion_compensation": {
            "enabled": not args.no_motion_compensation,
            "maximum_prediction_horizon_frames": 4,
            "mean_display_lag_frames": (
                float(np.mean(display_lag_frames))
                if display_lag_frames
                else 0.0
            ),
            "p95_display_lag_frames": (
                float(np.percentile(display_lag_frames, 95))
                if display_lag_frames
                else 0.0
            ),
        },
        "port": port_name,
        "baud": args.baud,
        "clock_hz": args.clock_hz,
        "uart_paced_stream_fps": {
            "mean": float(np.mean([item.stream_fps for item in results]))
            if results
            else 0.0,
            "minimum": float(np.min([item.stream_fps for item in results]))
            if results
            else 0.0,
        },
        "timing_scope": (
            "First FPGA input handshake through final detector output; includes "
            "UART-paced input arrival and excludes host serial API overhead."
        ),
        "round_trip_ms": {
            "mean": float(np.mean(round_trips) * 1000.0) if results else 0.0,
            "p95": float(np.percentile(round_trips, 95) * 1000.0)
            if results
            else 0.0,
        },
        "sparse_payload_bytes": {
            "mean": float(np.mean([item.sparse_bytes for item in results]))
            if results
            else 0.0,
            "maximum": int(max((item.sparse_bytes for item in results), default=0)),
        },
        "output_video": str(output_video),
        "architecture": (
            "30-FPS nonblocking display with a latest-frame queue; FPGA sparse "
            "detections update independently, stale inference jobs are dropped, "
            "and short-horizon droplet motion predicts display-only box positions."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"FPGA_SPARSE_REALTIME_PASS: {output_video}")


if __name__ == "__main__":
    main()
