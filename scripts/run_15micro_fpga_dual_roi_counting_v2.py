#!/usr/bin/env python3
"""Prioritize downstream FPGA inference and sample the upstream verification ROI."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qnn.fpga_io import (
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    prepare_image,
    unpack_sparse_detection_axis,
)
from scripts import run_15micro_fpga_dual_roi_counting as base
from scripts.dual_roi_temporal_counter import CrossRoiAssociator
from scripts.run_video_finn_uart import InputTransform, map_roi_detections, scaled_roi
from scripts.send_frame_finn_uart import detect_serial_port
from scripts.send_frame_finn_uart_sparse import transact_sparse


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(base.DEFAULT_VIDEO))
    parser.add_argument("--manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=12_000_000)
    parser.add_argument("--clock-hz", type=int, default=108_000_000)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--upstream-stride", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "15micro_fpga_dual_roi_counting_v2",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-pace", action="store_true")
    return parser.parse_args()


def optional_region(
    result: base.DualHardwareResult,
    name: str,
) -> base.RegionHardwareResult | None:
    for item in result.regions:
        if item.name == name:
            return item
    return None


def main() -> None:
    args = parse_args()
    if args.upstream_stride < 1:
        raise ValueError("upstream-stride must be at least one")
    manifest = load_manifest(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    region_configs = config["rois"]
    tracking_config = config["tracking"]
    verification_config = config["cross_roi_verification"]
    region_order = ("detection", "verification")

    source = base.resolve_source(args.source)
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
        canvas_width,
        canvas_height,
        canvas_width,
        canvas_height,
        0,
        0,
    )
    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    trackers = {
        name: base.make_tracker(
            name,
            geometries[name],
            region_configs[name],
            tracking_config,
        )
        for name in region_order
    }
    associator = CrossRoiAssociator(
        minimum_delay_frames=int(verification_config["minimum_delay_frames"]),
        maximum_delay_frames=int(verification_config["maximum_delay_frames"]),
        maximum_cross_axis_distance=(
            float(verification_config["maximum_cross_axis_distance_fraction"])
            * (geometries["verification"][3] - geometries["verification"][1])
        ),
        line_distance=trackers["verification"].line_x - trackers["detection"].line_x,
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

    jobs: queue.Queue[base.FrameJob | None] = queue.Queue(maxsize=1)
    result_lock = threading.Lock()
    hardware_results: list[base.DualHardwareResult] = []
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
                processed_jobs = 0
                while not stop_event.is_set():
                    try:
                        job = jobs.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if job is None:
                        jobs.task_done()
                        break
                    include_upstream = processed_jobs % args.upstream_stride == 0
                    names = (
                        ("detection", "verification")
                        if include_upstream
                        else ("verification",)
                    )
                    pair_started = time.perf_counter()
                    region_results: list[base.RegionHardwareResult] = []
                    try:
                        for name in names:
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
                            detections = tuple(
                                map_roi_detections(
                                    decoded,
                                    class_names,
                                    geometries[name],
                                    transform,
                                )
                            )
                            region_results.append(
                                base.RegionHardwareResult(
                                    name,
                                    detections,
                                    cycles,
                                    args.clock_hz / cycles,
                                    elapsed,
                                    len(response),
                                )
                            )
                        result = base.DualHardwareResult(
                            job.frame_index,
                            time.perf_counter(),
                            tuple(region_results),
                            time.perf_counter() - pair_started,
                        )
                        with result_lock:
                            hardware_results.append(result)
                        processed_jobs += 1
                    finally:
                        jobs.task_done()
        except BaseException as error:
            worker_error = error
            stop_event.set()

    worker_thread = threading.Thread(
        target=worker,
        name="arty-s7-priority-dual-roi-uart",
        daemon=True,
    )
    worker_thread.start()

    detection_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    verified_counts = {name: 0 for name in class_names}
    upstream_event_count = {name: 0 for name in class_names}
    processed_result_count = 0

    def process_result(result: base.DualHardwareResult) -> None:
        upstream = optional_region(result, "detection")
        downstream = optional_region(result, "verification")
        if downstream is None:
            raise RuntimeError("A scheduled result is missing the verification ROI")
        if upstream is not None:
            upstream_events = trackers["detection"].update(
                base.detection_samples(upstream.detections), result.frame_index
            )
            associator.add_upstream(upstream_events)
            for event in upstream_events:
                upstream_event_count[event.class_name] += 1
        downstream_events = trackers["verification"].update(
            base.detection_samples(downstream.detections), result.frame_index
        )
        for event in downstream_events:
            association = associator.associate(event)
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
                    "roi_a_delay_frames": (
                        association.delay_frames
                        if association.delay_frames is not None
                        else ""
                    ),
                    "roi_a_cross_axis_distance": (
                        association.cross_axis_distance
                        if association.cross_axis_distance is not None
                        else ""
                    ),
                }
            )
        for region in result.regions:
            geometry = geometries[region.name]
            margin = float(tracking_config.get("edge_margin", 0.04)) * min(
                geometry[2] - geometry[0], geometry[3] - geometry[1]
            )
            for item in region.detections:
                x1, y1, x2, y2 = item.box
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
                        "touches_roi_edge": (
                            x1 <= geometry[0] + margin
                            or y1 <= geometry[1] + margin
                            or x2 >= geometry[2] - margin
                            or y2 >= geometry[3] - margin
                        ),
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

    displayed = 0
    submitted = 0
    dropped_jobs = 0
    display_started = time.perf_counter()
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
                base.replace_pending_job(
                    jobs,
                    base.FrameJob(frame_index, time.perf_counter(), frame.copy()),
                )
            )
            with result_lock:
                new_results = list(hardware_results[processed_result_count:])
                all_results = list(hardware_results)
            for result in new_results:
                process_result(result)
            processed_result_count += len(new_results)

            latest_result = all_results[-1] if all_results else None
            recent_times = [item.pair_round_trip_seconds for item in all_results[-30:]]
            verification_update_fps = (
                1.0 / float(np.mean(recent_times)) if recent_times else 0.0
            )
            elapsed_display = max(time.perf_counter() - display_started, 1e-9)
            display_fps = displayed / elapsed_display if displayed else source_fps
            annotated = base.draw_overlay(
                frame,
                geometries=geometries,
                trackers=trackers,
                current_frame_index=frame_index,
                latest_result=latest_result,
                display_fps=display_fps,
                pair_update_fps=verification_update_fps,
                maximum_prediction_frames=int(
                    tracking_config.get("maximum_prediction_frames", 4)
                ),
                verified_counts=verified_counts,
            )
            writer.write(annotated)
            if args.show:
                cv2.imshow("Arty S7 Priority Dual ROI Counting", annotated)
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
                    "verification_roi_update_fps": verification_update_fps,
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
                    f"roi_b={verification_update_fps:.1f}FPS results={len(all_results)} "
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
        raise RuntimeError("Priority dual-ROI sparse UART worker failed") from worker_error
    with result_lock:
        results = list(hardware_results)
    for result in results[processed_result_count:]:
        process_result(result)
    processed_result_count = len(results)
    total_elapsed = time.perf_counter() - display_started
    transactions = [region for result in results for region in result.regions]
    upstream_transactions = [
        item for item in transactions if item.name == "detection"
    ]
    downstream_transactions = [
        item for item in transactions if item.name == "verification"
    ]
    result_times = [item.pair_round_trip_seconds for item in results]

    base.write_csv(
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
    base.write_csv(
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
    base.write_csv(
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
    base.write_csv(
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
            "verification_roi_update_fps",
        ],
    )

    report: dict[str, Any] = {
        "source": str(args.source),
        "source_fps": source_fps,
        "displayed_frames": displayed,
        "display_elapsed_seconds": total_elapsed,
        "display_fps": displayed / total_elapsed if total_elapsed else 0.0,
        "fpga_results": len(results),
        "fpga_roi_transactions": len(transactions),
        "verification_roi_transactions": len(downstream_transactions),
        "upstream_roi_transactions": len(upstream_transactions),
        "upstream_stride": args.upstream_stride,
        "verification_roi_update_fps": (
            1.0 / float(np.mean(result_times)) if result_times else 0.0
        ),
        "raw_single_roi_transaction_fps": (
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
                "schedule": (
                    f"sampled every {args.upstream_stride} FPGA jobs"
                    if name == "detection"
                    else "processed on every FPGA job and owns the count"
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
            "scheduled_job_mean": (
                float(np.mean(result_times) * 1000.0) if result_times else 0.0
            ),
            "scheduled_job_p95": (
                float(np.percentile(result_times, 95) * 1000.0)
                if result_times
                else 0.0
            ),
            "single_transaction_mean": (
                float(np.mean([item.round_trip_seconds for item in transactions]) * 1000.0)
                if transactions
                else 0.0
            ),
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
                "Every ROI-B sample and each scheduled ROI-A sample run through "
                "the programmed Arty S7-25 QNN, requantization, radial guard, "
                "thresholding, and sparse serializer."
            ),
            "host_validation_stage": (
                "MP4 decode, ROI crop/transport, temporal tracking, A-to-B "
                "association, line crossing count, logging, and rendering."
            ),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"FPGA_PRIORITY_DUAL_ROI_COUNTING_PASS: {output_video}")


if __name__ == "__main__":
    main()
