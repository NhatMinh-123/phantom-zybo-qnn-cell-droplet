#!/usr/bin/env python3
"""Run upstream FPGA QNN and downstream classical ROI with strict AND fusion."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

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
from scripts.dual_branch_radial_v2 import RadialConfig
from scripts.dual_roi_hybrid_fusion import (
    ClassicalBranchConfig,
    CrossBranchConsensus,
    DownstreamClassicalDetector,
    HybridCountEvent,
    HybridFusionConfig,
)
from scripts.dual_roi_temporal_counter import DetectionSample
from scripts.run_video_finn_uart import InputTransform, map_roi_detections, scaled_roi
from scripts.send_frame_finn_uart import detect_serial_port
from scripts.send_frame_finn_uart_sparse import transact_sparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "15micro_hybrid_qnn_roiA_classical_roiB_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(base.DEFAULT_VIDEO))
    parser.add_argument("--manifest", type=Path, default=base.DEFAULT_MANIFEST)
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
        default=ROOT / "reports" / "15micro_fpga_qnn_classical_fusion",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-pace", action="store_true")
    return parser.parse_args()


def make_classical_config(raw: dict[str, Any]) -> ClassicalBranchConfig:
    cell = RadialConfig(**raw["cell"])
    values = {key: value for key, value in raw.items() if key != "cell"}
    return ClassicalBranchConfig(**values, cell=cell)


def make_fusion_config(raw: dict[str, Any], roi_height: int) -> HybridFusionConfig:
    return HybridFusionConfig(
        minimum_delay_frames=int(raw["minimum_delay_frames"]),
        maximum_delay_frames=int(raw["maximum_delay_frames"]),
        maximum_cross_axis_distance=(
            float(raw["maximum_cross_axis_distance_fraction"]) * roi_height
        ),
        nominal_delay_frames=float(raw["nominal_delay_frames"]),
    )


def draw_sample(
    frame: np.ndarray,
    detection: DetectionSample,
    color: tuple[int, int, int],
    label_prefix: str,
) -> None:
    x1, y1, x2, y2 = base.clamp_box(detection.box, frame.shape[1], frame.shape[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    if detection.class_name == "cell":
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 2, color, -1)
    cv2.putText(
        frame,
        f"{label_prefix}:{detection.class_name[0].upper()} {detection.confidence:.2f}",
        (x1, max(14, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    *,
    qnn_geometry: tuple[int, int, int, int],
    classical_geometry: tuple[int, int, int, int],
    qnn_tracker: Any,
    classical_tracker: Any,
    frame_index: int,
    qnn_detections: Iterable[Any],
    classical_detections: list[DetectionSample],
    counts: dict[str, int],
    qnn_fps: float,
    classical_fps: float,
    display_fps: float,
    pending_qnn: int,
    pending_classical: int,
    latest_fusion: HybridCountEvent | None,
) -> np.ndarray:
    output = frame.copy()
    qnn_color = (235, 145, 25)
    classical_color = (35, 205, 235)
    accepted_color = (40, 220, 80)
    for geometry, color, label, tracker in (
        (qnn_geometry, qnn_color, "ROI A: FPGA QNN", qnn_tracker),
        (classical_geometry, classical_color, "ROI B: IMAGE PROCESS", classical_tracker),
    ):
        x1, y1, x2, y2 = geometry
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
        line_x = int(round(tracker.line_x))
        cv2.line(output, (line_x, y1), (line_x, y2), color, 2, cv2.LINE_AA)

    for detection in qnn_detections:
        draw_sample(
            output,
            DetectionSample(
                detection.class_id,
                detection.class_name,
                detection.confidence,
                detection.box,
            ),
            qnn_color,
            "Q",
        )
    for detection in classical_detections:
        draw_sample(output, detection, classical_color, "I")

    cv2.rectangle(output, (0, 0), (output.shape[1], 76), (13, 13, 13), -1)
    cv2.putText(
        output,
        (
            f"Arty S7-25 hybrid | display {display_fps:.1f} FPS | "
            f"QNN FPGA {qnn_fps:.1f} FPS | image {classical_fps:.1f} FPS"
        ),
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        (
            f"AND accepted: droplet {counts.get('droplet', 0)} | "
            f"cell {counts.get('cell', 0)} | pending Q/I {pending_qnn}/{pending_classical}"
        ),
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        accepted_color,
        1,
        cv2.LINE_AA,
    )
    detail = "QNN candidate must match downstream image-processing event"
    if latest_fusion is not None:
        detail = (
            f"last accepted {latest_fusion.class_name} | delay "
            f"{latest_fusion.delay_frames} frames | fused "
            f"{latest_fusion.fused_confidence:.2f}"
        )
    cv2.putText(
        output,
        detail,
        (12, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = base.resolve_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    qnn_geometry = scaled_roi(width, height, config["rois"]["qnn"])
    classical_geometry = scaled_roi(width, height, config["rois"]["classical"])
    if classical_geometry[0] <= qnn_geometry[0]:
        raise ValueError("Classical ROI must be downstream of the QNN ROI")

    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    class_ids = {name: index for index, name in enumerate(class_names)}
    classical_detector = DownstreamClassicalDetector(
        roi_geometry=classical_geometry,
        class_ids=class_ids,
        config=make_classical_config(config["classical_detector"]),
    )
    qnn_tracker = base.make_tracker(
        "qnn",
        qnn_geometry,
        config["rois"]["qnn"],
        config["qnn_tracking"],
    )
    classical_tracker = base.make_tracker(
        "classical",
        classical_geometry,
        config["rois"]["classical"],
        config["classical_tracking"],
    )
    fusion = CrossBranchConsensus(
        make_fusion_config(
            config["fusion"], classical_geometry[3] - classical_geometry[1]
        )
    )

    resize = manifest["preprocessing"]["resize"]
    transform = InputTransform(
        int(resize["width"]),
        int(resize["height"]),
        int(resize["width"]),
        int(resize["height"]),
        0,
        0,
    )
    if args.start_sec > 0.0 and not isinstance(source, int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(args.start_sec * source_fps))
    start_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    maximum_frames = (
        int(round(args.duration_sec * source_fps)) if args.duration_sec > 0.0 else 0
    )
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "fpga_qnn_classical_fusion.mp4"
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
    qnn_results: list[base.DualHardwareResult] = []
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
                    started = time.perf_counter()
                    try:
                        x1, y1, x2, y2 = qnn_geometry
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
                                qnn_geometry,
                                transform,
                            )
                        )
                        region = base.RegionHardwareResult(
                            "qnn",
                            detections,
                            cycles,
                            args.clock_hz / cycles,
                            elapsed,
                            len(response),
                        )
                        result = base.DualHardwareResult(
                            job.frame_index,
                            time.perf_counter(),
                            (region,),
                            time.perf_counter() - started,
                        )
                        with result_lock:
                            qnn_results.append(result)
                    finally:
                        jobs.task_done()
        except BaseException as error:
            worker_error = error
            stop_event.set()

    worker_thread = threading.Thread(
        target=worker,
        name="arty-s7-qnn-roi-a-uart",
        daemon=True,
    )
    worker_thread.start()

    qnn_detection_rows: list[dict[str, Any]] = []
    classical_detection_rows: list[dict[str, Any]] = []
    branch_event_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    classical_times: list[float] = []
    qnn_result_index = 0
    latest_qnn_detections: tuple[Any, ...] = ()
    latest_classical_detections: list[DetectionSample] = []
    latest_fusion: HybridCountEvent | None = None
    raw_qnn_crossings = {name: 0 for name in class_names}
    raw_classical_crossings = {name: 0 for name in class_names}

    def record_fusions(events: Iterable[HybridCountEvent]) -> None:
        nonlocal latest_fusion
        for event in events:
            latest_fusion = event
            fusion_rows.append(asdict(event))

    def process_qnn(result: base.DualHardwareResult, current_frame: int) -> None:
        nonlocal latest_qnn_detections
        region = result.regions[0]
        latest_qnn_detections = region.detections
        for item in region.detections:
            qnn_detection_rows.append(
                {
                    "frame_index": result.frame_index,
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                    "x1": item.box[0],
                    "y1": item.box[1],
                    "x2": item.box[2],
                    "y2": item.box[3],
                }
            )
        events = qnn_tracker.update(
            base.detection_samples(region.detections), result.frame_index
        )
        for event in events:
            raw_qnn_crossings[event.class_name] += 1
            branch_event_rows.append({"branch": "qnn_fpga", **asdict(event)})
        record_fusions(fusion.add_qnn(events, current_frame))

    displayed = 0
    submitted = 0
    dropped_jobs = 0
    display_started = time.perf_counter()
    try:
        while maximum_frames <= 0 or displayed < maximum_frames:
            if stop_event.is_set():
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = start_frame + displayed

            classical_started = time.perf_counter()
            x1, y1, x2, y2 = classical_geometry
            latest_classical_detections, diagnostics = classical_detector.detect(
                frame[y1:y2, x1:x2]
            )
            classical_times.append(time.perf_counter() - classical_started)
            classical_events = classical_tracker.update(
                latest_classical_detections, frame_index
            )
            for item in latest_classical_detections:
                classical_detection_rows.append(
                    {
                        "frame_index": frame_index,
                        "class_id": item.class_id,
                        "class_name": item.class_name,
                        "confidence": item.confidence,
                        "x1": item.box[0],
                        "y1": item.box[1],
                        "x2": item.box[2],
                        "y2": item.box[3],
                        "droplet_candidates": diagnostics["droplet_candidates"],
                        "cell_candidates": diagnostics["cell_candidates"],
                    }
                )
            for event in classical_events:
                raw_classical_crossings[event.class_name] += 1
                branch_event_rows.append(
                    {"branch": "classical_downstream", **asdict(event)}
                )
            record_fusions(fusion.add_classical(classical_events, frame_index))

            submitted += 1
            dropped_jobs += int(
                base.replace_pending_job(
                    jobs,
                    base.FrameJob(frame_index, time.perf_counter(), frame.copy()),
                )
            )
            with result_lock:
                new_results = list(qnn_results[qnn_result_index:])
                all_results = list(qnn_results)
            for result in new_results:
                process_qnn(result, frame_index)
            qnn_result_index += len(new_results)

            display_elapsed = max(time.perf_counter() - display_started, 1e-9)
            display_fps = displayed / display_elapsed if displayed else source_fps
            qnn_times = [item.pair_round_trip_seconds for item in all_results[-30:]]
            qnn_fps = 1.0 / float(np.mean(qnn_times)) if qnn_times else 0.0
            classical_fps = (
                1.0 / float(np.mean(classical_times[-60:]))
                if classical_times
                else 0.0
            )
            annotated = draw_overlay(
                frame,
                qnn_geometry=qnn_geometry,
                classical_geometry=classical_geometry,
                qnn_tracker=qnn_tracker,
                classical_tracker=classical_tracker,
                frame_index=frame_index,
                qnn_detections=latest_qnn_detections,
                classical_detections=latest_classical_detections,
                counts=fusion.counts,
                qnn_fps=qnn_fps,
                classical_fps=classical_fps,
                display_fps=display_fps,
                pending_qnn=fusion.pending_qnn,
                pending_classical=fusion.pending_classical,
                latest_fusion=latest_fusion,
            )
            writer.write(annotated)
            if args.show:
                cv2.imshow("Arty S7 QNN + downstream image processing", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            displayed += 1
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / source_fps,
                    "accepted_cell": fusion.counts.get("cell", 0),
                    "accepted_droplet": fusion.counts.get("droplet", 0),
                    "raw_qnn_cell": raw_qnn_crossings.get("cell", 0),
                    "raw_qnn_droplet": raw_qnn_crossings.get("droplet", 0),
                    "raw_classical_cell": raw_classical_crossings.get("cell", 0),
                    "raw_classical_droplet": raw_classical_crossings.get(
                        "droplet", 0
                    ),
                    "pending_qnn": fusion.pending_qnn,
                    "pending_classical": fusion.pending_classical,
                    "display_fps": display_fps,
                    "qnn_update_fps": qnn_fps,
                    "classical_fps": classical_fps,
                }
            )
            if not args.no_pace and not isinstance(source, int):
                target = display_started + displayed / source_fps
                delay = target - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
            if displayed % max(1, round(source_fps)) == 0:
                print(
                    f"display={displayed} QNN={qnn_fps:.1f}FPS "
                    f"classical={classical_fps:.1f}FPS AND="
                    f"D{fusion.counts.get('droplet', 0)}/C{fusion.counts.get('cell', 0)}",
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
        raise RuntimeError("FPGA QNN worker failed") from worker_error
    with result_lock:
        remaining = list(qnn_results[qnn_result_index:])
        all_results = list(qnn_results)
    current_frame = start_frame + max(0, displayed - 1)
    for result in remaining:
        process_qnn(result, current_frame)

    base.write_csv(
        output_dir / "qnn_fpga_detections.csv",
        qnn_detection_rows,
        ["frame_index", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
    )
    base.write_csv(
        output_dir / "classical_detections.csv",
        classical_detection_rows,
        [
            "frame_index",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "droplet_candidates",
            "cell_candidates",
        ],
    )
    base.write_csv(
        output_dir / "branch_crossing_events.csv",
        branch_event_rows,
        [
            "branch",
            "region_name",
            "track_id",
            "class_id",
            "class_name",
            "frame_index",
            "center_x",
            "center_y",
            "confidence",
            "hits",
            "velocity_x",
            "velocity_y",
            "cumulative_count",
        ],
    )
    base.write_csv(
        output_dir / "fusion_events.csv",
        fusion_rows,
        list(asdict(HybridCountEvent(0, 0, "", 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)).keys()),
    )
    base.write_csv(
        output_dir / "frame_summary.csv",
        frame_rows,
        [
            "frame_index",
            "time_seconds",
            "accepted_cell",
            "accepted_droplet",
            "raw_qnn_cell",
            "raw_qnn_droplet",
            "raw_classical_cell",
            "raw_classical_droplet",
            "pending_qnn",
            "pending_classical",
            "display_fps",
            "qnn_update_fps",
            "classical_fps",
        ],
    )

    total_elapsed = max(time.perf_counter() - display_started, 1e-9)
    qnn_times = [item.pair_round_trip_seconds for item in all_results]
    report = {
        "architecture": "ROI-A FPGA QNN + ROI-B independent classical + temporal AND",
        "source": str(args.source),
        "source_fps": source_fps,
        "displayed_frames": displayed,
        "display_fps": displayed / total_elapsed,
        "qnn_fpga_results": len(all_results),
        "qnn_update_fps": 1.0 / float(np.mean(qnn_times)) if qnn_times else 0.0,
        "classical_processing_fps": (
            1.0 / float(np.mean(classical_times)) if classical_times else 0.0
        ),
        "jobs_submitted": submitted,
        "stale_jobs_dropped": dropped_jobs,
        "port": port_name,
        "baud": args.baud,
        "rois": {
            "qnn_upstream": qnn_geometry,
            "classical_downstream": classical_geometry,
            "qnn_line_x": qnn_tracker.line_x,
            "classical_line_x": classical_tracker.line_x,
        },
        "raw_branch_crossings": {
            "qnn_fpga": raw_qnn_crossings,
            "classical_downstream": raw_classical_crossings,
        },
        "accepted_consensus_counts": fusion.counts,
        "disagreement_or_expired": {
            "qnn": fusion.rejected_qnn,
            "classical": fusion.rejected_classical,
            "pending_qnn": fusion.pending_qnn,
            "pending_classical": fusion.pending_classical,
        },
        "fusion_events": len(fusion_rows),
        "output_video": str(output_video),
        "truth_boundary": {
            "fpga": "ROI A QNN, requantization, radial guard v3, threshold and sparse UART",
            "host_reference": "ROI B Hough/radial image processing, temporal AND, count, logging and rendering",
            "next_rtl_step": "Replace Hough reference with calibrated line/DoG detector and move ROI-B validator plus fusion FSM into RTL",
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"FPGA_QNN_CLASSICAL_FUSION_PASS: {output_video}")


if __name__ == "__main__":
    main()
