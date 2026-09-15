#!/usr/bin/env python3
"""Run Zybo dual-ROI QNN synchronously using the proven Arty video layout."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import serial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import decode_output_tensor, load_manifest, pack_input_axis, prepare_image
from scripts.dual_roi_temporal_counter import (
    ClassAwareLineTracker,
    CrossRoiAssociator,
    DetectionSample,
)
from scripts.run_video_zybo_dual_roi_realtime import (
    map_detections,
    scaled_roi,
    sparse_transaction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default="COM13")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "exports/15micro_qnn_w4a6_96_roi120_v1/fpga_manifest_raw_core.json",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=ROOT / "configs/15micro_dual_qnn_fpga.json",
    )
    return parser.parse_args()


def draw_tracked_roi(
    frame,
    tracker,
    name,
    marker,
    color,
    observations,
    object_ids,
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
        name,
        (x1, max(22, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    object_colors = {"cell": (30, 60, 245), "droplet": (245, 90, 20)}
    for item in observations:
        bx1, by1, bx2, by2 = (int(round(value)) for value in item.box)
        bx1 = max(x1, min(x2 - 1, bx1))
        bx2 = max(x1 + 1, min(x2, bx2))
        by1 = max(y1, min(y2 - 1, by1))
        by2 = max(y1 + 1, min(y2, by2))
        detection_color = object_colors.get(item.class_name, color)
        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            detection_color,
            2 if item.confirmed else 1,
            cv2.LINE_AA,
        )
        object_id = object_ids.get(item.track_id)
        if object_id is None:
            prefix = "C" if item.class_name == "cell" else "D"
            object_id = f"{marker}-{prefix}{item.track_id}"
        suffix = "~" if item.predicted else ""
        cv2.putText(
            frame,
            f"{object_id}{suffix} {item.confidence:.2f}",
            (bx1, max(18, by1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            detection_color,
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    runtime_config = json.loads(args.roi_config.read_text(encoding="utf-8"))
    config = runtime_config["rois"]
    tracking = runtime_config["tracking"]
    verification = runtime_config["cross_roi_verification"]
    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    rois = (
        scaled_roi(width, height, config["left"]),
        scaled_roi(width, height, config["right"]),
    )
    line_fractions = (
        float(config["left"]["line_fraction"]),
        float(config["right"]["line_fraction"]),
    )
    trackers = tuple(
        ClassAwareLineTracker(
            region_name=name,
            roi_geometry=roi,
            line_fraction=line_fraction,
            direction=str(tracking["direction"]),
            minimum_hits=tracking["minimum_hits"],
            max_misses=tracking["max_misses"],
            max_center_distance=tracking["max_center_distance"],
            count_hysteresis=float(tracking["count_hysteresis"]),
            reverse_tolerance=float(tracking["reverse_tolerance"]),
            edge_margin=float(tracking["edge_margin"]),
        )
        for name, roi, line_fraction in zip(
            ("candidate", "confirm"), rois, line_fractions
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
    first = max(0, int(round(args.start_sec * fps)))
    last = min(total_frames, first + int(round(args.duration_sec * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)

    args.output.mkdir(parents=True, exist_ok=True)
    output_video = args.output / "zybo_dual_roi_exact_arty_style.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    qnn_us_values: list[int] = []
    uart_values: list[float] = []
    record_values: list[int] = []
    confirmed_counts = {name: 0 for name in class_names}
    candidate_serials = {name: 0 for name in class_names}
    roi_1_object_ids: dict[int, str] = {}
    roi_2_object_ids: dict[int, str] = {}
    confirmed_events: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    processed = 0
    started = time.perf_counter()
    preview_saved = False
    try:
        with serial.Serial(
            args.port, args.baud, timeout=3, write_timeout=3
        ) as port:
            port.dtr = False
            port.rts = False
            port.reset_input_buffer()
            for frame_index in range(first, last):
                ok, frame = capture.read()
                if not ok:
                    break
                all_detections = []
                frame_uart_seconds = 0.0
                for roi in rois:
                    x1, y1, x2, y2 = roi
                    codes = prepare_image(
                        frame[y1:y2, x1:x2], manifest, array_is_bgr=True
                    )
                    payload = pack_input_axis(codes, manifest)
                    tensor, qnn_us, records, uart_seconds = sparse_transaction(
                        port, payload, manifest
                    )
                    decoded = decode_output_tensor(tensor, manifest)[0]
                    all_detections.append(
                        list(map_detections(decoded, class_names, roi))
                    )
                    qnn_us_values.append(qnn_us)
                    record_values.append(records)
                    frame_uart_seconds += uart_seconds
                uart_values.append(frame_uart_seconds)

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
                    for detections in all_detections
                )
                upstream_events = trackers[0].update(samples[0], frame_index)
                observations_1 = tuple(
                    trackers[0].observations(
                        frame_index,
                        maximum_prediction_frames=int(
                            tracking["maximum_prediction_frames"]
                        ),
                    )
                )
                for observation in observations_1:
                    if (
                        not observation.confirmed
                        or observation.track_id in roi_1_object_ids
                    ):
                        continue
                    candidate_serials[observation.class_name] += 1
                    prefix = "C" if observation.class_name == "cell" else "D"
                    roi_1_object_ids[observation.track_id] = (
                        f"{prefix}{candidate_serials[observation.class_name]:04d}"
                    )
                associator.add_upstream(upstream_events)
                downstream_events = trackers[1].update(samples[1], frame_index)
                frame_confirmations = []
                for downstream_event in downstream_events:
                    association = associator.associate(downstream_event)
                    if not association.verified:
                        continue
                    upstream_event = association.upstream_event
                    if upstream_event is None:
                        continue
                    object_id = roi_1_object_ids.get(upstream_event.track_id)
                    if object_id is None:
                        candidate_serials[upstream_event.class_name] += 1
                        prefix = (
                            "C" if upstream_event.class_name == "cell" else "D"
                        )
                        object_id = (
                            f"{prefix}{candidate_serials[upstream_event.class_name]:04d}"
                        )
                        roi_1_object_ids[upstream_event.track_id] = object_id
                    roi_2_object_ids[downstream_event.track_id] = object_id
                    confirmed_counts[downstream_event.class_name] += 1
                    event_row = {
                        "event_id": len(confirmed_events) + 1,
                        "object_id": object_id,
                        "class_id": downstream_event.class_id,
                        "class_name": downstream_event.class_name,
                        "roi_1_frame": upstream_event.frame_index,
                        "roi_2_frame": downstream_event.frame_index,
                        "roi_1_time_s": upstream_event.frame_index / fps,
                        "roi_2_time_s": downstream_event.frame_index / fps,
                        "delay_frames": association.delay_frames,
                        "cross_axis_distance_px": association.cross_axis_distance,
                        "roi_1_track_id": upstream_event.track_id,
                        "roi_2_track_id": downstream_event.track_id,
                        "roi_1_confidence": upstream_event.confidence,
                        "roi_2_confidence": downstream_event.confidence,
                        "cumulative_class_count": confirmed_counts[
                            downstream_event.class_name
                        ],
                    }
                    confirmed_events.append(event_row)
                    frame_confirmations.append(downstream_event.class_name)
                    (args.output / "live_counts.json").write_text(
                        json.dumps(
                            {
                                "cell": confirmed_counts.get("cell", 0),
                                "droplet": confirmed_counts.get("droplet", 0),
                                "last_event": event_row,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                observations_2 = tuple(
                    trackers[1].observations(
                        frame_index,
                        maximum_prediction_frames=int(
                            tracking["maximum_prediction_frames"]
                        ),
                    )
                )

                output = frame.copy()
                draw_tracked_roi(
                    output,
                    trackers[0],
                    "ROI 1: QNN candidate",
                    "1",
                    (40, 210, 40),
                    observations_1,
                    roi_1_object_ids,
                )
                draw_tracked_roi(
                    output,
                    trackers[1],
                    "ROI 2: QNN confirm",
                    "2",
                    (235, 220, 20),
                    observations_2,
                    roi_2_object_ids,
                )
                cells = sum(
                    item.class_name == "cell"
                    for detections in all_detections
                    for item in detections
                )
                droplets = sum(
                    item.class_name == "droplet"
                    for detections in all_detections
                    for item in detections
                )
                hardware_fps = 1.0 / max(frame_uart_seconds, 1e-9)
                cv2.rectangle(output, (0, 0), (min(width, 940), 56), (10, 13, 17), -1)
                cv2.putText(
                    output,
                    f"Zybo Z7-10 | ROI 1 candidate -> ROI 2 confirm | {hardware_fps:.1f} FPGA FPS",
                    (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    f"confirmed once A->B | cell={confirmed_counts.get('cell', 0)} droplet={confirmed_counts.get('droplet', 0)} | frame {frame_index + 1}",
                    (12, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (80, 230, 80),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(output)
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_s": frame_index / fps,
                        "roi_1_detections": len(all_detections[0]),
                        "roi_2_detections": len(all_detections[1]),
                        "confirmed_this_frame": ";".join(frame_confirmations),
                        "confirmed_cell_total": confirmed_counts.get("cell", 0),
                        "confirmed_droplet_total": confirmed_counts.get(
                            "droplet", 0
                        ),
                        "pending_roi_1_events": len(associator.pending),
                        "dual_roi_uart_ms": frame_uart_seconds * 1000.0,
                    }
                )
                if not preview_saved and processed >= 10:
                    cv2.imwrite(str(args.output / "preview.jpg"), output)
                    preview_saved = True
                processed += 1
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started
    event_fields = [
        "event_id",
        "object_id",
        "class_id",
        "class_name",
        "roi_1_frame",
        "roi_2_frame",
        "roi_1_time_s",
        "roi_2_time_s",
        "delay_frames",
        "cross_axis_distance_px",
        "roi_1_track_id",
        "roi_2_track_id",
        "roi_1_confidence",
        "roi_2_confidence",
        "cumulative_class_count",
    ]
    with (args.output / "confirmed_events.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=event_fields)
        writer_csv.writeheader()
        writer_csv.writerows(confirmed_events)
    if frame_rows:
        with (args.output / "frame_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
            writer_csv.writeheader()
            writer_csv.writerows(frame_rows)
    (args.output / "live_counts.json").write_text(
        json.dumps(
            {
                "cell": confirmed_counts.get("cell", 0),
                "droplet": confirmed_counts.get("droplet", 0),
                "confirmed_events": len(confirmed_events),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "hardware": "Digilent Zybo Z7-10 XC7Z010-1CLG400C",
        "mode": "synchronous_per_source_frame_matching_Arty_reference",
        "source": str(args.video.resolve()),
        "source_fps": fps,
        "processed_frames": processed,
        "playback_fps": fps,
        "generation_wall_fps": processed / max(elapsed, 1e-9),
        "mean_qnn_ms": statistics.fmean(qnn_us_values) / 1000.0,
        "mean_dual_roi_uart_ms": statistics.fmean(uart_values) * 1000.0,
        "mean_sparse_records_per_roi": statistics.fmean(record_values),
        "counting_rule": "ROI 1 candidate plus matching ROI 2 confirmation increments once",
        "confirmed_counts": confirmed_counts,
        "candidate_ids_issued": candidate_serials,
        "confirmed_events": len(confirmed_events),
        "edge_track_policy": (
            "Confirmed tracks retain a predicted box for up to 10 missing frames; "
            "a predicted line crossing may trigger the ROI event once"
        ),
        "rois": {"candidate": list(rois[0]), "confirm": list(rois[1])},
        "frame_alignment": "Each displayed frame uses QNN results from that exact frame",
        "inference_location": "Both ROI QNN inferences execute in FPGA programmable logic",
        "output_video": str(output_video.resolve()),
        "artifacts": {
            "confirmed_events": str(
                (args.output / "confirmed_events.csv").resolve()
            ),
            "frame_summary": str((args.output / "frame_summary.csv").resolve()),
            "live_counts": str((args.output / "live_counts.json").resolve()),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"ZYBO_DUAL_ROI_EXACT_PASS: {output_video.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
