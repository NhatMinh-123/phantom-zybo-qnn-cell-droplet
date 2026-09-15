#!/usr/bin/env python3
"""Run two adjacent 15-micron ROIs through one Arty S7-25 QNN core."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
from scripts.run_video_finn_uart import InputTransform, map_roi_detections, scaled_roi
from scripts.send_frame_finn_uart import detect_serial_port
from scripts.send_frame_finn_uart_sparse import transact_sparse

DEFAULT_MANIFEST = (
    ROOT / "final_results" / "15micro_pipeline_v1" / "23_fpga_qnn96_dual_guard_v3"
    / "config" / "fpga_manifest_sparse_uart.json"
)
DEFAULT_CONFIG = ROOT / "configs" / "15micro_dual_qnn_fpga.json"
DEFAULT_SOURCE = ROOT / "data" / "raw" / "09_07_2026" / "09_07_2026" / "3.4.mp4"
HYBRID_MAGIC = b"HY\xff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=12_000_000)
    parser.add_argument("--clock-hz", type=int, default=96_000_000)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "15micro_dual_qnn_fpga_video")
    parser.add_argument("--fast-video", action="store_true")
    parser.add_argument("--no-pace", action="store_true")
    return parser.parse_args()


def draw_roi_detections(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    name: str,
    marker: str,
    color: tuple[int, int, int],
    detections: list[Any],
) -> None:
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, name, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)
    object_colors = {"cell": (30, 60, 245), "droplet": (245, 90, 20)}
    for item in detections:
        bx1, by1, bx2, by2 = (round(value) for value in item.box)
        detection_color = object_colors.get(item.class_name, color)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), detection_color, 2)
        cv2.putText(frame, f"{marker} {item.class_name} {item.confidence:.2f}",
                    (bx1, max(18, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    detection_color, 1, cv2.LINE_AA)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def split_consensus_response(payload: bytes) -> tuple[bytes, dict[str, int]]:
    """Remove the FPGA status record while retaining sparse QNN records."""

    detector = bytearray()
    status: dict[str, int] | None = None
    if len(payload) % 8:
        raise ValueError("FPGA response is not record-aligned")
    for offset in range(0, len(payload), 8):
        record = payload[offset : offset + 8]
        if record[:3] == HYBRID_MAGIC:
            status = {
                "flags": record[3],
                "droplet_count": int.from_bytes(record[4:6], "little"),
                "cell_count": int.from_bytes(record[6:8], "little"),
            }
        else:
            detector.extend(record)
    if status is None:
        raise ValueError("FPGA consensus status record is missing")
    return bytes(detector), status


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.source.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    left_roi = scaled_roi(width, height, config["rois"]["left"])
    right_roi = scaled_roi(width, height, config["rois"]["right"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(args.start_sec * source_fps)))
    first_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    max_frames = round(args.duration_sec * source_fps) if args.duration_sec > 0 else None
    output_scale = 0.5 if args.fast_video else 1.0
    out_size = (max(2, int(width * output_scale) // 2 * 2),
                max(2, int(height * output_scale) // 2 * 2))
    video_path = output_dir / "dual_qnn_fpga_result.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             source_fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {video_path}")

    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    transform = InputTransform(96, 96, 96, 96, 0, 0)
    port_name = args.port or detect_serial_port()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    processed = 0
    import serial
    with serial.Serial(port_name, args.baud, timeout=0.01,
                       write_timeout=max(3.0, args.timeout)) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        while max_frames is None or processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            all_detections: list[list[Any]] = []
            fpga_statuses: list[dict[str, int]] = []
            transaction_seconds = 0.0
            transaction_cycles = 0
            for roi_index, roi in enumerate((left_roi, right_roi)):
                x1, y1, x2, y2 = roi
                codes = prepare_image(frame[y1:y2, x1:x2], manifest, array_is_bgr=True)
                payload = pack_input_axis(codes, manifest)
                if len(payload) != 9216:
                    raise ValueError(f"Expected 9216 bytes, got {len(payload)}")
                response, cycles, elapsed = transact_sparse(
                    port, payload, frame_id=((processed * 2 + roi_index + 1) & 0xFFFF),
                    timeout=args.timeout,
                )
                response, fpga_status = split_consensus_response(response)
                decoded = decode_output_tensor(
                    unpack_sparse_detection_axis(response, manifest), manifest
                )[0]
                all_detections.append(map_roi_detections(decoded, class_names, roi, transform))
                fpga_statuses.append(fpga_status)
                transaction_seconds += elapsed
                transaction_cycles += cycles
            output = frame.copy()
            draw_roi_detections(output, left_roi, "ROI 1: QNN candidate", "1", (40, 210, 40), all_detections[0])
            draw_roi_detections(output, right_roi, "ROI 2: QNN confirm", "2", (235, 220, 20), all_detections[1])
            cell_total = sum(item.class_name == "cell" for group in all_detections for item in group)
            droplet_total = sum(item.class_name == "droplet" for group in all_detections for item in group)
            consensus = fpga_statuses[1]
            accepted_cell = (consensus["flags"] >> 5) & 1
            accepted_droplet = (consensus["flags"] >> 4) & 1
            fps = 1.0 / transaction_seconds if transaction_seconds else 0.0
            cv2.rectangle(output, (0, 0), (min(width, 780), 56), (10, 13, 17), -1)
            cv2.putText(output, f"Arty S7-25 | ROI 1 candidate -> ROI 2 confirm | {fps:.1f} FPS",
                        (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(output, (
                            f"FPGA confirmed once | cell/particle={consensus['cell_count']} "
                            f"droplet={consensus['droplet_count']} | "
                            f"A->B match C{accepted_cell} D{accepted_droplet}"
                        ),
                        (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 230, 80), 1, cv2.LINE_AA)
            if output_scale != 1.0:
                output = cv2.resize(output, out_size, interpolation=cv2.INTER_AREA)
            writer.write(output)
            rows.append({
                "source_frame": first_frame + processed,
                "roi_1_candidate_detections": len(all_detections[0]),
                "roi_2_confirm_detections": len(all_detections[1]),
                "roi_a_detections": len(all_detections[0]),
                "roi_b_detections": len(all_detections[1]),
                "cell_detections": cell_total,
                "droplet_detections": droplet_total,
                "accepted_cell": accepted_cell,
                "accepted_droplet": accepted_droplet,
                "fpga_cell_count": consensus["cell_count"],
                "fpga_droplet_count": consensus["droplet_count"],
                "fpga_cycles_two_rois": transaction_cycles,
                "uart_round_trip_ms_two_rois": transaction_seconds * 1000.0,
                "system_fps_two_rois": fps,
            })
            processed += 1
            if not args.no_pace:
                time.sleep(max(0.0, 1.0 / source_fps - transaction_seconds))
    cap.release()
    writer.release()
    elapsed = time.perf_counter() - started
    write_csv(output_dir / "frame_summary.csv", rows)
    report = {
        "architecture": "ROI 1 creates pending candidates; ROI 2 confirms matching candidates; each A-to-B match increments exactly one FPGA counter",
        "source": str(source), "port": port_name, "processed_frames": processed,
        "wall_fps": processed / elapsed if elapsed else 0.0,
        "mean_system_fps_two_rois": float(np.mean([row["system_fps_two_rois"] for row in rows])) if rows else 0.0,
        "mean_uart_round_trip_ms_two_rois": float(np.mean([row["uart_round_trip_ms_two_rois"] for row in rows])) if rows else 0.0,
        "final_fpga_counts": {
            "cell_particle": rows[-1]["fpga_cell_count"] if rows else 0,
            "droplet": rows[-1]["fpga_droplet_count"] if rows else 0
        },
        "accepted_events": {
            "cell_particle": sum(row["accepted_cell"] for row in rows),
            "droplet": sum(row["accepted_droplet"] for row in rows)
        },
        "rois": {"candidate_roi_1": left_roi, "confirm_roi_2": right_roi}, "video": str(video_path),
        "classical_processing": "disabled",
        "truth_boundary": {
            "fpga": "QNN inference for both ROIs; ROI 1 candidate history; temporal A-to-B matching; one counter increment only after ROI 2 confirmation",
            "host": "video decode, ROI crop, UART transport, logging and rendering only"
        }
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
