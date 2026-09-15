#!/usr/bin/env python3
"""Stream grayscale QNN and RGB332 classical ROIs to the Arty S7-25."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
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
from scripts.send_frame_finn_uart_sparse import transact_sparse
from scripts.send_frame_finn_uart import detect_serial_port
from scripts.dual_roi_rtl_classical import pack_rgb332

DEFAULT_MANIFEST = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "23_fpga_qnn96_dual_guard_v3"
    / "config"
    / "fpga_manifest_sparse_uart.json"
)
DEFAULT_CONFIG = ROOT / "configs" / "15micro_hybrid_v6_rgb332_fpga.json"
DEFAULT_SOURCE = ROOT / "data" / "raw" / "09_07_2026" / "09_07_2026" / "3.4.mp4"
HYBRID_MAGIC = b"HY\xff"
CLASSICAL_BOX_MAGIC = b"CB"
WORKING_SIZE = 96


def prepare_classical_rgb332(frame_bgr: np.ndarray) -> np.ndarray:
    """Return an NHWC byte tensor whose pixels use the RRRGGGBB contract."""

    return pack_rgb332(frame_bgr, WORKING_SIZE)[..., None]


class AsyncVideoWriter:
    """Overlap MP4 encoding with the next FPGA transaction."""

    _STOP = object()

    def __init__(
        self, path: Path, fourcc: int, fps: float, size: tuple[int, int]
    ) -> None:
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create video: {path}")
        self._queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=12)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is self._STOP:
                    break
                self._writer.write(frame)
        except BaseException as error:
            self._error = error
        finally:
            self._writer.release()

    def _enqueue(self, item: np.ndarray | object) -> None:
        while True:
            if self._error is not None:
                raise RuntimeError("Video encoder failed") from self._error
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def write(self, frame: np.ndarray) -> None:
        self._enqueue(frame)

    def close(self) -> None:
        self._enqueue(self._STOP)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Video encoder failed") from self._error


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
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "15micro_hybrid_v4_fpga_video",
    )
    parser.add_argument(
        "--draw-qnn-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help="Scale only the rendered video; FPGA input stays 96x96 per ROI.",
    )
    parser.add_argument(
        "--fast-video",
        action="store_true",
        help="Render at 50%% size to reduce encoding cost.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-pace", action="store_true")
    return parser.parse_args()


def split_hybrid_response(
    payload: bytes,
) -> tuple[bytes, dict[str, int], list[dict[str, Any]]]:
    if len(payload) % 8:
        raise ValueError("FPGA response is not record-aligned")
    status: dict[str, int] | None = None
    detector_records = bytearray()
    classical_boxes: list[dict[str, Any]] = []
    for offset in range(0, len(payload), 8):
        record = payload[offset : offset + 8]
        if record[:3] == HYBRID_MAGIC:
            if status is not None:
                raise ValueError("FPGA returned more than one HY status record")
            status = {
                "flags": record[3],
                "droplet_count": int.from_bytes(record[4:6], "little"),
                "cell_count": int.from_bytes(record[6:8], "little"),
            }
        elif record[:2] == CLASSICAL_BOX_MAGIC:
            class_id = record[2]
            if class_id > 1:
                raise ValueError(f"Unknown classical box class id: {class_id}")
            x1, y1, x2, y2 = record[4:8]
            if x1 > x2 or y1 > y2 or x2 >= WORKING_SIZE or y2 >= WORKING_SIZE:
                raise ValueError(f"Invalid classical box record: {record.hex()}")
            classical_boxes.append(
                {
                    "class_id": class_id,
                    "class_name": "cell" if class_id == 0 else "droplet",
                    "score_code": record[3],
                    "working_box": (x1, y1, x2, y2),
                }
            )
        else:
            detector_records.extend(record)
    if status is None:
        raise ValueError("FPGA response is missing the HY status record")
    return bytes(detector_records), status, classical_boxes


def map_classical_boxes(
    boxes: list[dict[str, Any]],
    roi: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    rx1, ry1, rx2, ry2 = roi
    scale_x = (rx2 - rx1) / WORKING_SIZE
    scale_y = (ry2 - ry1) / WORKING_SIZE
    mapped: list[dict[str, Any]] = []
    for item in boxes:
        x1, y1, x2, y2 = item["working_box"]
        mapped.append(
            {
                **item,
                "box": (
                    round(rx1 + x1 * scale_x),
                    round(ry1 + y1 * scale_y),
                    round(rx1 + (x2 + 1) * scale_x) - 1,
                    round(ry1 + (y2 + 1) * scale_y) - 1,
                ),
            }
        )
    return mapped


def flag(status: dict[str, int], bit: int) -> int:
    return (status["flags"] >> bit) & 1


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_overlay(
    frame: np.ndarray,
    qnn_roi: tuple[int, int, int, int],
    classical_roi: tuple[int, int, int, int],
    qnn_gate_fraction: float,
    classical_gate_fraction: float,
    status: dict[str, int],
    system_fps: float,
    detections: list[Any],
    classical_boxes: list[dict[str, Any]],
    render_scale: float,
    classical_display_droplet: bool,
    classical_display_cell: bool,
) -> np.ndarray:
    output = frame.copy()
    ui_scale = max(0.6, render_scale)
    thin = max(1, round(2 * ui_scale))
    thick = max(2, round(5 * ui_scale))
    qnn_roi = tuple(round(value * render_scale) for value in qnn_roi)
    classical_roi = tuple(round(value * render_scale) for value in classical_roi)
    qx1, qy1, qx2, qy2 = qnn_roi
    cx1, cy1, cx2, cy2 = classical_roi
    classical_color = (255, 180, 30)
    classical_label = ""
    if classical_display_cell:
        classical_color = (30, 60, 245)
        classical_label = "ROI B FPGA CELL"
    elif classical_display_droplet:
        classical_color = (255, 90, 20)
        classical_label = "ROI B FPGA DROPLET"
    cv2.rectangle(output, (qx1, qy1), (qx2, qy2), (40, 210, 40), thin)
    cv2.rectangle(
        output,
        (cx1, cy1),
        (cx2, cy2),
        classical_color,
        thick if classical_label else thin,
    )
    q_line = round(qx1 + (qx2 - qx1) * qnn_gate_fraction)
    c_line = round(cx1 + (cx2 - cx1) * classical_gate_fraction)
    cv2.line(output, (q_line, qy1), (q_line, qy2), (40, 210, 40), 1)
    cv2.line(output, (c_line, cy1), (c_line, cy2), classical_color, 1)
    cv2.putText(
        output,
        "ROI A: FPGA QNN",
        (qx1, max(round(24 * ui_scale), qy1 - round(8 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * ui_scale,
        (40, 210, 40),
        thin,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "ROI B: FPGA integer image processing",
        (cx1, max(round(24 * ui_scale), cy1 - round(8 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * ui_scale,
        classical_color,
        thin,
        cv2.LINE_AA,
    )
    if classical_label:
        label_y = min(
            output.shape[0] - round(8 * ui_scale),
            cy2 + round(24 * ui_scale),
        )
        (label_width, label_height), _ = cv2.getTextSize(
            classical_label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 * ui_scale,
            thin,
        )
        cv2.rectangle(
            output,
            (cx1, label_y - label_height - round(7 * ui_scale)),
            (
                min(
                    output.shape[1] - 1,
                    cx1 + label_width + round(8 * ui_scale),
                ),
                label_y + round(4 * ui_scale),
            ),
            (10, 13, 17),
            -1,
        )
        cv2.putText(
            output,
            classical_label,
            (cx1 + round(4 * ui_scale), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 * ui_scale,
            classical_color,
            thin,
            cv2.LINE_AA,
        )
    colors = {"cell": (30, 60, 245), "droplet": (245, 90, 20)}
    for item in detections:
        x1, y1, x2, y2 = (
            round(value * render_scale) for value in item.box
        )
        color = colors.get(item.class_name, (220, 220, 220))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            output, f"{item.class_name} {item.confidence:.2f}",
            (x1, max(18, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            color, 1, cv2.LINE_AA,
        )
    for item in classical_boxes:
        x1, y1, x2, y2 = (
            round(value * render_scale) for value in item["box"]
        )
        color = colors[item["class_name"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, max(2, thin))
        cv2.putText(
            output,
            f"B {item['class_name']} score={item['score_code']}",
            (
                x1,
                min(
                    output.shape[0] - round(5 * ui_scale),
                    y2 + round(18 * ui_scale),
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43 * ui_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    accepted = []
    if flag(status, 5):
        accepted.append("CELL")
    if flag(status, 4):
        accepted.append("DROPLET")
    accepted_text = "+".join(accepted) if accepted else "none"
    panel_lines = (
        f"Arty S7-25 hybrid v6 RGB332 | system {system_fps:.1f} FPS",
        f"FPGA consensus count | cell={status['cell_count']} droplet={status['droplet_count']}",
        f"this frame accepted: {accepted_text}",
        (
            f"events A(qnn): C{flag(status, 1)} D{flag(status, 0)} | "
            f"B(classical): C{flag(status, 3)} D{flag(status, 2)}"
        ),
    )
    panel_height = round((26 + 24 * len(panel_lines)) * ui_scale)
    cv2.rectangle(output, (0, 0),
                  (min(output.shape[1], round(760 * ui_scale)), panel_height),
                  (10, 13, 17), -1)
    for index, text in enumerate(panel_lines):
        cv2.putText(
            output,
            text,
            (round(12 * ui_scale), round((27 + 24 * index) * ui_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 * ui_scale,
            (235, 240, 245),
            1,
            cv2.LINE_AA,
        )
    return output


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.source.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_scale = 0.5 if args.fast_video else args.output_scale
    if not 0.0 < output_scale <= 1.0:
        raise ValueError("--output-scale must be greater than 0 and at most 1")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    qnn_roi = scaled_roi(width, height, config["rois"]["qnn"])
    classical_roi = scaled_roi(width, height, config["rois"]["classical"])
    first_frame = max(0, round(args.start_sec * source_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    maximum_frames = (
        round(args.duration_sec * source_fps) if args.duration_sec > 0 else None
    )
    output_width = max(2, round(width * output_scale))
    output_height = max(2, round(height * output_scale))
    output_width -= output_width % 2
    output_height -= output_height % 2
    video_path = output_dir / "hybrid_v6_rgb332_fpga_result.mp4"
    writer = AsyncVideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (output_width, output_height),
    )

    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    transform = InputTransform(96, 96, 96, 96, 0, 0)
    port_name = args.port or detect_serial_port()
    import serial

    rows: list[dict[str, Any]] = []
    processed = 0
    classical_droplet_display_frames = 0
    classical_cell_display_frames = 0
    started = time.perf_counter()
    next_deadline = started
    with serial.Serial(
        port_name, args.baud, timeout=0.01, write_timeout=max(3.0, args.timeout)
    ) as port:
        try:
            port.set_buffer_size(rx_size=1 << 20, tx_size=1 << 20)
        except (AttributeError, OSError):
            pass
        port.reset_input_buffer()
        port.reset_output_buffer()
        while maximum_frames is None or processed < maximum_frames:
            ok, frame = capture.read()
            if not ok:
                break
            qx1, qy1, qx2, qy2 = qnn_roi
            cx1, cy1, cx2, cy2 = classical_roi
            qnn_codes = prepare_image(
                frame[qy1:qy2, qx1:qx2], manifest, array_is_bgr=True
            )
            classical_codes = prepare_classical_rgb332(
                frame[cy1:cy2, cx1:cx2]
            )
            payload = pack_input_axis(qnn_codes, manifest) + pack_input_axis(
                classical_codes, manifest
            )
            if len(payload) != 18432:
                raise ValueError(f"Expected 18432 dual-ROI bytes, got {len(payload)}")
            response, cycles, elapsed = transact_sparse(
                port,
                payload,
                frame_id=(processed + 1) & 0xFFFF,
                timeout=args.timeout,
            )
            detector_payload, status, classical_working_boxes = split_hybrid_response(
                response
            )
            classical_boxes = map_classical_boxes(
                classical_working_boxes, classical_roi
            )
            classical_droplet_display_frames = max(
                0, classical_droplet_display_frames - 1
            )
            classical_cell_display_frames = max(0, classical_cell_display_frames - 1)
            if flag(status, 2):
                classical_droplet_display_frames = 5
            if flag(status, 3):
                classical_cell_display_frames = 5
            detections: list[Any] = []
            if args.draw_qnn_boxes:
                tensor = unpack_sparse_detection_axis(detector_payload, manifest)
                decoded = decode_output_tensor(tensor, manifest)[0]
                detections = map_roi_detections(
                    decoded, class_names, qnn_roi, transform
                )
            system_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            render_frame = frame
            if (output_width, output_height) != (width, height):
                render_frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            annotated = draw_overlay(
                render_frame,
                qnn_roi,
                classical_roi,
                float(config["rois"]["qnn"]["gate_fraction"]),
                float(config["rois"]["classical"]["gate_fraction"]),
                status,
                system_fps,
                detections,
                classical_boxes,
                output_scale,
                classical_droplet_display_frames > 0,
                classical_cell_display_frames > 0,
            )
            writer.write(annotated)
            classical_droplet_boxes = sum(
                item["class_id"] == 1 for item in classical_boxes
            )
            classical_cell_boxes = sum(
                item["class_id"] == 0 for item in classical_boxes
            )
            rows.append(
                {
                    "source_frame": first_frame + processed,
                    "status_flags": status["flags"],
                    "qnn_droplet_event": flag(status, 0),
                    "qnn_cell_event": flag(status, 1),
                    "classical_droplet_event": flag(status, 2),
                    "classical_cell_event": flag(status, 3),
                    "accepted_droplet": flag(status, 4),
                    "accepted_cell": flag(status, 5),
                    "droplet_count": status["droplet_count"],
                    "cell_count": status["cell_count"],
                    "qnn_sparse_records": len(detector_payload) // 8,
                    "classical_box_count": len(classical_boxes),
                    "classical_droplet_box_count": classical_droplet_boxes,
                    "classical_cell_box_count": classical_cell_boxes,
                    "classical_boxes_json": json.dumps(
                        classical_boxes, separators=(",", ":")
                    ),
                    "fpga_cycles": cycles,
                    "fpga_transaction_ms": cycles / args.clock_hz * 1000.0,
                    "uart_round_trip_ms": elapsed * 1000.0,
                    "system_fps": system_fps,
                }
            )
            if args.show:
                cv2.imshow("15micro hybrid v6 RGB332 FPGA", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            processed += 1
            if not args.no_pace:
                next_deadline += 1.0 / source_fps
                remaining = next_deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

    total_elapsed = time.perf_counter() - started
    capture.release()
    writer.close()
    if args.show:
        cv2.destroyAllWindows()
    write_rows(output_dir / "frame_summary.csv", rows)
    report = {
        "architecture": "ROI A QNN + ROI B integer classical + temporal AND/count on FPGA",
        "source": str(source),
        "source_fps": source_fps,
        "processed_frames": processed,
        "wall_fps": processed / total_elapsed if total_elapsed > 0 else 0.0,
        "output_scale": output_scale,
        "output_size": [output_width, output_height],
        "async_video_writer": True,
        "mean_system_fps": float(np.mean([row["system_fps"] for row in rows]))
        if rows else 0.0,
        "mean_fpga_transaction_ms": float(
            np.mean([row["fpga_transaction_ms"] for row in rows])
        ) if rows else 0.0,
        "final_counts": {
            "cell": rows[-1]["cell_count"] if rows else 0,
            "droplet": rows[-1]["droplet_count"] if rows else 0,
        },
        "accepted_events": {
            "cell": sum(row["accepted_cell"] for row in rows),
            "droplet": sum(row["accepted_droplet"] for row in rows),
        },
        "classical_bbox_frames": {
            "any": sum(row["classical_box_count"] > 0 for row in rows),
            "cell": sum(row["classical_cell_box_count"] > 0 for row in rows),
            "droplet": sum(
                row["classical_droplet_box_count"] > 0 for row in rows
            ),
        },
        "rois": {"qnn": qnn_roi, "classical": classical_roi},
        "port": port_name,
        "baud": args.baud,
        "video": str(video_path),
        "truth_boundary": config["truth_boundary"],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
