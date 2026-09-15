#!/usr/bin/env python3
"""Render a full video while Zybo QNN asynchronously updates two ROIs."""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import decode_output_tensor, load_manifest, pack_input_axis, prepare_image


RECORD_BYTES = 13
OBJECT_CHANNELS = (0, 5, 10)
ROI_COLORS = ((30, 220, 30), (230, 210, 30))
CLASS_COLORS = {"cell": (35, 35, 235), "droplet": (235, 150, 20)}


@dataclass(frozen=True)
class GlobalDetection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class FPGAResult:
    frame_id: int
    detections: tuple[tuple[GlobalDetection, ...], tuple[GlobalDetection, ...]]
    qnn_us: tuple[int, int]
    uart_seconds: float
    records: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default="COM13")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "exports/15micro_qnn_w4a6_96_roi120_v1/fpga_manifest_raw_core.json",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=ROOT / "configs/15micro_qnn_dual_roi120_downstream_v1.json",
    )
    return parser.parse_args()


def read_exact(port: serial.Serial, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        chunk = port.read(count - len(data))
        if not chunk:
            raise TimeoutError(f"UART timeout after {len(data)}/{count} bytes")
        data.extend(chunk)
    return bytes(data)


def scaled_roi(
    width: int, height: int, config: dict[str, Any]
) -> tuple[int, int, int, int]:
    sx = width / int(config["reference_width"])
    sy = height / int(config["reference_height"])
    x1 = int(round(float(config["x"]) * sx))
    y1 = int(round(float(config["y"]) * sy))
    x2 = int(round((float(config["x"]) + float(config["width"])) * sx))
    y2 = int(round((float(config["y"]) + float(config["height"])) * sy))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ROI {(x1, y1, x2, y2)} is outside {width}x{height}")
    return x1, y1, x2, y2


def sparse_transaction(
    port: serial.Serial, payload: bytes, manifest: dict[str, Any]
) -> tuple[np.ndarray, int, int, float]:
    started = time.perf_counter()
    port.write(b"\xA5\x5A" + payload)
    port.flush()
    header = read_exact(port, 2)
    if header == b"ER":
        raise RuntimeError("Zybo DMA application reported an error")
    if header != b"\x5A\xA6":
        raise RuntimeError(f"Unexpected sparse response header: {header.hex()}")
    record_count = int.from_bytes(read_exact(port, 2), "little")
    qnn_us = int.from_bytes(read_exact(port, 4), "little")
    records = read_exact(port, record_count * RECORD_BYTES)
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
        if grid_index >= 576 or slot >= 3:
            raise ValueError(f"Invalid sparse record grid={grid_index} slot={slot}")
        values = np.frombuffer(record[3:13], dtype="<i2")
        gy, gx = divmod(grid_index, 24)
        tensor[gy, gx, slot * 5 : slot * 5 + 5] = values
    return tensor, qnn_us, record_count, time.perf_counter() - started


def map_detections(
    decoded: list[Any], class_names: list[str], roi: tuple[int, int, int, int]
) -> tuple[GlobalDetection, ...]:
    x1, y1, x2, y2 = roi
    width, height = x2 - x1, y2 - y1
    output: list[GlobalDetection] = []
    for item in decoded:
        bx1, by1, bx2, by2 = item.box
        output.append(
            GlobalDetection(
                class_id=item.class_id,
                class_name=class_names[item.class_id],
                confidence=float(item.confidence),
                box=(
                    int(round(x1 + bx1 * width)),
                    int(round(y1 + by1 * height)),
                    int(round(x1 + bx2 * width)),
                    int(round(y1 + by2 * height)),
                ),
            )
        )
    return tuple(output)


class FPGAWorker(threading.Thread):
    def __init__(
        self,
        port_name: str,
        baud: int,
        manifest: dict[str, Any],
        class_names: list[str],
        rois: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    ) -> None:
        super().__init__(daemon=True)
        self.port_name = port_name
        self.baud = baud
        self.manifest = manifest
        self.class_names = class_names
        self.rois = rois
        self.jobs: queue.Queue[tuple[int, tuple[np.ndarray, np.ndarray]] | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[FPGAResult] = queue.Queue()
        self.error: BaseException | None = None
        self.dropped = 0

    def submit(self, frame_id: int, crops: tuple[np.ndarray, np.ndarray]) -> None:
        try:
            self.jobs.put_nowait((frame_id, crops))
        except queue.Full:
            try:
                self.jobs.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.jobs.put_nowait((frame_id, crops))

    def stop(self) -> None:
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            self.jobs.put_nowait(None)

    def run(self) -> None:
        try:
            with serial.Serial(
                self.port_name, self.baud, timeout=3, write_timeout=3
            ) as port:
                port.dtr = False
                port.rts = False
                port.reset_input_buffer()
                while True:
                    job = self.jobs.get()
                    if job is None:
                        break
                    frame_id, crops = job
                    all_detections: list[tuple[GlobalDetection, ...]] = []
                    qnn_times: list[int] = []
                    record_counts: list[int] = []
                    started = time.perf_counter()
                    for crop, roi in zip(crops, self.rois):
                        codes = prepare_image(crop, self.manifest, array_is_bgr=True)
                        payload = pack_input_axis(codes, self.manifest)
                        tensor, qnn_us, records, _ = sparse_transaction(
                            port, payload, self.manifest
                        )
                        decoded = decode_output_tensor(tensor, self.manifest)[0]
                        all_detections.append(
                            map_detections(decoded, self.class_names, roi)
                        )
                        qnn_times.append(qnn_us)
                        record_counts.append(records)
                    self.results.put(
                        FPGAResult(
                            frame_id=frame_id,
                            detections=(all_detections[0], all_detections[1]),
                            qnn_us=(qnn_times[0], qnn_times[1]),
                            uart_seconds=time.perf_counter() - started,
                            records=(record_counts[0], record_counts[1]),
                        )
                    )
        except BaseException as exc:
            self.error = exc


def draw_overlay(
    frame: np.ndarray,
    rois: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    result: FPGAResult | None,
    display_frame: int,
    update_fps: float,
) -> np.ndarray:
    output = frame.copy()
    for roi_index, ((x1, y1, x2, y2), color) in enumerate(zip(rois, ROI_COLORS)):
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"QNN ROI {'A' if roi_index == 0 else 'B'}",
            (x1, max(70, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    counts = [{"cell": 0, "droplet": 0}, {"cell": 0, "droplet": 0}]
    if result is not None:
        for roi_index, detections in enumerate(result.detections):
            for item in detections:
                counts[roi_index][item.class_name] += 1
                x1, y1, x2, y2 = item.box
                color = CLASS_COLORS.get(item.class_name, ROI_COLORS[roi_index])
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    output,
                    f"{item.class_name} {item.confidence:.2f}",
                    (x1, max(70, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    source_result = "warming up" if result is None else f"FPGA frame {result.frame_id}"
    cv2.rectangle(output, (0, 0), (output.shape[1], 58), (12, 12, 12), -1)
    cv2.putText(
        output,
        f"Zybo Z7-10 | dual QNN FPGA | video frame {display_frame} | {source_result}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"A cell={counts[0]['cell']} drop={counts[0]['droplet']} | B cell={counts[1]['cell']} drop={counts[1]['droplet']} | FPGA updates={update_fps:.1f} FPS",
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 230, 70),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    runtime_config = json.loads(args.roi_config.read_text(encoding="utf-8"))
    config = runtime_config["rois"]
    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    rois = (
        scaled_roi(width, height, config["detection"]),
        scaled_roi(width, height, config["verification"]),
    )
    first = max(0, int(round(args.start_sec * fps)))
    last = min(total_frames, first + int(round(args.duration_sec * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)

    args.output.mkdir(parents=True, exist_ok=True)
    output_video = args.output / "zybo_dual_roi_qnn_full_video.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    worker = FPGAWorker(args.port, args.baud, manifest, class_names, rois)
    worker.start()
    latest: FPGAResult | None = None
    update_times: list[float] = []
    qnn_us_values: list[int] = []
    uart_values: list[float] = []
    record_values: list[int] = []
    displayed = 0
    preview_saved = False
    started = time.perf_counter()
    next_deadline = started
    display_elapsed = 0.0
    try:
        for frame_id in range(first, last):
            ok, frame = capture.read()
            if not ok:
                break
            crops = tuple(
                frame[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in rois
            )
            worker.submit(frame_id + 1, crops)  # type: ignore[arg-type]
            while True:
                try:
                    latest = worker.results.get_nowait()
                    update_times.append(time.perf_counter())
                    qnn_us_values.extend(latest.qnn_us)
                    uart_values.append(latest.uart_seconds)
                    record_values.extend(latest.records)
                except queue.Empty:
                    break
            if worker.error is not None:
                raise worker.error
            elapsed_updates = max(time.perf_counter() - started, 1e-6)
            update_fps = len(update_times) / elapsed_updates
            annotated = draw_overlay(
                frame,
                rois,
                latest,
                frame_id + 1,
                update_fps,
            )
            writer.write(annotated)
            if latest is not None and not preview_saved:
                cv2.imwrite(str(args.output / "preview.jpg"), annotated)
                preview_saved = True
            displayed += 1
            next_deadline += 1.0 / fps
            delay = next_deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        display_elapsed = time.perf_counter() - started
    finally:
        worker.stop()
        worker.join(timeout=8)
        writer.release()
        capture.release()

    elapsed = time.perf_counter() - started
    if not update_times:
        raise RuntimeError("No FPGA dual-ROI result was received during playback")
    report = {
        "status": "PASS" if worker.error is None else "ERROR",
        "hardware": "Digilent Zybo Z7-10 XC7Z010-1CLG400C",
        "hardware_path": "PC UART -> Zynq PS -> AXI DMA -> FINN QNN PL -> sparse PS response",
        "source": str(args.video.resolve()),
        "source_fps": fps,
        "displayed_frames": displayed,
        "display_fps": displayed / max(display_elapsed, 1e-6),
        "fpga_dual_roi_updates": len(update_times),
        "fpga_update_fps": len(update_times) / elapsed,
        "qnn_inferences": len(qnn_us_values),
        "mean_qnn_ms": statistics.fmean(qnn_us_values) / 1000.0 if qnn_us_values else None,
        "mean_dual_roi_uart_seconds": statistics.fmean(uart_values) if uart_values else None,
        "mean_sparse_records_per_roi": statistics.fmean(record_values) if record_values else None,
        "jobs_dropped_for_freshness": worker.dropped,
        "rois": {
            "A_detection": list(rois[0]),
            "B_verification": list(rois[1]),
        },
        "inference_location": "Both ROI A and ROI B execute in FPGA programmable logic",
        "host_tasks": ["video decode", "ROI crop", "raw-head decode", "overlay drawing"],
        "output_video": str(output_video.resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if worker.error is not None:
        raise worker.error
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"ZYBO_DUAL_ROI_VIDEO_PASS: {output_video.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
