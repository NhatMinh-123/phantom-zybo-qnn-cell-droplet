#!/usr/bin/env python3
"""Run sampled video ROIs through the Zybo Z7-10 QNN over PS UART."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import (
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    prepare_image,
    unpack_output_axis,
)


RX_HEADER = b"\xA5\x5A"
TX_HEADER = b"\x5A\xA5"
ERROR_HEADER = b"ER"
COLORS = {"cell": (35, 35, 235), "droplet": (235, 150, 20)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default="COM13")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--start-sec", type=float, default=2.0)
    parser.add_argument("--end-sec", type=float)
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


def transact(
    port: serial.Serial, payload: bytes, output_bytes: int
) -> tuple[bytes, float]:
    started = time.perf_counter()
    port.write(RX_HEADER + payload)
    port.flush()
    header = read_exact(port, 2)
    if header == ERROR_HEADER:
        raise RuntimeError("Zybo DMA application reported an error")
    if header != TX_HEADER:
        raise RuntimeError(f"Unexpected Zybo response header: {header.hex()}")
    result = read_exact(port, output_bytes)
    return result, time.perf_counter() - started


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


def draw_result(
    frame: np.ndarray,
    detections: list[Any],
    class_names: list[str],
    roi: tuple[int, int, int, int],
    source_frame: int,
    sample_index: int,
    sample_count: int,
    uart_seconds: float,
) -> np.ndarray:
    output = frame.copy()
    x1, y1, x2, y2 = roi
    roi_width, roi_height = x2 - x1, y2 - y1
    cv2.rectangle(output, (x1, y1), (x2, y2), (40, 215, 40), 2)
    counts = {name: 0 for name in class_names}
    for detection in detections:
        name = class_names[detection.class_id]
        counts[name] += 1
        bx1, by1, bx2, by2 = detection.box
        left = int(round(x1 + bx1 * roi_width))
        top = int(round(y1 + by1 * roi_height))
        right = int(round(x1 + bx2 * roi_width))
        bottom = int(round(y1 + by2 * roi_height))
        color = COLORS.get(name, (40, 210, 40))
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        cv2.putText(
            output,
            f"{name} {detection.confidence:.2f}",
            (left, max(68, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (0, 0), (output.shape[1], 58), (12, 12, 12), -1)
    cv2.putText(
        output,
        f"Zybo Z7-10 FPGA QNN | sample {sample_index}/{sample_count} | source frame {source_frame}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"cell={counts.get('cell', 0)} droplet={counts.get('droplet', 0)} | UART round trip={uart_seconds:.3f}s",
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (70, 230, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Inference: FPGA PL | crop/decode/draw: PC",
        (12, output.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    roi_config = json.loads(args.roi_config.read_text(encoding="utf-8"))[
        "rois"
    ]["detection"]
    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    input_bytes = int(manifest["fpga_core"]["input_stream"]["bytes_per_frame"])
    output_bytes = int(manifest["fpga_core"]["output_stream"]["bytes_per_frame"])

    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    roi = scaled_roi(width, height, roi_config)
    first = max(0, int(round(args.start_sec * fps)))
    requested_end = args.end_sec if args.end_sec is not None else frame_count / fps - 2.0
    last = min(frame_count - 1, int(round(requested_end * fps)))
    selected = np.linspace(first, last, args.samples).round().astype(int).tolist()

    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    tensors_dir = args.output / "tensors"
    frames_dir.mkdir(exist_ok=True)
    tensors_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    annotated_frames: list[np.ndarray] = []
    x1, y1, x2, y2 = roi

    with serial.Serial(
        args.port, args.baud, timeout=8, write_timeout=8
    ) as port:
        port.dtr = False
        port.rts = False
        port.reset_input_buffer()
        for sample_index, frame_index in enumerate(selected, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index + 1}")
            codes = prepare_image(frame[y1:y2, x1:x2], manifest, array_is_bgr=True)
            payload = pack_input_axis(codes, manifest)
            if len(payload) != input_bytes:
                raise RuntimeError("Unexpected QNN input length")
            raw, elapsed = transact(port, payload, output_bytes)
            tensor = unpack_output_axis(raw, manifest)
            detections = decode_output_tensor(tensor, manifest)[0]
            np.save(tensors_dir / f"sample_{sample_index:02d}.npy", tensor)
            annotated = draw_result(
                frame,
                detections,
                class_names,
                roi,
                frame_index + 1,
                sample_index,
                len(selected),
                elapsed,
            )
            cv2.imwrite(str(frames_dir / f"sample_{sample_index:02d}.jpg"), annotated)
            annotated_frames.append(annotated)
            rows.append(
                {
                    "sample": sample_index,
                    "source_frame": frame_index + 1,
                    "source_time_sec": frame_index / fps,
                    "cells": sum(item.class_id == 0 for item in detections),
                    "droplets": sum(item.class_id == 1 for item in detections),
                    "uart_seconds": elapsed,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_min": int(tensor.min()),
                    "raw_max": int(tensor.max()),
                }
            )
            print(
                f"sample={sample_index}/{len(selected)} frame={frame_index + 1} "
                f"cell={rows[-1]['cells']} droplet={rows[-1]['droplets']} "
                f"uart={elapsed:.3f}s",
                flush=True,
            )
    capture.release()

    with (args.output / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    writer = cv2.VideoWriter(
        str(args.output / "zybo_fpga_qnn_samples.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create result video")
    for frame in annotated_frames:
        for _ in range(15):
            writer.write(frame)
    writer.release()

    thumb_width = 640
    thumb_height = int(round(height * thumb_width / width))
    thumbs = [cv2.resize(frame, (thumb_width, thumb_height)) for frame in annotated_frames]
    if len(thumbs) % 2:
        thumbs.append(np.zeros_like(thumbs[0]))
    sheet = np.vstack([np.hstack(thumbs[i : i + 2]) for i in range(0, len(thumbs), 2)])
    cv2.imwrite(str(args.output / "contact_sheet.jpg"), sheet)

    report = {
        "status": "PASS",
        "hardware": "Digilent Zybo Z7-10 XC7Z010-1CLG400C",
        "hardware_path": "PC UART -> Zynq PS -> AXI DMA -> FINN QNN in PL -> AXI DMA -> PS UART",
        "source": str(args.video.resolve()),
        "source_fps": fps,
        "source_size": [width, height],
        "roi": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
        "qnn_input": [96, 96, 1],
        "samples": len(rows),
        "selected_source_frames": [row["source_frame"] for row in rows],
        "total_cells": sum(row["cells"] for row in rows),
        "total_droplets": sum(row["droplets"] for row in rows),
        "mean_uart_seconds": float(np.mean([row["uart_seconds"] for row in rows])),
        "inference_location": "FPGA programmable logic",
        "host_tasks": ["video decode", "ROI crop", "raw-head decode", "box drawing"],
        "realtime_note": "UART 115200 is a hardware validation transport, not a realtime camera path.",
        "output_video": str((args.output / "zybo_fpga_qnn_samples.mp4").resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"ZYBO_VIDEO_QNN_PASS: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
