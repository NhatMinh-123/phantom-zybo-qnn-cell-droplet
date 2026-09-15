#!/usr/bin/env python3
"""Verify the 12-Mbaud sparse-detection UART bitstream on labeled images."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import (
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    pack_sparse_detection_axis,
    prepare_image,
    unpack_sparse_detection_axis,
)
from scripts.send_frame_finn_uart import (
    RESPONSE_HEADER_SIZE,
    STATUS_NAMES,
    checkpoint_output_tensor,
    checksum16,
    detect_serial_port,
    make_request,
    read_exact,
    resolve_checkpoint,
)


DEFAULT_DATA = ROOT / "dataset" / "cell_droplet_roi384_grouped"
DEFAULT_MANIFEST = (
    ROOT
    / "final_results"
    / "arty_s7_25_qnn_detection"
    / "07_50fps_optimized"
    / "fpga_manifest_60fps.json"
)
SPARSE_RESPONSE_MAGIC = b"RDS1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=12_000_000)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--clock-hz", type=int, default=108_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "fpga_sparse_uart_12m_validation",
    )
    return parser.parse_args()


def parse_sparse_header(header: bytes) -> tuple[int, int, int, int]:
    if len(header) != RESPONSE_HEADER_SIZE:
        raise ValueError(
            f"Expected {RESPONSE_HEADER_SIZE} response header bytes, got {len(header)}"
        )
    if header[:4] != SPARSE_RESPONSE_MAGIC:
        raise ValueError(f"Bad sparse response magic: {header[:4]!r}")
    return struct.unpack("<HBII", header[4:])


def transact_sparse(
    port: Any,
    payload: bytes,
    *,
    frame_id: int,
    timeout: float,
) -> tuple[bytes, int, float]:
    request = make_request(payload, frame_id)
    started = time.perf_counter()
    written = port.write(request)
    port.flush()
    if written != len(request):
        raise IOError(f"Only wrote {written} of {len(request)} request bytes")

    header = read_exact(port, RESPONSE_HEADER_SIZE, timeout)
    response_frame_id, status, payload_length, stream_cycles = parse_sparse_header(
        header
    )
    if response_frame_id != frame_id:
        raise ValueError(
            f"Frame ID mismatch: sent {frame_id}, got {response_frame_id}"
        )
    if status != 0:
        name = STATUS_NAMES.get(status, "CANDIDATE_OVERFLOW" if status == 4 else None)
        raise RuntimeError(f"FPGA sparse response status: {name or status}")
    if payload_length % 8:
        raise ValueError(f"Sparse payload length is not record-aligned: {payload_length}")
    response = read_exact(port, payload_length, timeout)
    response_checksum = int.from_bytes(read_exact(port, 2, timeout), "little")
    if response_checksum != checksum16(response):
        raise ValueError("FPGA sparse-output checksum mismatch")
    return response, stream_cycles, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    checkpoint = resolve_checkpoint(manifest, args.checkpoint)
    image_dir = args.data.resolve() / args.split / "images"
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found under {image_dir}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    port_name = args.port or detect_serial_port()

    import serial

    rows: list[dict[str, Any]] = []
    exact_count = 0
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
        for index, image_path in enumerate(image_paths, start=1):
            codes = prepare_image(image_path, manifest)
            input_payload = pack_input_axis(codes, manifest)
            checkpoint_tensor = checkpoint_output_tensor(
                codes,
                manifest,
                checkpoint,
            )
            expected_payload = pack_sparse_detection_axis(
                checkpoint_tensor,
                manifest,
            )
            response, cycles, elapsed = transact_sparse(
                port,
                input_payload,
                frame_id=index,
                timeout=args.timeout,
            )
            restored = unpack_sparse_detection_axis(response, manifest)
            expected_detections = decode_output_tensor(checkpoint_tensor, manifest)
            hardware_detections = decode_output_tensor(restored, manifest)
            payload_exact = response == expected_payload
            detections_exact = hardware_detections == expected_detections
            exact = payload_exact and detections_exact
            exact_count += int(exact)
            stream_fps = args.clock_hz / cycles
            row = {
                "image": image_path.name,
                "records": len(response) // 8,
                "payload_bytes": len(response),
                "payload_exact": payload_exact,
                "detections_exact": detections_exact,
                "stream_cycles": cycles,
                "stream_ms": cycles / args.clock_hz * 1000.0,
                "stream_fps": stream_fps,
                "uart_round_trip_ms": elapsed * 1000.0,
                "system_fps": 1.0 / elapsed,
            }
            rows.append(row)
            print(
                f"{index:02d}/{len(image_paths)} {image_path.name} "
                f"records={row['records']} sparse={len(response)}B "
                f"stream={row['stream_ms']:.3f}ms/{stream_fps:.2f}FPS "
                f"roundtrip={row['uart_round_trip_ms']:.2f}ms/"
                f"{row['system_fps']:.2f}FPS exact={exact}",
                flush=True,
            )

    report = {
        "protocol": "RDS1 sparse detector records",
        "port": port_name,
        "baud": args.baud,
        "clock_hz": args.clock_hz,
        "images": len(rows),
        "exact_images": exact_count,
        "all_exact": exact_count == len(rows),
        "mean_payload_bytes": float(np.mean([row["payload_bytes"] for row in rows])),
        "maximum_payload_bytes": int(max(row["payload_bytes"] for row in rows)),
        "mean_uart_paced_stream_fps": float(
            np.mean([row["stream_fps"] for row in rows])
        ),
        "timing_scope": (
            "First FPGA input handshake through final detector output; includes "
            "UART-paced input arrival and excludes host serial API overhead."
        ),
        "mean_uart_round_trip_ms": float(
            np.mean([row["uart_round_trip_ms"] for row in rows])
        ),
        "mean_system_fps": float(np.mean([row["system_fps"] for row in rows])),
        "rows": rows,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["all_exact"]:
        raise RuntimeError(
            f"Sparse hardware mismatch: {exact_count}/{len(rows)} exact"
        )
    print(f"FPGA_SPARSE_UART_PASS: {exact_count}/{len(rows)} exact")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
