#!/usr/bin/env python3
"""Bit-exact UART verification for the 15-micrometre dual-guard v2 FPGA."""

from __future__ import annotations

import argparse
import json
import sys
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
    checkpoint_output_tensor,
    detect_serial_port,
    resolve_checkpoint,
)
from scripts.send_frame_finn_uart_sparse import transact_sparse


DEFAULT_DATA = ROOT / "dataset" / "15micro_roi120_center_v1"
DEFAULT_MANIFEST = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "22_fpga_qnn96_roi120_validated_3_4_3_5"
    / "fpga_manifest_sparse_uart.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "23_fpga_qnn96_dual_guard_v2"
    / "hardware_equivalence"
)

CELL_LOW_OBJECT_CODE = 66
CELL_HIGH_OBJECT_CODE = 105
RADIAL_THRESHOLD_X8 = 224
SUPPORT_RADIUS = 1
GRID_STRIDE = 4
GRID_CENTER_OFFSET = 2
RADIAL_OFFSETS = (
    (0, -3),
    (0, 3),
    (-3, 0),
    (3, 0),
    (-2, -2),
    (2, -2),
    (-2, 2),
    (2, 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=12_000_000)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--clock-hz", type=int, default=108_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def radial_guard_passes(image_codes: np.ndarray, grid_x: int, grid_y: int) -> bool:
    """Mirror qnn_cell_radial_guard_filter_v2.vhd exactly."""

    grayscale = image_codes[..., 0]
    height, width = grayscale.shape
    base_x = grid_x * GRID_STRIDE + GRID_CENTER_OFFSET
    base_y = grid_y * GRID_STRIDE + GRID_CENTER_OFFSET

    for support_y in range(-SUPPORT_RADIUS, SUPPORT_RADIUS + 1):
        for support_x in range(-SUPPORT_RADIUS, SUPPORT_RADIUS + 1):
            point_x = clamp(base_x + support_x, 0, width - 1)
            point_y = clamp(base_y + support_y, 0, height - 1)
            response = 8 * int(grayscale[point_y, point_x])
            for offset_x, offset_y in RADIAL_OFFSETS:
                sample_x = clamp(point_x + offset_x, 0, width - 1)
                sample_y = clamp(point_y + offset_y, 0, height - 1)
                response -= int(grayscale[sample_y, sample_x])
            if response >= RADIAL_THRESHOLD_X8:
                return True
    return False


def apply_dual_guard_v2(
    image_codes: np.ndarray,
    output_nhwc: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Promote guard-confirmed low-confidence cell slots to code 105."""

    guarded = output_nhwc.copy()
    grid_height, grid_width, _ = guarded.shape
    eligible = 0
    promoted = 0
    for grid_y in range(grid_height):
        for grid_x in range(grid_width):
            for slot_index in (0, 1):
                object_channel = slot_index * 5
                object_code = int(guarded[grid_y, grid_x, object_channel])
                if not CELL_LOW_OBJECT_CODE <= object_code < CELL_HIGH_OBJECT_CODE:
                    continue
                eligible += 1
                if radial_guard_passes(image_codes, grid_x, grid_y):
                    guarded[grid_y, grid_x, object_channel] = np.int8(
                        CELL_HIGH_OBJECT_CODE
                    )
                    promoted += 1
    return guarded, eligible, promoted


def image_paths(data: Path, split: str, limit: int) -> list[Path]:
    directory = data.resolve() / split / "images"
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"No images found under {directory}")
    return paths


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    checkpoint = resolve_checkpoint(manifest, args.checkpoint)
    paths = image_paths(args.data, args.split, args.limit)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    port_name = args.port or detect_serial_port()

    import serial

    rows: list[dict[str, Any]] = []
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

        for frame_id, image_path in enumerate(paths, start=1):
            codes = prepare_image(image_path, manifest)
            qnn_output = checkpoint_output_tensor(codes, manifest, checkpoint)
            guarded_output, eligible, promoted = apply_dual_guard_v2(codes, qnn_output)
            expected_payload = pack_sparse_detection_axis(guarded_output, manifest)
            response, cycles, elapsed = transact_sparse(
                port,
                pack_input_axis(codes, manifest),
                frame_id=frame_id,
                timeout=args.timeout,
            )
            restored = unpack_sparse_detection_axis(response, manifest)
            payload_exact = response == expected_payload
            detections_exact = decode_output_tensor(
                restored, manifest
            ) == decode_output_tensor(guarded_output, manifest)
            row = {
                "image": image_path.name,
                "records": len(response) // 8,
                "eligible_low_cell_slots": eligible,
                "promoted_cell_slots": promoted,
                "payload_exact": payload_exact,
                "detections_exact": detections_exact,
                "stream_cycles": cycles,
                "stream_ms": cycles / args.clock_hz * 1000.0,
                "stream_fps": args.clock_hz / cycles,
                "uart_round_trip_ms": elapsed * 1000.0,
                "system_fps": 1.0 / elapsed,
            }
            rows.append(row)
            print(
                f"{frame_id:02d}/{len(paths)} {image_path.name} "
                f"records={row['records']} guard={promoted}/{eligible} "
                f"stream={row['stream_ms']:.3f}ms/{row['stream_fps']:.2f}FPS "
                f"roundtrip={row['uart_round_trip_ms']:.2f}ms "
                f"exact={payload_exact and detections_exact}",
                flush=True,
            )

    all_exact = all(
        row["payload_exact"] and row["detections_exact"] for row in rows
    )
    report = {
        "protocol": "RDS1 sparse records with RTL dual-guard v2",
        "truth_boundary": (
            "QNN, output requantization, radial guard, and sparse serialization "
            "executed in the programmed Arty S7-25 bitstream."
        ),
        "port": port_name,
        "baud": args.baud,
        "clock_hz": args.clock_hz,
        "images": len(rows),
        "exact_images": sum(
            bool(row["payload_exact"] and row["detections_exact"]) for row in rows
        ),
        "all_exact": all_exact,
        "guard_configuration": {
            "cell_low_object_code": CELL_LOW_OBJECT_CODE,
            "cell_high_object_code": CELL_HIGH_OBJECT_CODE,
            "radial_threshold_x8": RADIAL_THRESHOLD_X8,
            "support_radius": SUPPORT_RADIUS,
            "grid_stride": GRID_STRIDE,
            "grid_center_offset": GRID_CENTER_OFFSET,
        },
        "totals": {
            "eligible_low_cell_slots": sum(
                int(row["eligible_low_cell_slots"]) for row in rows
            ),
            "promoted_cell_slots": sum(int(row["promoted_cell_slots"]) for row in rows),
        },
        "performance": {
            "mean_stream_fps": float(np.mean([row["stream_fps"] for row in rows])),
            "mean_stream_ms": float(np.mean([row["stream_ms"] for row in rows])),
            "mean_uart_round_trip_ms": float(
                np.mean([row["uart_round_trip_ms"] for row in rows])
            ),
            "mean_system_fps": float(np.mean([row["system_fps"] for row in rows])),
        },
        "rows": rows,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not all_exact:
        raise RuntimeError(
            f"Dual-guard hardware mismatch: {report['exact_images']}/{len(rows)} exact"
        )
    print(f"FPGA_DUAL_GUARD_V2_PASS: {report['exact_images']}/{len(rows)} exact")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
