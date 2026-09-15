#!/usr/bin/env python3
"""Print record-level differences for one dual-guard v2 FPGA transaction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import (
    load_manifest,
    pack_input_axis,
    pack_sparse_detection_axis,
    prepare_image,
)
from scripts.send_frame_15micro_dual_guard_v2_uart import (
    DEFAULT_DATA,
    DEFAULT_MANIFEST,
    apply_dual_guard_v2,
)
from scripts.send_frame_finn_uart import (
    checkpoint_output_tensor,
    resolve_checkpoint,
)
from scripts.send_frame_finn_uart_sparse import transact_sparse


def records(payload: bytes) -> dict[tuple[int, int], tuple[int, ...]]:
    result: dict[tuple[int, int], tuple[int, ...]] = {}
    for offset in range(0, len(payload), 8):
        record = payload[offset : offset + 8]
        key = (int.from_bytes(record[0:2], "little"), int(record[2]))
        result[key] = tuple(
            int(value) for value in np.frombuffer(record[3:8], dtype=np.int8)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM12")
    parser.add_argument("--baud", type=int, default=12_000_000)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    manifest = load_manifest(DEFAULT_MANIFEST)
    checkpoint = resolve_checkpoint(manifest, None)
    paths = sorted((DEFAULT_DATA / "test" / "images").glob("*"))
    image_path = paths[args.index]
    image_codes = prepare_image(image_path, manifest)
    qnn_output = checkpoint_output_tensor(image_codes, manifest, checkpoint)
    guarded, eligible, promoted = apply_dual_guard_v2(image_codes, qnn_output)
    expected_payload = pack_sparse_detection_axis(guarded, manifest)

    import serial

    with serial.Serial(args.port, args.baud, timeout=0.01, write_timeout=3.0) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        actual_payload, cycles, elapsed = transact_sparse(
            port,
            pack_input_axis(image_codes, manifest),
            frame_id=1,
            timeout=3.0,
        )

    expected = records(expected_payload)
    actual = records(actual_payload)
    expected_keys = set(expected)
    actual_keys = set(actual)
    report = {
        "image": image_path.name,
        "eligible": eligible,
        "promoted": promoted,
        "cycles": cycles,
        "elapsed_ms": elapsed * 1000.0,
        "expected_records": len(expected),
        "actual_records": len(actual),
        "missing": [
            {"grid_slot": list(key), "values": list(expected[key])}
            for key in sorted(expected_keys - actual_keys)
        ],
        "unexpected": [
            {"grid_slot": list(key), "values": list(actual[key])}
            for key in sorted(actual_keys - expected_keys)
        ],
        "changed": [
            {
                "grid_slot": list(key),
                "expected": list(expected[key]),
                "actual": list(actual[key]),
            }
            for key in sorted(expected_keys & actual_keys)
            if expected[key] != actual[key]
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
