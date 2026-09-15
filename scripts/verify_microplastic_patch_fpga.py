#!/usr/bin/env python3
"""Verify the Arty patch-classifier bitstream against FINN golden vectors."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

try:
    from scripts.send_microplastic_patch_uart import (
        DEFAULT_MANIFEST,
        decode_response_payload,
        detect_serial_port,
        load_manifest,
        pack_patch,
        prepare_patch,
        transact,
    )
except ModuleNotFoundError:
    from send_microplastic_patch_uart import (  # type: ignore[no-redef]
        DEFAULT_MANIFEST,
        decode_response_payload,
        detect_serial_port,
        load_manifest,
        pack_patch,
        prepare_patch,
        transact,
    )


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = WORKSPACE_ROOT / "reports" / "microplastic_patch_uart_hardware"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--port", help="Serial port, for example COM12")
    parser.add_argument("--baud", type=int)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_vector_path(manifest_path: Path, relative_path: str) -> Path:
    candidate = manifest_path.resolve().parent / relative_path
    if not candidate.is_file():
        raise FileNotFoundError(f"Verification patch not found: {candidate}")
    return candidate


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    vectors = manifest["verification"]["hardware_test_vectors"]
    if not vectors:
        raise ValueError("Manifest contains no hardware test vectors")

    import serial

    port_name = args.port or detect_serial_port()
    baud = args.baud or int(manifest["transport"]["baud"])
    results = []
    with serial.Serial(port_name, baud, timeout=0.05, write_timeout=3.0) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        time.sleep(0.05)
        for index, vector in enumerate(vectors, start=1):
            path = resolve_vector_path(manifest_path, vector["path"])
            payload = pack_patch(prepare_patch(path, manifest))
            transaction = transact(port, payload, index, args.timeout)
            decoded = decode_response_payload(transaction.pop("payload"), manifest)
            expected = int(vector["expected_raw_sum"])
            exact = decoded["raw_sum"] == expected
            results.append(
                {
                    "index": index,
                    "path": vector["path"],
                    "expected_class": vector["expected_class"],
                    "expected_raw_sum": expected,
                    "exact": exact,
                    **transaction,
                    **decoded,
                }
            )
            print(
                f"{index:02d}/{len(vectors):02d} {vector['expected_class']:<10} "
                f"expected={expected:>8} got={decoded['raw_sum']:>8} "
                f"{'PASS' if exact else 'FAIL'}"
            )

    clock_hz = int(manifest["fpga"]["clock_hz"])
    cycle_values = [int(item["accelerator_cycles"]) for item in results]
    round_trip = [float(item["round_trip_seconds"]) for item in results]
    exact_count = sum(bool(item["exact"]) for item in results)
    report = {
        "name": "microplastic_patch32_w4a6_hardware_exactness",
        "status": "pass" if exact_count == len(results) else "fail",
        "port": port_name,
        "baud": baud,
        "vectors": len(results),
        "exact_vectors": exact_count,
        "maximum_raw_sum_error": max(
            abs(int(item["raw_sum"]) - int(item["expected_raw_sum"]))
            for item in results
        ),
        "accelerator_cycles": {
            "minimum": min(cycle_values),
            "mean": statistics.fmean(cycle_values),
            "maximum": max(cycle_values),
        },
        "accelerator_patch_rate": clock_hz / statistics.fmean(cycle_values),
        "uart_round_trip_ms": {
            "minimum": min(round_trip) * 1000.0,
            "mean": statistics.fmean(round_trip) * 1000.0,
            "maximum": max(round_trip) * 1000.0,
        },
        "items": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "hardware_exactness.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Hardware exactness: {exact_count}/{len(results)}; "
        f"core={report['accelerator_patch_rate']:.1f} patch/s; "
        f"UART mean={report['uart_round_trip_ms']['mean']:.2f} ms"
    )
    print(f"Report: {report_path.resolve()}")
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
