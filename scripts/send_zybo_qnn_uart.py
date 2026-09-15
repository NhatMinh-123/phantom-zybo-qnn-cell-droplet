"""Send one quantized 96x96 ROI to the Zybo QNN accelerator over PS UART.

This validates the complete hardware path: PC tensor -> UART -> PS DMA ->
FINN QNN in PL -> DMA -> UART. It is deliberately a smoke-test transport,
not a realtime-video link.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qnn.fpga_io import load_manifest, pack_input_axis, prepare_image, unpack_output_axis


RX_HEADER = b"\xA5\x5A"
TX_HEADER = b"\x5A\xA5"
ERROR_HEADER = b"ER"


def read_exact(port: serial.Serial, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        chunk = port.read(count - len(data))
        if not chunk:
            raise TimeoutError(f"Timed out after {len(data)}/{count} response bytes")
        data.extend(chunk)
    return bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM13")
    parser.add_argument("--image", type=Path, help="Source ROI or image to resize/quantize")
    parser.add_argument("--zero", action="store_true", help="Use an all-zero 96x96 tensor")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "exports/15micro_qnn_w4a6_96_roi120_v1/fpga_manifest_raw_core.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "final_results/zybo_qnn_uart")
    args = parser.parse_args()
    if bool(args.image) == bool(args.zero):
        parser.error("Select exactly one of --image or --zero")

    manifest = load_manifest(args.manifest)
    in_bytes = int(manifest["fpga_core"]["input_stream"]["bytes_per_frame"])
    out_bytes = int(manifest["fpga_core"]["output_stream"]["bytes_per_frame"])
    if args.zero:
        payload = bytes(in_bytes)
    else:
        payload = pack_input_axis(prepare_image(args.image, manifest), manifest)

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with serial.Serial(args.port, 115200, timeout=8, write_timeout=8) as port:
        port.dtr = False
        port.rts = False
        port.reset_input_buffer()
        port.write(RX_HEADER + payload)
        port.flush()
        header = read_exact(port, 2)
        if header == ERROR_HEADER:
            raise RuntimeError("Zybo DMA application reported an error")
        if header != TX_HEADER:
            raise RuntimeError(f"Unexpected response header {header.hex()}")
        result = read_exact(port, out_bytes)
    elapsed = time.perf_counter() - started

    tensor = unpack_output_axis(result, manifest)
    (args.output / "raw_head_axis.bin").write_bytes(result)
    np.save(args.output / "raw_head_nhwc_int16.npy", tensor)
    report = {
        "transport": "PS UART -> AXI DMA -> FINN QNN PL -> AXI DMA -> PS UART",
        "port": args.port,
        "input_bytes": len(payload),
        "output_bytes": len(result),
        "elapsed_seconds": elapsed,
        "transport_fps": 1.0 / elapsed,
        "output_sha256": hashlib.sha256(result).hexdigest(),
        "raw_min": int(tensor.min()),
        "raw_max": int(tensor.max()),
        "raw_sum": int(tensor.sum()),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
