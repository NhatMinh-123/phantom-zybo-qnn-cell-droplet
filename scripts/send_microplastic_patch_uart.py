#!/usr/bin/env python3
"""Send one 32x32 grayscale candidate patch to the Arty S7-25 QNN."""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import cv2
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"
WORKSPACE_MANIFEST = (
    WORKSPACE_ROOT
    / "exports"
    / "arty_s7_25_microplastic_patch32_w4a6_uart"
    / "manifest.json"
)
SPATIAL_WORKSPACE_MANIFEST = (
    WORKSPACE_ROOT
    / "exports"
    / "arty_s7_25_microplastic_patch32_w4a6_spatial_uart"
    / "manifest.json"
)
DEFAULT_MANIFEST = (
    PACKAGED_MANIFEST
    if PACKAGED_MANIFEST.is_file()
    else (
        SPATIAL_WORKSPACE_MANIFEST
        if SPATIAL_WORKSPACE_MANIFEST.is_file()
        else WORKSPACE_MANIFEST
    )
)
DEFAULT_OUTPUT = WORKSPACE_ROOT / "reports" / "microplastic_patch_uart_hardware"

REQUEST_MAGIC = b"CDQ1"
RESPONSE_MAGIC = b"RDQ1"
RESPONSE_HEADER_SIZE = 15
STATUS_NAMES = {
    0: "OK",
    1: "INPUT_CHECKSUM_MISMATCH",
    2: "INPUT_AXIS_OVERFLOW",
    3: "INPUT_AXIS_INCOMPLETE",
    4: "CANDIDATE_OVERFLOW",
    5: "STREAM_FIFO_OVERFLOW",
}


def checksum16(payload: bytes) -> int:
    return sum(payload) & 0xFFFF


def make_request(payload: bytes, frame_id: int) -> bytes:
    if not 0 <= frame_id <= 0xFFFF:
        raise ValueError("frame_id must be between 0 and 65535")
    return (
        REQUEST_MAGIC
        + struct.pack("<H", frame_id)
        + payload
        + struct.pack("<H", checksum16(payload))
    )


def parse_response_header(header: bytes) -> tuple[int, int, int, int]:
    if len(header) != RESPONSE_HEADER_SIZE:
        raise ValueError(
            f"Expected {RESPONSE_HEADER_SIZE} response header bytes, got {len(header)}"
        )
    if header[:4] != RESPONSE_MAGIC:
        raise ValueError(f"Bad response magic: {header[:4]!r}")
    return struct.unpack("<HBII", header[4:])


def read_exact(port: Any, size: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    chunks = bytearray()
    while len(chunks) < size:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Received {len(chunks)} of {size} bytes")
        chunk = port.read(size - len(chunks))
        if chunk:
            chunks.extend(chunk)
    return bytes(chunks)


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Release manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    supported_names = {
        "arty_s7_25_microplastic_patch32_w4a6_uart",
        "arty_s7_25_microplastic_patch32_w4a6_spatial_uart",
    }
    if manifest.get("name") not in supported_names:
        raise ValueError(f"Unsupported release manifest: {path}")
    return manifest


def quantize_normalized(
    normalized: np.ndarray,
    input_thresholds: list[float] | np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(normalized, dtype=np.float32)
    if pixels.shape != (32, 32):
        raise ValueError(f"Expected normalized shape (32, 32), got {pixels.shape}")
    if not np.all(np.isfinite(pixels)):
        raise ValueError("Input contains NaN or infinity")
    thresholds = np.asarray(input_thresholds, dtype=np.float32).reshape(-1)
    if thresholds.size != 255:
        raise ValueError(f"Expected 255 UINT8 thresholds, got {thresholds.size}")
    codes = np.searchsorted(thresholds, pixels, side="right")
    return np.ascontiguousarray(codes.astype(np.uint8))


def apply_image_transform(image: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    mode = manifest.get("preprocessing", {}).get("image_transform", "raw")
    source = np.asarray(image, dtype=np.uint8)
    if mode == "raw":
        return source.copy()
    if mode == "dark_residual":
        baseline = cv2.blur(source, (9, 9), borderType=cv2.BORDER_REFLECT_101)
        residual = baseline.astype(np.int16) - source.astype(np.int16)
        return np.clip(residual * 8, 0, 255).astype(np.uint8)
    if mode == "bright_residual":
        baseline = cv2.blur(source, (9, 9), borderType=cv2.BORDER_REFLECT_101)
        residual = source.astype(np.int16) - baseline.astype(np.int16)
        return np.clip(residual * 8, 0, 255).astype(np.uint8)
    if mode == "contrast_residual":
        baseline = cv2.blur(source, (9, 9), borderType=cv2.BORDER_REFLECT_101)
        residual = np.abs(
            source.astype(np.int16) - baseline.astype(np.int16)
        )
        return np.clip(residual * 8, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown patch image transform: {mode}")


def prepare_patch(path: Path, manifest: dict[str, Any]) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("L")
        if image.size != (32, 32):
            image = image.resize((32, 32), Image.Resampling.BILINEAR)
        pixels = apply_image_transform(np.asarray(image, dtype=np.uint8), manifest)
        normalized = pixels.astype(np.float32) / np.float32(255.0)
    thresholds = manifest["preprocessing"]["input_quantization"]["thresholds"]
    return quantize_normalized(normalized, thresholds)


def pack_patch(codes: np.ndarray) -> bytes:
    array = np.asarray(codes)
    if array.shape != (32, 32) or array.dtype != np.uint8:
        raise ValueError("FPGA patch must be a UINT8 array with shape (32, 32)")
    return np.ascontiguousarray(array).tobytes(order="C")


def decode_raw_sum(raw_sum: int, manifest: dict[str, Any]) -> dict[str, Any]:
    post = manifest["postprocessing"]
    if post.get("mode") == "int8_logit":
        output_code = int(raw_sum)
        if not -128 <= output_code <= 127:
            raise ValueError(f"INT8 output code out of range: {output_code}")
        logit = output_code * float(post["output_scale"])
        if logit >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            probability = exponential / (1.0 + exponential)
        positive = output_code >= int(post["particle_output_code_threshold"])
        return {
            "raw_sum": int(raw_sum),
            "output_code": output_code,
            "logit": float(logit),
            "particle_probability": float(probability),
            "class_id": 1 if positive else 0,
            "class_name": "particle" if positive else "background",
            "accepted": positive,
        }

    average = (
        raw_sum * float(post["raw_scale"]) / int(post["average_count"])
        + float(post["raw_bias"])
    )
    thresholds = np.asarray(post["output_thresholds"], dtype=np.float64)
    crossings = int(np.count_nonzero(average >= thresholds))
    output_code = int(post["output_bias"]) + crossings
    logit = output_code * float(post["output_scale"])
    if logit >= 0.0:
        probability = 1.0 / (1.0 + math.exp(-logit))
    else:
        exponential = math.exp(logit)
        probability = exponential / (1.0 + exponential)
    positive = raw_sum >= int(post["particle_raw_sum_threshold"])
    return {
        "raw_sum": int(raw_sum),
        "average": float(average),
        "output_code": output_code,
        "logit": float(logit),
        "particle_probability": float(probability),
        "class_id": 1 if positive else 0,
        "class_name": "particle" if positive else "background",
        "accepted": positive,
    }


def decode_response_payload(
    payload: bytes,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    expected = int(manifest["fpga_boundary"]["uart_output_bytes"])
    if len(payload) != expected or expected != 4:
        raise ValueError(f"Expected one signed INT32 ({expected} bytes), got {len(payload)}")
    raw_sum = struct.unpack("<i", payload)[0]
    return decode_raw_sum(raw_sum, manifest)


def detect_serial_port() -> str:
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    ranked: list[tuple[int, str, str]] = []
    for port in ports:
        description = " ".join(
            str(value or "")
            for value in (
                port.description,
                port.manufacturer,
                port.product,
                port.interface,
            )
        ).lower()
        score = 0
        if port.vid == 0x0403 and port.pid == 0x6010:
            score += 100
        if "digilent" in description:
            score += 50
        if "uart" in description or "usb serial" in description:
            score += 10
        if score:
            ranked.append((score, port.device, port.description or ""))
    if not ranked:
        visible = ", ".join(
            f"{port.device} ({port.description})" for port in ports
        ) or "none"
        raise RuntimeError(
            "No Digilent UART found; pass --port COMx. Visible ports: " + visible
        )
    ranked.sort(reverse=True)
    best_score = ranked[0][0]
    best = [item for item in ranked if item[0] == best_score]
    if len(best) != 1:
        candidates = ", ".join(f"{item[1]} ({item[2]})" for item in best)
        raise RuntimeError("Multiple Digilent UART ports found: " + candidates)
    return best[0][1]


def transact(
    port: Any,
    payload: bytes,
    frame_id: int,
    timeout: float,
) -> dict[str, Any]:
    request = make_request(payload, frame_id)
    started = time.perf_counter()
    written = port.write(request)
    port.flush()
    if written != len(request):
        raise IOError(f"Only wrote {written} of {len(request)} request bytes")
    header = read_exact(port, RESPONSE_HEADER_SIZE, timeout)
    response_frame_id, status, payload_length, accelerator_cycles = (
        parse_response_header(header)
    )
    if response_frame_id != frame_id:
        raise ValueError(f"Frame ID mismatch: sent {frame_id}, got {response_frame_id}")
    response_payload = read_exact(port, payload_length, timeout)
    response_checksum = struct.unpack("<H", read_exact(port, 2, timeout))[0]
    if status != 0:
        status_name = STATUS_NAMES.get(status, f"UNKNOWN_{status}")
        raise RuntimeError(f"FPGA rejected patch: {status_name}")
    calculated = checksum16(response_payload)
    if response_checksum != calculated:
        raise ValueError(
            f"Output checksum mismatch: FPGA 0x{response_checksum:04X}, "
            f"PC 0x{calculated:04X}"
        )
    return {
        "payload": response_payload,
        "request_bytes": len(request),
        "response_bytes": RESPONSE_HEADER_SIZE + payload_length + 2,
        "accelerator_cycles": accelerator_cycles,
        "round_trip_seconds": time.perf_counter() - started,
        "status": STATUS_NAMES[status],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--port", help="Serial port, for example COM12")
    parser.add_argument("--baud", type=int)
    parser.add_argument("--frame-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--expected-raw-sum", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    codes = prepare_patch(args.image, manifest)
    payload = pack_patch(codes)
    request = make_request(payload, args.frame_id)
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / f"patch_{args.frame_id:05d}_request.bin"
    request_path.write_bytes(request)
    if args.dry_run:
        print(f"Prepared {len(request)} bytes: {request_path.resolve()}")
        print(f"payload_checksum=0x{checksum16(payload):04X}")
        return

    import serial

    port_name = args.port or detect_serial_port()
    baud = args.baud or int(manifest["transport"]["baud"])
    with serial.Serial(port_name, baud, timeout=0.05, write_timeout=3.0) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        transaction = transact(port, payload, args.frame_id, args.timeout)

    decoded = decode_response_payload(transaction.pop("payload"), manifest)
    clock_hz = int(manifest["fpga"]["clock_hz"])
    cycles = int(transaction["accelerator_cycles"])
    result = {
        "image": str(args.image.resolve()),
        "manifest": str(args.manifest.resolve()),
        "port": port_name,
        "baud": baud,
        **transaction,
        "accelerator_seconds": cycles / clock_hz,
        "accelerator_patch_rate": clock_hz / cycles if cycles else 0.0,
        **decoded,
    }
    if args.expected_raw_sum is not None:
        result["expected_raw_sum"] = args.expected_raw_sum
        result["exact"] = decoded["raw_sum"] == args.expected_raw_sum
    report_path = args.output / f"patch_{args.frame_id:05d}_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"FPGA {decoded['class_name']} p={decoded['particle_probability']:.4f} "
        f"raw_sum={decoded['raw_sum']} cycles={cycles} "
        f"core={result['accelerator_patch_rate']:.1f} patch/s "
        f"UART={transaction['round_trip_seconds'] * 1000.0:.2f} ms"
    )
    if args.expected_raw_sum is not None:
        verdict = "PASS" if result["exact"] else "FAIL"
        print(f"FPGA vs FINN golden {verdict}")
        if not result["exact"]:
            raise SystemExit(2)
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
