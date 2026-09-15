#!/usr/bin/env python3
"""Send one ROI image to the FINN UART validation wrapper and decode its tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import (
    DEFAULT_MANIFEST,
    decode_output_tensor,
    integer_dtype_range,
    load_manifest,
    output_tensor_to_logits,
    pack_input_axis,
    prepare_image,
    unpack_output_axis,
)


DEFAULT_IMAGE = (
    ROOT
    / "dataset"
    / "cell_droplet_roi384_grouped"
    / "test"
    / "images"
    / "3_4_roi_src001221_tile01_x0170_y0342.jpg"
)
DEFAULT_OUTPUT = ROOT / "reports" / "fpga_uart_hardware"
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
    return REQUEST_MAGIC + struct.pack("<H", frame_id) + payload + struct.pack(
        "<H", checksum16(payload)
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


def detection_dict(item: Any, class_names: list[str]) -> dict[str, Any]:
    return {
        "class_id": item.class_id,
        "class": class_names[item.class_id],
        "confidence": item.confidence,
        "box_normalized": list(item.box),
    }


def draw_detections(
    image_path: Path,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"cell": (225, 45, 65), "droplet": (28, 120, 235)}
    for item in detections:
        x1, y1, x2, y2 = item["box_normalized"]
        box = (
            round(x1 * image.width),
            round(y1 * image.height),
            round(x2 * image.width),
            round(y2 * image.height),
        )
        color = colors.get(item["class"], (0, 180, 80))
        draw.rectangle(box, outline=color, width=2)
        draw.text(
            (box[0] + 2, max(0, box[1] - 12)),
            f"{item['class']} {item['confidence']:.2f}",
            fill=color,
        )
    image.save(output_path, quality=95)


def checkpoint_output_tensor(
    codes: np.ndarray,
    manifest: dict[str, Any],
    checkpoint_path: Path,
) -> np.ndarray:
    """Calculate the exact FPGA-boundary tensor expected from the checkpoint."""

    import torch

    from qnn.model import TinyQuantDetector, config_from_dict

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TinyQuantDetector(config_from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_scale = np.float32(
        manifest["preprocessing"]["input_quantization"]["scale"]
    )
    fpga_input = torch.from_numpy(
        codes[:, :, 0].astype(np.float32) * input_scale
    )[None, None]
    with torch.inference_mode():
        features = model.features(model.input_quant(fpga_input))
        raw_head = model.head(features)
        quantized_logits = model.output_quant(raw_head)

    output_kind = manifest["fpga_core"].get("output_kind", "raw_accumulator")
    if output_kind == "raw_accumulator":
        output_config = manifest["fpga_core"]["raw_accumulator"]
        scale = np.float32(output_config["scale"])
        bias = np.asarray(output_config["bias_per_channel"], dtype=np.float32)
        output_nhwc = raw_head.detach().cpu().numpy()[0].transpose(1, 2, 0)
        rounded = np.rint((output_nhwc - bias.reshape(1, 1, -1)) / scale)
    elif output_kind == "quantized_logits":
        output_config = manifest["fpga_core"]["quantized_logits"]
        scale = np.float32(output_config["scale"])
        zero_point = np.float32(output_config.get("zero_point", 0))
        output_nhwc = quantized_logits.detach().cpu().numpy()[0].transpose(1, 2, 0)
        rounded = np.rint(output_nhwc / scale + zero_point)
    else:
        raise ValueError(f"Unsupported FPGA output kind: {output_kind}")

    _, bits, minimum, maximum = integer_dtype_range(
        str(manifest["fpga_core"]["output_stream"]["dtype"])
    )
    output_dtype = np.int8 if bits <= 8 else np.int16 if bits <= 16 else np.int32
    return np.clip(rounded, minimum, maximum).astype(output_dtype)


def resolve_checkpoint(
    manifest: dict[str, Any], override: Path | None
) -> Path:
    """Resolve and authenticate the checkpoint paired with the FPGA manifest."""

    checkpoint = override
    if checkpoint is None:
        configured = manifest.get("model", {}).get("checkpoint")
        if not configured:
            raise ValueError(
                "Manifest does not declare model.checkpoint; pass --checkpoint"
            )
        checkpoint = Path(configured)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    expected_hash = manifest.get("model", {}).get("checkpoint_sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual_hash.lower() != str(expected_hash).lower():
            raise ValueError(
                "Checkpoint SHA-256 does not match the FPGA manifest: "
                f"{checkpoint}"
            )
    return checkpoint


def detect_serial_port() -> str:
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found; connect the Arty S7-25 USB cable")

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
        )
        raise RuntimeError(
            "No Digilent UART port found; pass --port COMx. Visible ports: " + visible
        )

    ranked.sort(reverse=True)
    best_score = ranked[0][0]
    best = [item for item in ranked if item[0] == best_score]
    if len(best) != 1:
        candidates = ", ".join(f"{item[1]} ({item[2]})" for item in best)
        raise RuntimeError("Multiple Digilent UART ports found: " + candidates)

    print(f"Auto-detected Digilent UART: {best[0][1]} ({best[0][2]})")
    return best[0][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        help="Serial port, for example COM12; auto-detect Digilent UART if omitted",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="QAT checkpoint; defaults to model.checkpoint from the manifest",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save the request packet without opening a serial port",
    )
    parser.add_argument(
        "--no-golden-check",
        action="store_true",
        help="Decode FPGA output without comparing it to the QAT checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.frame_id <= 0xFFFF:
        raise ValueError("frame-id must be between 0 and 65535")
    if not args.dry_run and not args.port:
        args.port = detect_serial_port()

    manifest = load_manifest(args.manifest)
    checkpoint_path = None
    if not args.no_golden_check:
        checkpoint_path = resolve_checkpoint(manifest, args.checkpoint)
    codes = prepare_image(args.image, manifest)
    payload = pack_input_axis(codes, manifest)
    request = make_request(payload, args.frame_id)
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / f"frame_{args.frame_id:05d}_request.bin"
    request_path.write_bytes(request)

    if args.dry_run:
        print(f"Prepared {len(request)} bytes: {request_path.resolve()}")
        print(f"payload_checksum=0x{checksum16(payload):04X}")
        return

    import serial

    started = time.perf_counter()
    with serial.Serial(args.port, args.baud, timeout=0.1, write_timeout=10.0) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        written = port.write(request)
        port.flush()
        if written != len(request):
            raise IOError(f"Only wrote {written} of {len(request)} request bytes")

        header = read_exact(port, RESPONSE_HEADER_SIZE, args.timeout)
        response_frame_id, status, payload_length, accelerator_cycles = (
            parse_response_header(header)
        )
        if response_frame_id != args.frame_id:
            raise ValueError(
                f"Frame ID mismatch: sent {args.frame_id}, got {response_frame_id}"
            )
        status_name = STATUS_NAMES.get(status, f"UNKNOWN_{status}")
        if status != 0:
            response_checksum = struct.unpack("<H", read_exact(port, 2, args.timeout))[0]
            raise RuntimeError(
                f"FPGA rejected frame: {status_name}, checksum=0x{response_checksum:04X}"
            )

        expected_length = int(
            manifest["fpga_core"]["output_stream"]["bytes_per_frame"]
        )
        if payload_length != expected_length:
            raise ValueError(
                f"Expected {expected_length} output bytes, got {payload_length}"
            )
        response_payload = read_exact(port, payload_length, args.timeout)
        response_checksum = struct.unpack("<H", read_exact(port, 2, args.timeout))[0]

    elapsed = time.perf_counter() - started
    core_clock_hz = int(manifest["fpga_core"]["clock_hz"])
    accelerator_seconds = accelerator_cycles / core_clock_hz
    sequential_frame_rate = (
        core_clock_hz / accelerator_cycles if accelerator_cycles else 0.0
    )
    calculated_checksum = checksum16(response_payload)
    if response_checksum != calculated_checksum:
        raise ValueError(
            f"Output checksum mismatch: FPGA 0x{response_checksum:04X}, "
            f"PC 0x{calculated_checksum:04X}"
        )

    output_tensor = unpack_output_axis(response_payload, manifest)
    logits = output_tensor_to_logits(output_tensor, manifest)
    decoded = decode_output_tensor(output_tensor, manifest)[0]
    class_names = manifest["postprocessing"]["decoder"]["class_names"]
    detections = [detection_dict(item, class_names) for item in decoded]
    counts = {
        name: sum(item["class"] == name for item in detections) for name in class_names
    }

    golden_result: dict[str, Any] | None = None
    hardware_matches_checkpoint = True
    if not args.no_golden_check:
        assert checkpoint_path is not None
        golden = checkpoint_output_tensor(codes, manifest, checkpoint_path)
        delta = output_tensor.astype(np.int64) - golden.astype(np.int64)
        mismatch_count = int(np.count_nonzero(delta))
        hardware_matches_checkpoint = mismatch_count == 0
        golden_result = {
            "checkpoint": str(checkpoint_path),
            "exact": hardware_matches_checkpoint,
            "mismatch_count": mismatch_count,
            "maximum_absolute_error": int(np.max(np.abs(delta))),
        }
        np.save(args.output / f"frame_{args.frame_id:05d}_golden.npy", golden)

    output_kind = manifest["fpga_core"].get("output_kind", "raw_accumulator")
    np.save(args.output / f"frame_{args.frame_id:05d}_output.npy", output_tensor)
    if output_kind == "raw_accumulator":
        np.save(args.output / f"frame_{args.frame_id:05d}_accumulator.npy", output_tensor)
    np.save(args.output / f"frame_{args.frame_id:05d}_logits.npy", logits)
    draw_detections(
        args.image,
        detections,
        args.output / f"frame_{args.frame_id:05d}_detections.jpg",
    )
    report = {
        "frame_id": args.frame_id,
        "port": args.port,
        "baud": args.baud,
        "image": str(args.image.resolve()),
        "request_bytes": len(request),
        "response_payload_bytes": payload_length,
        "round_trip_seconds": elapsed,
        "accelerator_cycles": accelerator_cycles,
        "accelerator_seconds": accelerator_seconds,
        "sequential_frame_rate_fps": sequential_frame_rate,
        "status": status_name,
        "checksum": response_checksum,
        "output_kind": output_kind,
        "output_min": int(output_tensor.min()),
        "output_max": int(output_tensor.max()),
        "counts": counts,
        "detections": detections,
        "hardware_vs_checkpoint": golden_result,
        "note": "UART round-trip time is transport-limited and is not accelerator FPS.",
    }
    report_path = args.output / f"frame_{args.frame_id:05d}_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"FPGA response OK in {elapsed:.3f} s; "
        f"accelerator={accelerator_cycles} cycles "
        f"({accelerator_seconds * 1000.0:.3f} ms); counts={counts}"
    )
    if golden_result is not None:
        verdict = "PASS" if hardware_matches_checkpoint else "FAIL"
        print(
            f"Hardware vs checkpoint {verdict}: "
            f"mismatches={golden_result['mismatch_count']}, "
            f"max_abs_error={golden_result['maximum_absolute_error']}"
        )
    print(f"Report: {report_path.resolve()}")
    if not hardware_matches_checkpoint:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
