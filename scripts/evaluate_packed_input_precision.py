#!/usr/bin/env python3
"""Measure detector accuracy after reducing the UART input pixel precision."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.dataset import YoloDetectionDataset, detection_collate
from qnn.evaluate_qat import evaluate_cached
from qnn.fpga_io import (
    decoder_box_calibration,
    decoder_box_constraints,
    decoder_nms_iou,
    load_manifest,
)
from qnn.model import TinyQuantDetector, config_from_dict


DEFAULT_CHECKPOINT = (
    ROOT / "models" / "qnn_cell_droplet_v2_w4a6_square192_grouped" / "best.pt"
)
DEFAULT_MANIFEST = (
    ROOT
    / "final_results"
    / "arty_s7_25_qnn_detection"
    / "07_50fps_optimized"
    / "fpga_manifest_60fps.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "dataset" / "cell_droplet_roi384_grouped",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=(8, 6, 4, 3, 2),
        help="Packed pixel precisions to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baud", type=int, default=12_000_000)
    parser.add_argument("--core-fps", type=float, default=56.1156)
    parser.add_argument("--host-overhead-ms", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "packed_input_precision",
    )
    return parser.parse_args()


def reduce_uint8_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize UINT8 codes to `bits`, then expand them back to UINT8 codes."""

    if not 1 <= bits <= 8:
        raise ValueError("Packed input precision must be between 1 and 8 bits")
    if bits == 8:
        return codes
    reduced_maximum = (1 << bits) - 1
    reduced = torch.round(codes * reduced_maximum / 255.0)
    return torch.round(reduced * 255.0 / reduced_maximum)


def collect_predictions(
    model: TinyQuantDetector,
    dataset: YoloDetectionDataset,
    *,
    bits: int,
    input_scale: float,
    device: torch.device,
    batch_size: int,
) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
    )
    cached: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            uint8_codes = torch.clamp(torch.round(images / input_scale), 0, 255)
            expanded_codes = reduce_uint8_codes(uint8_codes, bits)
            predictions = model(expanded_codes * input_scale).cpu()
            cached.append((predictions, targets))
    return cached


def projected_transport(
    *,
    bits: int,
    pixels: int,
    baud: int,
    core_fps: float,
    host_overhead_ms: float,
) -> dict[str, float | int]:
    payload_bytes = math.ceil(pixels * bits / 8)
    uart_ms = payload_bytes * 10.0 / baud * 1000.0
    core_ms = 1000.0 / core_fps
    total_ms = uart_ms + core_ms + host_overhead_ms
    return {
        "payload_bytes": payload_bytes,
        "reduction_percent": 100.0 * (1.0 - payload_bytes / pixels),
        "input_uart_ms": uart_ms,
        "core_ms": core_ms,
        "host_overhead_ms": host_overhead_ms,
        "projected_round_trip_ms": total_ms,
        "projected_update_fps": 1000.0 / total_ms,
    }


def main() -> None:
    args = parse_args()
    if len(set(args.bits)) != len(args.bits):
        raise ValueError("--bits values must be unique")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    manifest = load_manifest(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    decoder = manifest["postprocessing"]["decoder"]
    thresholds = tuple(float(value) for value in decoder["confidence_thresholds"])
    nms_iou = decoder_nms_iou(decoder)
    constraints = decoder_box_constraints(decoder)
    calibration = decoder_box_calibration(decoder)
    input_scale = float(
        manifest["preprocessing"]["input_quantization"]["scale"]
    )
    pixels = config.image_width * config.image_height

    datasets = {
        split: YoloDetectionDataset(
            args.data,
            split,
            input_size=(config.image_width, config.image_height),
            num_classes=config.num_classes,
        )
        for split in ("valid", "test")
    }
    results: dict[str, Any] = {}
    for bits in args.bits:
        split_metrics: dict[str, Any] = {}
        for split, dataset in datasets.items():
            cached = collect_predictions(
                model,
                dataset,
                bits=bits,
                input_scale=input_scale,
                device=device,
                batch_size=args.batch_size,
            )
            split_metrics[split] = evaluate_cached(
                cached,
                config=config,
                thresholds=thresholds,
                nms_iou=nms_iou,
                box_constraints=constraints,
                box_calibration=calibration,
            )
        results[str(bits)] = {
            "input_bits": bits,
            "accuracy": split_metrics,
            "transport": projected_transport(
                bits=bits,
                pixels=pixels,
                baud=args.baud,
                core_fps=args.core_fps,
                host_overhead_ms=args.host_overhead_ms,
            ),
        }
        valid = split_metrics["valid"]
        test = split_metrics["test"]
        print(
            f"{bits}-bit valid_F1={valid['f1']:.4f} test_F1={test['f1']:.4f} "
            f"projected={results[str(bits)]['transport']['projected_update_fps']:.2f}FPS",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "input_scale": input_scale,
        "selection_note": (
            "Postprocessing is fixed to the validation-selected deployment "
            "configuration in the manifest. Test is reported but not used to "
            "choose thresholds."
        ),
        "projection_note": (
            "UART projection includes 10 serial bits per payload byte, measured "
            "CNN core time, and the supplied fixed host/driver overhead. Sparse "
            "response time is small and included in that measured overhead."
        ),
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "packed_input_precision.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PACKED_INPUT_PRECISION_PASS: {output_path.resolve()}")


if __name__ == "__main__":
    main()
