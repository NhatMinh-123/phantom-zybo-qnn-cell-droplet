#!/usr/bin/env python3
"""Verify the generated VHDL requantizer against the 15 um QNN checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import prepare_image, sparse_objectness_codes
from qnn.model import TinyQuantDetector, config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "models"
        / "15micro"
        / "qnn_w4a6_192_v2_from_w8a8"
        / "best_fixed_f1.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "exports"
        / "15micro_qnn_w4a6_192_v2"
        / "fpga_manifest_sparse_uart.json",
    )
    parser.add_argument(
        "--threshold-vhdl",
        type=Path,
        default=ROOT
        / "fpga_rtl"
        / "generated_15micro"
        / "detector_output_thresholds_15micro_pkg.vhd",
    )
    parser.add_argument(
        "--top-vhdl",
        type=Path,
        default=ROOT
        / "fpga_rtl"
        / "generated_15micro"
        / "cell_droplet_15micro_finn_uart_sparse_stream_top.vhd",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "dataset" / "15micro_yolov11_v1" / "test" / "images",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "reports"
        / "15micro_qnn_v1"
        / "fpga_requant_equivalence"
        / "verification.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_thresholds(path: Path, channels: int) -> np.ndarray:
    text = path.read_text(encoding="ascii")
    values = [int(value) for value in re.findall(r"to_signed\((-?\d+),", text)]
    expected = channels * 255
    if len(values) != expected:
        raise ValueError(f"Expected {expected} thresholds, found {len(values)}")
    table = np.asarray(values, dtype=np.int64).reshape(channels, 255)
    if np.any(np.diff(table, axis=1) < 0):
        raise ValueError("VHDL threshold table is not monotonic")
    return table


def parse_top_codes(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="ascii")
    cell = re.search(r"CELL_OBJECT_CODE\s*=>\s*(-?\d+)", text)
    droplet = re.search(r"DROPLET_OBJECT_CODE\s*=>\s*(-?\d+)", text)
    if cell is None or droplet is None:
        raise ValueError("Could not parse sparse objectness codes from top VHDL")
    return int(cell.group(1)), int(droplet.group(1))


def select_images(directory: Path, limit: int) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"No images found under {directory}")
    count = min(max(1, limit), len(paths))
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[int(index)] for index in indices]


def original_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L").resize((width, height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return torch.from_numpy(np.ascontiguousarray(pixels))[None, None]


def verify_image(
    path: Path,
    model: TinyQuantDetector,
    manifest: dict[str, Any],
    thresholds: np.ndarray,
) -> dict[str, Any]:
    config = model.config
    input_scale = np.float32(manifest["preprocessing"]["input_quantization"]["scale"])
    accumulator = manifest["fpga_core"]["internal_raw_accumulator"]
    accumulator_scale = np.float32(accumulator["scale"])
    bias = np.asarray(accumulator["bias_per_channel"], dtype=np.float32)
    output_scale = np.float32(manifest["fpga_core"]["quantized_logits"]["scale"])

    codes = prepare_image(path, manifest)
    fpga_input = torch.from_numpy(codes[:, :, 0].astype(np.float32) * input_scale)[
        None, None
    ]
    original = original_tensor(path, config.image_width, config.image_height)
    with torch.inference_mode():
        quantized_original = model.input_quant(original)
        features = model.features(model.input_quant(fpga_input))
        raw = model.head(features)
        reference = model.output_quant(raw)

    raw_nhwc = raw.detach().cpu().numpy()[0].transpose(1, 2, 0)
    unrounded_accumulator = (
        raw_nhwc - bias.reshape(1, 1, -1)
    ) / accumulator_scale
    accumulator_codes = np.rint(unrounded_accumulator).astype(np.int64)
    table_codes = np.empty_like(accumulator_codes, dtype=np.int16)
    for channel in range(config.output_channels):
        table_codes[..., channel] = (
            np.searchsorted(
                thresholds[channel], accumulator_codes[..., channel], side="right"
            )
            - 128
        )

    reference_nhwc = reference.detach().cpu().numpy()[0].transpose(1, 2, 0)
    reference_codes = np.clip(np.rint(reference_nhwc / output_scale), -128, 127).astype(
        np.int16
    )
    difference = table_codes.astype(np.int32) - reference_codes.astype(np.int32)
    mismatch_count = int(np.count_nonzero(difference))
    return {
        "image": str(path.resolve().relative_to(ROOT).as_posix()),
        "values_checked": int(reference_codes.size),
        "input_quantization_max_abs_error": float(
            torch.max(torch.abs(quantized_original - fpga_input)).item()
        ),
        "accumulator_integer_residual_max": float(
            np.max(np.abs(unrounded_accumulator - accumulator_codes))
        ),
        "requant_mismatch_count": mismatch_count,
        "requant_max_abs_code_error": int(np.max(np.abs(difference))),
        "reference_code_min": int(reference_codes.min()),
        "reference_code_max": int(reference_codes.max()),
        "passed": mismatch_count == 0,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinyQuantDetector(config_from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    thresholds = parse_thresholds(args.threshold_vhdl, model.config.output_channels)
    expected_codes = sparse_objectness_codes(manifest)
    top_codes = parse_top_codes(args.top_vhdl)
    if top_codes != expected_codes:
        raise RuntimeError(
            f"Sparse threshold mismatch: VHDL={top_codes}, manifest={expected_codes}"
        )

    images = select_images(args.images, args.limit)
    rows = [verify_image(path, model, manifest, thresholds) for path in images]
    passed = all(row["passed"] for row in rows)
    report = {
        "passed": passed,
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT).as_posix()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve().relative_to(ROOT).as_posix()),
        "threshold_vhdl": str(args.threshold_vhdl.resolve().relative_to(ROOT).as_posix()),
        "threshold_vhdl_sha256": sha256(args.threshold_vhdl),
        "images_checked": len(rows),
        "values_checked": sum(row["values_checked"] for row in rows),
        "total_requant_mismatches": sum(row["requant_mismatch_count"] for row in rows),
        "maximum_code_error": max(row["requant_max_abs_code_error"] for row in rows),
        "maximum_accumulator_residual": max(
            row["accumulator_integer_residual_max"] for row in rows
        ),
        "sparse_objectness_codes": {
            "manifest": list(expected_codes),
            "top_vhdl": list(top_codes),
        },
        "images": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))
    if not passed:
        raise SystemExit("FPGA requantization equivalence FAILED")
    print(f"FPGA_REQUANT_EQUIVALENCE_PASS: {args.report.resolve()}")


if __name__ == "__main__":
    main()
