#!/usr/bin/env python3
"""Verify the host/FPGA numerical boundary against the trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from qnn.detection import Detection, decode_predictions
from qnn.fpga_io import (
    DEFAULT_MANIFEST,
    decoder_box_calibration,
    decoder_box_constraints,
    decoder_nms_iou,
    integer_dtype_range,
    load_manifest,
    output_tensor_to_logits,
    pack_output_axis,
    prepare_image,
    unpack_output_axis,
)
from qnn.model import TinyQuantDetector, config_from_dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "qnn_cell_droplet_v2_w4a4_grouped" / "best.pt"
DEFAULT_IMAGES = ROOT / "dataset" / "cell_droplet_roi384_grouped" / "test" / "images"
DEFAULT_REPORT = (
    ROOT / "reports" / "fpga_io_equivalence_w4a4_rawhead" / "verification.json"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _serialize_detection(item: Detection) -> dict[str, Any]:
    return {
        "class_id": item.class_id,
        "confidence": item.confidence,
        "box": list(item.box),
    }


def _decode(logits: np.ndarray, manifest: dict[str, Any]) -> list[Detection]:
    decoder = manifest["postprocessing"]["decoder"]
    return decode_predictions(
        torch.from_numpy(logits),
        confidence_threshold=tuple(float(x) for x in decoder["confidence_thresholds"]),
        nms_iou=decoder_nms_iou(decoder),
        box_constraints=decoder_box_constraints(decoder),
        box_calibration=decoder_box_calibration(decoder),
        pre_nms_topk=int(decoder["pre_nms_topk"]),
        max_detections=int(decoder["max_detections"]),
        anchors=tuple(tuple(float(x) for x in pair) for pair in decoder["anchors"]),
        slots_per_class=tuple(int(x) for x in decoder["slots_per_class"]),
    )[0]


def _detections_equal(left: list[Detection], right: list[Detection]) -> bool:
    if len(left) != len(right):
        return False
    for first, second in zip(left, right):
        if first.class_id != second.class_id:
            return False
        if abs(first.confidence - second.confidence) > 1e-6:
            return False
        if max(abs(a - b) for a, b in zip(first.box, second.box)) > 1e-6:
            return False
    return True


def _load_original(path: Path, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L").resize((width, height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return torch.from_numpy(np.ascontiguousarray(pixels))[None, None]


def verify_image(
    path: Path,
    model: TinyQuantDetector,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    config = model.config
    input_scale = np.float32(
        manifest["preprocessing"]["input_quantization"]["scale"]
    )
    codes = prepare_image(path, manifest)
    fpga_input = torch.from_numpy(
        codes[:, :, 0].astype(np.float32) * input_scale
    )[None, None]
    original_input = _load_original(path, config.image_width, config.image_height)

    with torch.inference_mode():
        quantized_original = model.input_quant(original_input)
        features = model.features(model.input_quant(fpga_input))
        raw_head = model.head(features)
        reference_logits = model.output_quant(raw_head)

    output_kind = manifest["fpga_core"].get("output_kind", "raw_accumulator")
    if output_kind == "raw_accumulator":
        output_config = manifest["fpga_core"]["raw_accumulator"]
        output_scale = np.float32(output_config["scale"])
        bias = np.asarray(output_config["bias_per_channel"], dtype=np.float32)
        output_nhwc = raw_head.detach().cpu().numpy()[0].transpose(1, 2, 0)
        unrounded = (output_nhwc - bias.reshape(1, 1, -1)) / output_scale
    elif output_kind == "quantized_logits":
        output_config = manifest["fpga_core"]["quantized_logits"]
        output_scale = np.float32(output_config["scale"])
        zero_point = np.float32(output_config.get("zero_point", 0))
        output_nhwc = reference_logits.detach().cpu().numpy()[0].transpose(1, 2, 0)
        unrounded = output_nhwc / output_scale + zero_point
    else:
        raise ValueError(f"Unsupported FPGA output kind: {output_kind}")

    rounded = np.rint(unrounded)
    _, output_bits, output_minimum, output_maximum = integer_dtype_range(
        str(manifest["fpga_core"]["output_stream"]["dtype"])
    )
    clipped = np.clip(rounded, output_minimum, output_maximum)
    output_dtype = np.int8 if output_bits <= 8 else np.int16 if output_bits <= 16 else np.int32
    output_tensor = clipped.astype(output_dtype)

    payload = pack_output_axis(output_tensor, manifest)
    unpacked = unpack_output_axis(payload, manifest)
    reconstructed_logits = output_tensor_to_logits(unpacked, manifest)
    reference = reference_logits.detach().cpu().numpy()

    reference_detections = _decode(reference, manifest)
    reconstructed_detections = _decode(reconstructed_logits, manifest)
    mismatch_count = int(np.count_nonzero(reference != reconstructed_logits))
    clipped_count = int(np.count_nonzero(rounded != clipped))

    result = {
        "image": str(path.resolve().relative_to(ROOT).as_posix()),
        "output_kind": output_kind,
        "input_quantization_max_abs_error": float(
            torch.max(torch.abs(quantized_original - fpga_input)).item()
        ),
        "output_integer_residual_max": float(np.max(np.abs(unrounded - rounded))),
        "output_min": int(output_tensor.min()),
        "output_max": int(output_tensor.max()),
        "output_clipped_values": clipped_count,
        "logit_max_abs_error": float(np.max(np.abs(reference - reconstructed_logits))),
        "logit_mismatch_count": mismatch_count,
        "reference_detection_count": len(reference_detections),
        "reconstructed_detection_count": len(reconstructed_detections),
        "detections_exact": _detections_equal(
            reference_detections, reconstructed_detections
        ),
        "detections": [_serialize_detection(item) for item in reconstructed_detections],
    }
    if output_kind == "raw_accumulator":
        result.update(
            {
                "accumulator_integer_residual_max": result["output_integer_residual_max"],
                "accumulator_min": result["output_min"],
                "accumulator_max": result["output_max"],
                "accumulator_clipped_values": result["output_clipped_values"],
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TinyQuantDetector(config_from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if args.images.is_file():
        paths = [args.images.resolve()]
    else:
        paths = sorted(
            path
            for path in args.images.resolve().iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No test images found under {args.images}")

    images = [verify_image(path, model, manifest) for path in paths]
    passed = all(
        item["input_quantization_max_abs_error"] <= 1e-6
        and item["output_clipped_values"] == 0
        and item["logit_max_abs_error"] <= 1e-6
        and item["detections_exact"]
        for item in images
    )
    report = {
        "passed": passed,
        "checkpoint": str(checkpoint_path.relative_to(ROOT).as_posix()),
        "manifest": str(manifest_path.relative_to(ROOT).as_posix()),
        "images_checked": len(images),
        "maximums": {
            "input_quantization_abs_error": max(
                item["input_quantization_max_abs_error"] for item in images
            ),
            "output_integer_residual": max(
                item["output_integer_residual_max"] for item in images
            ),
            "logit_abs_error": max(item["logit_max_abs_error"] for item in images),
        },
        "images": images,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["maximums"], indent=2))
    print(f"detections_exact={all(item['detections_exact'] for item in images)}")
    print(f"Verification {'PASS' if passed else 'FAIL'}: {args.report.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
