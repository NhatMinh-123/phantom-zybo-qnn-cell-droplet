#!/usr/bin/env python3
"""Create the exact host/FPGA numerical contract for the FINN detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from qnn.model import TinyQuantDetector, config_from_dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "qnn_cell_droplet_v2_w4a4_grouped" / "best.pt"
DEFAULT_ONNX = (
    ROOT
    / "exports"
    / "qnn_cell_droplet_v2"
    / "tiny_detector_192x144_w4a4_rawhead.onnx"
)
DEFAULT_SYNTHESIS = (
    ROOT
    / "finn_build"
    / "windows_hls_bridge_w4a4_rawhead"
    / "stitched_windows_v2"
    / "ooc_reports"
    / "synthesis_summary.json"
)
DEFAULT_EVALUATION = (
    ROOT / "reports" / "qnn_cell_droplet_v2_w4a4_grouped" / "evaluation.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "exports"
    / "qnn_cell_droplet_v2"
    / "tiny_detector_192x144_w4a4_rawhead_fpga.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _scalar(value: torch.Tensor, name: str) -> float:
    detached = value.detach().cpu().reshape(-1)
    if detached.numel() != 1:
        raise ValueError(f"{name} must be per-tensor, got {detached.numel()} values")
    return float(detached.item())


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = TinyQuantDetector(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output_kind = str(args.output_kind)
    qonnx_path = args.qonnx if args.qonnx is not None else args.raw_head_onnx
    model_tag = (
        f"tiny_detector_{config.image_width}x{config.image_height}_"
        f"w{config.weight_bits}a{config.activation_bits}_"
        f"{'rawhead' if output_kind == 'raw_accumulator' else 'int8'}"
    )

    synthesis = json.loads(args.synthesis_summary.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    streams = synthesis["stream_contract"]
    stream_dtype = str(streams["output"]["dtype"]).upper()
    stream_dtype_match = re.fullmatch(r"(U?INT)(\d+)", stream_dtype)
    if stream_dtype_match is None or stream_dtype_match.group(1) != "INT":
        raise ValueError(f"Expected signed integer output stream, got {stream_dtype}")
    stream_bits = int(stream_dtype_match.group(2))
    if output_kind == "quantized_logits" and stream_bits != config.output_bits:
        raise ValueError(
            f"Quantized-logit stream is {stream_dtype}, checkpoint output is "
            f"INT{config.output_bits}"
        )

    expected_input_shape = [1, config.image_height, config.image_width, 1]
    expected_output_shape = [
        1,
        config.grid_height,
        config.grid_width,
        config.output_channels,
    ]
    if streams["input"]["shape_nhwc"] != expected_input_shape:
        raise ValueError("Synthesized input shape does not match checkpoint")
    if streams["output"]["shape_nhwc"] != expected_output_shape:
        raise ValueError("Synthesized output shape does not match checkpoint")

    input_scale = _scalar(model.input_quant.act_quant.scale(), "input scale")
    feature_scale = _scalar(
        model.features[-1].act.act_quant.scale(), "last activation scale"
    )
    head_weight_scale = _scalar(model.head.quant_weight().scale, "head weight scale")
    output_scale = _scalar(model.output_quant.act_quant.scale(), "output scale")
    accumulator_scale = feature_scale * head_weight_scale
    head_bias = [float(value) for value in model.head.bias.detach().cpu().tolist()]

    validation_thresholds = evaluation["validation"]["thresholds"]
    thresholds = [
        float(validation_thresholds[class_name]) for class_name in config.class_names
    ]
    nms_iou: float | list[float] = args.nms_iou
    box_constraints = None
    box_calibration = None
    postprocess_calibration = None
    if args.postprocess_config is not None:
        postprocess_calibration = json.loads(
            args.postprocess_config.read_text(encoding="utf-8")
        )
        selected = postprocess_calibration["selected"]
        thresholds = [
            float(selected["confidence_thresholds"][class_name])
            for class_name in config.class_names
        ]
        nms_iou = [
            float(selected["nms_iou"][class_name])
            for class_name in config.class_names
        ]
        selected_constraints = selected.get("box_constraints")
        if selected_constraints is not None:
            box_constraints = [
                selected_constraints.get(class_name)
                for class_name in config.class_names
            ]
        selected_calibration = selected.get("box_calibration")
        if selected_calibration is not None:
            box_calibration = [
                selected_calibration.get(class_name)
                for class_name in config.class_names
            ]

    output_quantization = {
        "dtype": "INT8",
        "bits": config.output_bits,
        "signed": True,
        "scale": output_scale,
        "zero_point": 0,
        "formula": "q=clip(round(raw_float/scale),-128,127); logits=q*scale",
    }
    fpga_output_contract: dict[str, Any]
    if output_kind == "raw_accumulator":
        fpga_output_contract = {
            "raw_accumulator": {
                "dtype": stream_dtype,
                "bits": stream_bits,
                "scale": accumulator_scale,
                "feature_scale": feature_scale,
                "head_weight_scale": head_weight_scale,
                "bias_per_channel": head_bias,
                "formula": "raw_float=accumulator*scale+bias[channel]",
            }
        }
    else:
        fpga_output_contract = {
            "quantized_logits": {
                "dtype": stream_dtype,
                "bits": stream_bits,
                "scale": output_scale,
                "zero_point": 0,
                "formula": "logits=(q-zero_point)*scale",
            }
        }

    model_artifact = {
        "qonnx": _relative(qonnx_path),
        "qonnx_sha256": _sha256(qonnx_path),
    }
    if output_kind == "raw_accumulator":
        model_artifact.update(
            {
                "raw_head_qonnx": _relative(qonnx_path),
                "raw_head_qonnx_sha256": _sha256(qonnx_path),
            }
        )

    return {
        "schema_version": 1,
        "name": f"{model_tag}_arty_s7_25",
        "model": {
            "checkpoint": _relative(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            **model_artifact,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "configuration": config.to_dict(),
        },
        "preprocessing": {
            "color_mode": "grayscale_L",
            "resize": {
                "width": config.image_width,
                "height": config.image_height,
                "method": "PIL_BILINEAR",
            },
            "normalization": "pixel_uint8 / 255.0",
            "input_quantization": {
                "dtype": "UINT8",
                "bits": config.input_bits,
                "scale": input_scale,
                "zero_point": 0,
                "formula": "q=clip(round((pixel/255)/scale),0,255)",
            },
        },
        "fpga_core": {
            "device": synthesis["device"],
            "clock_hz": streams["clock_hz"],
            "output_kind": output_kind,
            "input_stream": streams["input"],
            "output_stream": {
                **streams["output"],
                "serialized_byte_order": "little_endian",
            },
            "frame_boundary": "count AXI handshakes; the core does not emit TLAST",
            **fpga_output_contract,
        },
        "postprocessing": {
            "output_requantization": output_quantization,
            "tensor_conversion": "FPGA NHWC -> decoder NCHW",
            "decoder": {
                "class_names": list(config.class_names),
                "confidence_thresholds": thresholds,
                "anchors": [list(pair) for pair in config.anchors],
                "slots_per_class": list(config.slots_per_class),
                "nms_iou": nms_iou,
                "box_constraints": box_constraints,
                "box_calibration": box_calibration,
                "pre_nms_topk": args.pre_nms_topk,
                "max_detections": args.max_detections,
            },
        },
        "reference_evaluation": {
            "source": _relative(args.evaluation),
            "validation": evaluation["validation"],
            "test": evaluation["test"],
            "postprocess_calibration": postprocess_calibration,
            "note": (
                f"Accuracy of W{config.weight_bits}A{config.activation_bits} checkpoint; "
                "board equivalence must be verified separately."
            ),
        },
        "synthesis": {
            "source": _relative(args.synthesis_summary),
            "vivado": synthesis["vivado"],
            "finn_estimate": synthesis["finn_estimate"],
            "caveats": synthesis["caveats"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--raw-head-onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument(
        "--qonnx",
        type=Path,
        help="QONNX implemented by the FPGA; overrides --raw-head-onnx",
    )
    parser.add_argument(
        "--output-kind",
        choices=("raw_accumulator", "quantized_logits"),
        default="raw_accumulator",
        help="Meaning of each value emitted by the synthesized output stream",
    )
    parser.add_argument("--synthesis-summary", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--postprocess-config",
        type=Path,
        help="Optional validation-calibrated confidence and class-specific NMS config",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--pre-nms-topk", type=int, default=100)
    parser.add_argument("--max-detections", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output.resolve()}")
    print(
        "Contract: "
        f"{manifest['fpga_core']['input_stream']['bytes_per_frame']} input bytes, "
        f"{manifest['fpga_core']['output_stream']['bytes_per_frame']} output bytes"
    )


if __name__ == "__main__":
    main()
