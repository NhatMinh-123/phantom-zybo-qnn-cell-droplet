from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import onnx
import torch
from torch import nn
from brevitas.export import export_qonnx

from qnn.model import TinyQuantDetector, config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained Brevitas model to QONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/qnn_cell_droplet/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/qnn_cell_droplet/tiny_detector_w4a4.onnx"),
    )
    parser.add_argument(
        "--output-bits",
        type=int,
        default=0,
        help="Override checkpoint output quantization bit width",
    )
    parser.add_argument(
        "--raw-head",
        action="store_true",
        help="Export head accumulators and move output requantization to host logic",
    )
    return parser.parse_args()


class RawHeadExport(nn.Module):
    def __init__(self, detector: TinyQuantDetector) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.detector.features(self.detector.input_quant(inputs))
        return self.detector.head(features)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_config = checkpoint["config"]
    config = config_from_dict(saved_config)
    if args.output_bits:
        config = replace(config, output_bits=args.output_bits)
    model = TinyQuantDetector(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sample = torch.zeros(1, 1, config.image_height, config.image_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_model: nn.Module = model
    if args.raw_head:
        export_model = RawHeadExport(model)
        input_scale = float(model.input_quant.act_quant.scale().detach().cpu())
        feature_scale = float(
            model.features[-1].act.act_quant.scale().detach().cpu()
        )
        head_weight_scale = float(model.head.quant_weight().scale.detach().cpu())
        output_scale = float(model.output_quant.act_quant.scale().detach().cpu())
        metadata = {
            "raw_head": True,
            "input_scale": input_scale,
            "head_feature_scale": feature_scale,
            "head_weight_scale": head_weight_scale,
            "head_accumulator_scale": feature_scale * head_weight_scale,
            "head_bias": [
                float(value) for value in model.head.bias.detach().cpu().tolist()
            ],
            "output_scale": output_scale,
            "output_zero_point": 0,
            "output_bits": config.output_bits,
            "output_signed": True,
            "quantize_formula": (
                "q=clamp(round(raw/output_scale),-2^(bits-1),2^(bits-1)-1); "
                "dequantized=q*output_scale"
            ),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    export_qonnx(export_model, args=sample, export_path=str(args.output))
    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    print(f"QONNX OK: {args.output}")
    print(f"nodes={len(exported.graph.node)} output_channels={config.output_channels}")
    if args.raw_head:
        print(f"Raw-head metadata: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
