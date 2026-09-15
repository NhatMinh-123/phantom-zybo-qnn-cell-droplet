#!/usr/bin/env python3
"""Export a trained TinyGridNet QAT checkpoint to QONNX with golden vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from brevitas.export import export_qonnx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

from tinygrid_qnn.config import TinyGridConfig
from tinygrid_qnn.data import FEATURE_MODES, FeatureGridDataset
from tinygrid_qnn.model import TinyGridNetQNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "dataset" / "cell_droplet_tinygrid_feature_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--golden-count", type=int, default=8)
    return parser.parse_args()


def load_qnn(checkpoint_path: Path) -> tuple[TinyGridNetQNN, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_mode = str(checkpoint["feature_mode"])
    quantization = dict(checkpoint["quantization"])
    model = TinyGridNetQNN(
        TinyGridConfig(),
        input_channels=len(FEATURE_MODES[feature_mode]),
        weight_bits=int(quantization["weight_bits"]),
        activation_bits=int(quantization["activation_bits"]),
        input_bits=int(quantization["input_bits"]),
        first_layer_weight_bits=int(quantization["first_layer_weight_bits"]),
        last_layer_weight_bits=int(quantization["last_layer_weight_bits"]),
        output_bits=int(quantization["output_bits"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    golden_dir = output / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_qnn(args.checkpoint.resolve())
    feature_mode = str(checkpoint["feature_mode"])
    channel_indices = FEATURE_MODES[feature_mode]
    config = TinyGridConfig()
    input_shape = (1, len(channel_indices), config.input_height, config.input_width)
    dummy = torch.zeros(input_shape, dtype=torch.float32)
    qonnx_path = output / "tinygrid_w4a6_qonnx.onnx"
    with torch.inference_mode():
        export_qonnx(model, input_t=dummy, export_path=str(qonnx_path))
    onnx_model = onnx.load(str(qonnx_path))
    onnx.checker.check_model(onnx_model)
    wrapped = ModelWrapper(str(qonnx_path))
    input_name = wrapped.graph.input[0].name
    output_name = wrapped.graph.output[0].name

    dataset = FeatureGridDataset(args.data.resolve(), args.split, feature_mode=feature_mode)
    count = min(max(1, args.golden_count), len(dataset))
    raw_inputs: list[np.ndarray] = []
    normalized_inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    pytorch_outputs: list[np.ndarray] = []
    qonnx_outputs: list[np.ndarray] = []
    sample_names: list[str] = []
    for sample_path in dataset.samples[:count]:
        with np.load(sample_path) as sample:
            raw = np.ascontiguousarray(sample["features"][list(channel_indices)])
            target = np.ascontiguousarray(sample["target"])
        normalized = raw.astype(np.float32) / 255.0
        with torch.inference_mode():
            pytorch_output = model(torch.from_numpy(normalized).unsqueeze(0)).numpy()
        qonnx_result = execute_onnx(wrapped, {input_name: normalized[None, ...]})[output_name]
        raw_inputs.append(raw)
        normalized_inputs.append(normalized)
        targets.append(target)
        pytorch_outputs.append(pytorch_output[0])
        qonnx_outputs.append(qonnx_result[0])
        sample_names.append(sample_path.stem)

    raw_array = np.stack(raw_inputs)
    normalized_array = np.stack(normalized_inputs)
    target_array = np.stack(targets)
    pytorch_array = np.stack(pytorch_outputs)
    qonnx_array = np.stack(qonnx_outputs)
    thresholds = np.asarray(checkpoint.get("thresholds", (0.5, 0.5)), dtype=np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-pytorch_array))
    decisions = (probabilities >= thresholds[None, :, None, None]).astype(np.uint8)

    np.save(golden_dir / "input_uint8.npy", raw_array)
    np.save(golden_dir / "input_normalized_float32.npy", normalized_array)
    np.save(golden_dir / "target_uint8.npy", target_array)
    np.save(golden_dir / "output_logits_pytorch.npy", pytorch_array)
    np.save(golden_dir / "output_logits_qonnx.npy", qonnx_array)
    np.save(golden_dir / "output_probability.npy", probabilities)
    np.save(golden_dir / "output_decision_uint8.npy", decisions)
    for index, sample_name in enumerate(sample_names):
        raw_array[index].tofile(golden_dir / f"{index:02d}_{sample_name}_input_chw_u8.bin")
        pytorch_array[index].astype(np.float32).tofile(
            golden_dir / f"{index:02d}_{sample_name}_output_chw_f32.bin"
        )

    difference = np.abs(pytorch_array - qonnx_array)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "qonnx": str(qonnx_path.resolve()),
        "feature_mode": feature_mode,
        "channel_indices": list(channel_indices),
        "quantization": checkpoint["quantization"],
        "input_name": input_name,
        "output_name": output_name,
        "input_shape_nchw": list(raw_array.shape),
        "output_shape_nchw": list(pytorch_array.shape),
        "thresholds": {
            name: float(thresholds[index]) for index, name in enumerate(config.class_names)
        },
        "golden_samples": sample_names,
        "pytorch_vs_qonnx": {
            "maximum_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "exact_element_fraction": float(np.mean(difference == 0.0)),
        },
        "onnx_checker": "PASS",
    }
    (output / "qonnx_export_report.json").write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
