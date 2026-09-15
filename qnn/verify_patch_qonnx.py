"""Infer QONNX shapes and verify the patch classifier against PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

from qnn.export_patch_qonnx import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    DEFAULT_PATCHES,
    ROOT,
    load_patch,
    sha256_file,
    verification_patches,
)
from qnn.patch_classifier import (
    TinyQuantPatchClassifier,
    patch_config_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--limit-per-class", type=int, default=8)
    parser.add_argument("--max-abs-error", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    model_path = args.model.resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = patch_config_from_dict(checkpoint["config"])
    pytorch_model = TinyQuantPatchClassifier(config)
    pytorch_model.load_state_dict(checkpoint["model_state"])
    pytorch_model.eval()

    wrapper = ModelWrapper(str(model_path)).transform(InferShapes())
    wrapper.save(str(model_path))
    input_name = wrapper.graph.input[0].name
    output_name = wrapper.graph.output[0].name
    paths = verification_patches(
        args.patches.resolve(),
        args.limit_per_class,
    )
    if not paths:
        raise FileNotFoundError(args.patches.resolve())

    results = []
    maximum_error = 0.0
    for path in paths:
        array = load_patch(path, config.input_size, config.input_transform)
        with torch.inference_mode():
            pytorch_logit = float(
                pytorch_model(torch.from_numpy(array)).item()
            )
        qonnx_logit = float(
            np.asarray(
                execute_onnx(wrapper, {input_name: array})[output_name]
            ).item()
        )
        error = abs(pytorch_logit - qonnx_logit)
        maximum_error = max(maximum_error, error)
        results.append(
            {
                "path": str(path.relative_to(ROOT).as_posix()),
                "pytorch_logit": pytorch_logit,
                "qonnx_logit": qonnx_logit,
                "absolute_error": error,
            }
        )

    passed = maximum_error <= args.max_abs_error
    metadata = {
        "schema_version": 1,
        "name": "microplastic_patch32_w4a6",
        "purpose": "candidate-gated particle/background classification",
        "checkpoint": str(checkpoint_path.relative_to(ROOT).as_posix()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "qonnx": str(model_path.relative_to(ROOT).as_posix()),
        "qonnx_sha256": sha256_file(model_path),
        "configuration": config.to_dict(),
        "input": {
            "shape_nchw": [1, 1, config.input_size, config.input_size],
            "dtype": "FLOAT32 normalized grayscale [0,1]",
        },
        "output": {
            "shape": [1, 1],
            "kind": "quantized_logit",
            "probability": "sigmoid(logit)",
            "particle_threshold": float(checkpoint.get("threshold", 0.5)),
        },
        "verification": {
            "patches": len(results),
            "maximum_absolute_logit_error": maximum_error,
            "tolerance": args.max_abs_error,
            "passed": passed,
            "items": results,
        },
        "accuracy_scope": (
            "Bootstrap patch labels; independent end-to-end video review is "
            "still required."
        ),
    }
    metadata_path = model_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"QONNX shape inference OK: {model_path}")
    print(f"patches={len(results)} maximum_logit_error={maximum_error:.8g}")
    print(f"Verification {'PASS' if passed else 'FAIL'}: {metadata_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

