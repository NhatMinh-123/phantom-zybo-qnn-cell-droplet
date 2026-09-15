"""Export the trained microplastic patch QNN to QONNX.

The exported graph contains only the 32x32 binary classifier. Candidate
generation and droplet localization remain outside the neural accelerator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from brevitas.export import export_qonnx
from PIL import Image
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

from qnn.patch_classifier import (
    TinyQuantPatchClassifier,
    patch_config_from_dict,
)
from qnn.patch_preprocess import preprocess_patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "models" / "qnn_microplastic_patch32_w4a6_v1" / "best.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "exports"
    / "qnn_microplastic_patch32_w4a6_v1"
    / "microplastic_patch32_w4a6.onnx"
)
DEFAULT_PATCHES = (
    ROOT / "dataset" / "microplastic_patch32_grouped_v1" / "test"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_patch(path: Path, input_size: int, input_transform: str) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("L")
        if image.size != (input_size, input_size):
            image = image.resize(
                (input_size, input_size),
                Image.Resampling.BILINEAR,
            )
        pixels_u8 = np.asarray(image, dtype=np.uint8)
        pixels_u8 = preprocess_patch(pixels_u8, input_transform)
        pixels = pixels_u8.astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(pixels[None, None])


def verification_patches(root: Path, limit_per_class: int) -> list[Path]:
    if root.is_file():
        return [root.resolve()]
    paths: list[Path] = []
    for class_name in ("background", "particle"):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        paths.extend(
            sorted(
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            )[:limit_per_class]
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--limit-per-class", type=int, default=8)
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=1e-5,
        help="Maximum accepted PyTorch/QONNX logit error",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = patch_config_from_dict(checkpoint["config"])
    model = TinyQuantPatchClassifier(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 1, config.input_size, config.input_size)
    export_qonnx(model, args=sample, export_path=str(output_path))
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)

    wrapper = ModelWrapper(str(output_path)).transform(InferShapes())
    input_name = wrapper.graph.input[0].name
    output_name = wrapper.graph.output[0].name
    patches = verification_patches(
        args.patches.resolve(),
        args.limit_per_class,
    )
    if not patches:
        raise FileNotFoundError(
            f"No verification patches found under {args.patches.resolve()}"
        )

    results = []
    maximum_error = 0.0
    for path in patches:
        array = load_patch(path, config.input_size, config.input_transform)
        with torch.inference_mode():
            pytorch_logit = float(model(torch.from_numpy(array)).item())
        qonnx_outputs = execute_onnx(wrapper, {input_name: array})
        qonnx_logit = float(np.asarray(qonnx_outputs[output_name]).item())
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

    metadata = {
        "schema_version": 1,
        "name": "microplastic_patch32_w4a6",
        "purpose": "candidate-gated particle/background classification",
        "checkpoint": str(checkpoint_path.relative_to(ROOT).as_posix()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "qonnx": str(output_path.relative_to(ROOT).as_posix()),
        "qonnx_sha256": sha256_file(output_path),
        "configuration": config.to_dict(),
        "input": {
            "shape_nchw": [1, 1, config.input_size, config.input_size],
            "dtype": "FLOAT32 carrying normalized transformed grayscale [0,1]",
            "image_transform": config.input_transform,
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
            "passed": maximum_error <= args.max_abs_error,
            "items": results,
        },
        "accuracy_scope": (
            "Bootstrap patch labels; independent end-to-end video review is "
            "still required."
        ),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"QONNX OK: {output_path}")
    print(f"nodes={len(exported.graph.node)} patches={len(results)}")
    print(f"maximum_logit_error={maximum_error:.8g}")
    print(f"metadata={metadata_path}")
    if maximum_error > args.max_abs_error:
        raise SystemExit(
            "PyTorch/QONNX equivalence failed: "
            f"{maximum_error} > {args.max_abs_error}"
        )


if __name__ == "__main__":
    main()
