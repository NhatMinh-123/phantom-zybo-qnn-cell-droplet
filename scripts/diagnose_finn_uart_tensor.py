#!/usr/bin/env python3
"""Compare a captured FINN UART tensor with likely input-stream failure modes."""

from __future__ import annotations

import argparse
import math
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.fpga_io import DEFAULT_MANIFEST, load_manifest, prepare_image
from qnn.model import TinyQuantDetector, config_from_dict


DEFAULT_CAPTURE = (
    ROOT / "reports" / "fpga_uart_hardware" / "frame_00001_accumulator.npy"
)
DEFAULT_IMAGE = (
    ROOT
    / "dataset"
    / "cell_droplet_roi384_grouped"
    / "test"
    / "images"
    / "3_4_roi_src001221_tile01_x0170_y0342.jpg"
)
DEFAULT_CHECKPOINT = (
    ROOT / "models" / "qnn_cell_droplet_v2_w4a4_grouped" / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> TinyQuantDetector:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TinyQuantDetector(config_from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def accumulator(
    model: TinyQuantDetector,
    codes: np.ndarray,
    manifest: dict,
) -> np.ndarray:
    input_scale = np.float32(
        manifest["preprocessing"]["input_quantization"]["scale"]
    )
    accumulator_config = manifest["fpga_core"]["raw_accumulator"]
    accumulator_scale = np.float32(accumulator_config["scale"])
    bias = np.asarray(accumulator_config["bias_per_channel"], dtype=np.float32)
    fpga_input = torch.from_numpy(
        codes[:, :, 0].astype(np.float32) * input_scale
    )[None, None]
    with torch.inference_mode():
        raw_head = model.head(model.features(model.input_quant(fpga_input)))
    raw_nhwc = raw_head.detach().cpu().numpy()[0].transpose(1, 2, 0)
    result = np.rint(
        (raw_nhwc - bias.reshape(1, 1, -1)) / accumulator_scale
    )
    return np.clip(result, -32768, 32767).astype(np.int16)


def compare(name: str, hardware: np.ndarray, candidate: np.ndarray) -> None:
    delta = hardware.astype(np.int32) - candidate.astype(np.int32)
    exact = int(np.count_nonzero(delta == 0))
    print(
        f"{name:>18}: exact={exact:5d}/{delta.size} "
        f"mae={np.mean(np.abs(delta)):9.3f} "
        f"max={np.max(np.abs(delta)):5d} "
        f"range=[{candidate.min():5d},{candidate.max():5d}]"
    )


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    model = load_model(args.checkpoint)
    hardware = np.load(args.capture)
    codes = prepare_image(args.image, manifest)

    nonzero = np.abs(hardware.astype(np.int64).reshape(-1))
    nonzero = nonzero[nonzero != 0]
    value_gcd = reduce(math.gcd, map(int, nonzero)) if nonzero.size else 0
    print(
        "hardware:",
        f"shape={hardware.shape}",
        f"range=[{hardware.min()},{hardware.max()}]",
        f"mean={hardware.mean():.3f}",
        f"std={hardware.std():.3f}",
        f"gcd={value_gcd}",
    )
    print(
        "input codes:",
        f"range=[{codes.min()},{codes.max()}]",
        f"mean={codes.mean():.3f}",
        f"unique={np.unique(codes).size}",
    )

    compare("original", hardware, accumulator(model, codes, manifest))

    flat = codes.reshape(-1)
    shifted_left = np.concatenate((flat[1:], flat[-1:])).reshape(codes.shape)
    shifted_right = np.concatenate((flat[:1], flat[:-1])).reshape(codes.shape)
    compare("stream shift -1", hardware, accumulator(model, shifted_left, manifest))
    compare("stream shift +1", hardware, accumulator(model, shifted_right, manifest))
    compare("reversed stream", hardware, accumulator(model, flat[::-1].reshape(codes.shape), manifest))

    for value in (0, 1, 15, 16, 64, 128, 192, 240, 255):
        constant = np.full_like(codes, value)
        compare(f"constant {value}", hardware, accumulator(model, constant, manifest))


if __name__ == "__main__":
    main()
