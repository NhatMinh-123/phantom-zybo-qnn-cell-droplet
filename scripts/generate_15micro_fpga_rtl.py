#!/usr/bin/env python3
"""Generate checkpoint-specific requantization RTL and sparse UART top."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "fpga_rtl" / "generated_15micro"
    )
    parser.add_argument(
        "--top-name",
        default="cell_droplet_15micro_finn_uart_sparse_stream_top",
    )
    parser.add_argument("--input-bytes", type=int, default=192 * 192)
    parser.add_argument("--grid-points", type=int, default=48 * 48)
    parser.add_argument("--input-fifo-depth", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_code(probability: float, output_scale: float) -> int:
    logit = math.log(probability / (1.0 - probability))
    return min(127, max(-128, math.ceil(logit / output_scale - 1e-12)))


def threshold_values(metadata: dict) -> list[int]:
    accumulator_scale = float(metadata["head_accumulator_scale"])
    output_scale = float(metadata["output_scale"])
    values: list[int] = []
    for bias in metadata["head_bias"]:
        for output_code in range(-128, 127):
            boundary = (
                ((output_code + 0.5) * output_scale) - float(bias)
            ) / accumulator_scale
            values.append(math.ceil(boundary - 1e-12))
    if len(values) != 15 * 255:
        raise RuntimeError(f"Unexpected threshold count: {len(values)}")
    minimum, maximum = min(values), max(values)
    if minimum < -(1 << 17) or maximum > (1 << 17) - 1:
        raise RuntimeError(f"Thresholds do not fit signed 18-bit: {minimum}..{maximum}")
    return values


def write_threshold_package(path: Path, values: list[int]) -> None:
    lines = [
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "use ieee.numeric_std.all;",
        "",
        "package detector_output_thresholds_15micro_pkg is",
        "    constant DETECTOR_OUTPUT_CHANNELS : positive := 15;",
        "    constant DETECTOR_THRESHOLDS_PER_CHANNEL : positive := 255;",
        "    constant DETECTOR_THRESHOLD_BITS : positive := 18;",
        "    type detector_threshold_rom_t is array (",
        "        0 to DETECTOR_OUTPUT_CHANNELS * DETECTOR_THRESHOLDS_PER_CHANNEL - 1",
        "    ) of signed(DETECTOR_THRESHOLD_BITS - 1 downto 0);",
        "    constant DETECTOR_OUTPUT_THRESHOLDS : detector_threshold_rom_t := (",
    ]
    for index, value in enumerate(values):
        suffix = "," if index + 1 < len(values) else ""
        lines.append(
            f"        {index} => to_signed({value}, DETECTOR_THRESHOLD_BITS){suffix}"
        )
    lines.extend(["    );", "end package detector_output_thresholds_15micro_pkg;", ""])
    path.write_text("\n".join(lines), encoding="ascii", newline="\n")


def main() -> None:
    args = parse_args()
    if args.input_bytes <= 0:
        raise ValueError("--input-bytes must be positive")
    if args.grid_points <= 0:
        raise ValueError("--grid-points must be positive")
    if args.input_fifo_depth <= 0:
        raise ValueError("--input-fifo-depth must be positive")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    postprocess_payload = json.loads(args.postprocess.read_text(encoding="utf-8"))
    selected = postprocess_payload.get("selected", postprocess_payload)
    output_scale = float(metadata["output_scale"])
    cell_code = object_code(
        float(selected["confidence_thresholds"]["cell"]), output_scale
    )
    droplet_code = object_code(
        float(selected["confidence_thresholds"]["droplet"]), output_scale
    )

    args.output.mkdir(parents=True, exist_ok=True)
    package_path = args.output / "detector_output_thresholds_15micro_pkg.vhd"
    requantizer_path = args.output / "detector_output_requantizer_15micro.vhd"
    top_path = args.output / f"{args.top_name}.vhd"
    values = threshold_values(metadata)
    write_threshold_package(package_path, values)

    requantizer = (ROOT / "fpga_rtl" / "detector_output_requantizer.vhd").read_text(
        encoding="utf-8"
    )
    requantizer = requantizer.replace(
        "use work.detector_output_thresholds_pkg.all;",
        "use work.detector_output_thresholds_15micro_pkg.all;",
    ).replace("detector_output_requantizer", "detector_output_requantizer_15micro")
    requantizer_path.write_text(requantizer, encoding="ascii", newline="\n")

    top = (
        ROOT
        / "fpga_rtl"
        / "cell_droplet_finn_uart_w4a6_square192_sparse_stream_top.vhd"
    ).read_text(encoding="utf-8")
    top = top.replace(
        "cell_droplet_finn_uart_w4a6_square192_sparse_stream_top",
        args.top_name,
    ).replace(
        "entity work.detector_output_requantizer",
        "entity work.detector_output_requantizer_15micro",
    )
    top = re.sub(
        r"INPUT_BYTES\s*=>\s*\d+",
        f"INPUT_BYTES         => {args.input_bytes}",
        top,
    )
    top = re.sub(
        r"OUTPUT_WORDS\s*=>\s*\d+",
        f"OUTPUT_WORDS        => {args.grid_points * 15}",
        top,
    )
    top = re.sub(
        r"GRID_POINTS\s*=>\s*\d+",
        f"GRID_POINTS         => {args.grid_points}",
        top,
    )
    top = re.sub(
        r"INPUT_FIFO_DEPTH\s*=>\s*\d+",
        f"INPUT_FIFO_DEPTH    => {args.input_fifo_depth}",
        top,
    )
    top = re.sub(
        r"CELL_OBJECT_CODE\s*=>\s*-?\d+",
        f"CELL_OBJECT_CODE    => {cell_code}",
        top,
    )
    top = re.sub(
        r"DROPLET_OBJECT_CODE\s*=>\s*-?\d+",
        f"DROPLET_OBJECT_CODE => {droplet_code}",
        top,
    )
    top_path.write_text(top, encoding="ascii", newline="\n")

    report = {
        "metadata": str(args.metadata.resolve()),
        "postprocess": str(args.postprocess.resolve()),
        "head_accumulator_scale": float(metadata["head_accumulator_scale"]),
        "output_scale": output_scale,
        "threshold_bits": 18,
        "threshold_count": len(values),
        "threshold_min": min(values),
        "threshold_max": max(values),
        "confidence_thresholds": selected["confidence_thresholds"],
        "object_codes": {"cell": cell_code, "droplet": droplet_code},
        "stream_shape": {
            "input_bytes": args.input_bytes,
            "grid_points": args.grid_points,
            "output_words": args.grid_points * 15,
            "input_fifo_depth": args.input_fifo_depth,
        },
        "files": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (package_path, requantizer_path, top_path)
        },
    }
    report_path = args.output / "hardware_requant_config.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
