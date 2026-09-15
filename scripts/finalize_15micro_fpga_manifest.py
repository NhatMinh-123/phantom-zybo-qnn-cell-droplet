#!/usr/bin/env python3
"""Finalize the 15 um QNN sparse-UART deployment manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        default=ROOT
        / "exports"
        / "15micro_qnn_w4a6_192_v2"
        / "fpga_manifest_raw_core.json",
    )
    parser.add_argument(
        "--requant-config",
        type=Path,
        default=ROOT
        / "fpga_rtl"
        / "generated_15micro"
        / "hardware_requant_config.json",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT
        / "fpga_build"
        / "cell_droplet_15micro_finn_uart_sparse_stream_12m",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=ROOT / "configs" / "15micro_yolo11n_roi256_v1.json",
    )
    parser.add_argument(
        "--pc-video-report",
        type=Path,
        default=ROOT
        / "final_results"
        / "15micro_pipeline_v1"
        / "08_pc_qnn_w4a6_roi256_complete"
        / "report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "exports"
        / "15micro_qnn_w4a6_192_v2"
        / "fpga_manifest_sparse_uart.json",
    )
    parser.add_argument(
        "--top",
        default="cell_droplet_15micro_finn_uart_sparse_stream_top",
    )
    parser.add_argument("--clock-hz", type=int, default=108_000_000)
    parser.add_argument("--name")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_timing(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = re.search(
        r"Design Timing Summary.*?WNS\(ns\).*?\n\s*-+.*?\n\s*"
        r"(?P<wns>-?\d+\.\d+)\s+(?P<tns>-?\d+\.\d+)\s+\d+\s+\d+\s+"
        r"(?P<whs>-?\d+\.\d+)\s+(?P<ths>-?\d+\.\d+)\s+\d+\s+\d+\s+"
        r"(?P<wpws>-?\d+\.\d+)",
        text,
        flags=re.DOTALL,
    )
    clock = re.search(
        r"(?P<clock>clk\d+_unbuffered)\s+\{[^}]+\}\s+"
        r"(?P<period>\d+\.\d+)\s+(?P<frequency>\d+\.\d+)",
        text,
    )
    if summary is None or clock is None:
        raise ValueError(f"Could not parse timing report: {path}")
    return {
        "clock": clock.group("clock"),
        "period_ns": float(clock.group("period")),
        "frequency_mhz": float(clock.group("frequency")),
        "wns_ns": float(summary.group("wns")),
        "tns_ns": float(summary.group("tns")),
        "whs_ns": float(summary.group("whs")),
        "ths_ns": float(summary.group("ths")),
        "wpws_ns": float(summary.group("wpws")),
        "constraints_met": "All user specified timing constraints are met." in text,
    }


def parse_utilization(path: Path) -> dict[str, dict[str, float | int]]:
    wanted = {
        "Slice LUTs": "slice_luts",
        "Slice Registers": "slice_registers",
        "Block RAM Tile": "bram_tiles",
        "DSPs": "dsps",
        "Bonded IOB": "bonded_iob",
        "MMCME2_ADV": "mmcm",
    }
    result: dict[str, dict[str, float | int]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        columns = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(columns) < 6 or columns[0] not in wanted:
            continue
        key = wanted[columns[0]]
        if key in result:
            continue
        try:
            used = float(columns[1].replace(",", ""))
            available = float(columns[4].replace(",", ""))
            result[key] = {
                "used": int(used) if used.is_integer() else used,
                "available": int(available) if available.is_integer() else available,
                "utilization_percent": float(columns[5]),
            }
        except ValueError:
            continue
    missing = set(wanted.values()) - set(result)
    if missing:
        raise ValueError(f"Missing utilization rows: {sorted(missing)}")
    return result


def parse_power(path: Path) -> dict[str, float | str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    labels = {
        "Total On-Chip Power (W)": "total_on_chip_w",
        "Dynamic (W)": "dynamic_w",
        "Device Static (W)": "device_static_w",
        "Junction Temperature (C)": "junction_temperature_c",
        "Confidence Level": "confidence",
    }
    result: dict[str, float | str] = {}
    for label, key in labels.items():
        match = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([^|]+?)\s*\|", text)
        if match is None:
            raise ValueError(f"Missing power row {label!r}")
        value = match.group(1).strip()
        result[key] = value if key == "confidence" else float(value)
    return result


def parse_drc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    total = re.search(r"Violations found:\s*(\d+)", text)
    if total is None:
        raise ValueError(f"Could not parse DRC report: {path}")
    rules = []
    for match in re.finditer(
        r"^\|\s*([A-Z0-9-]+)\s*\|\s*(Warning|Critical Warning|Error)\s*"
        r"\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|$",
        text,
        flags=re.MULTILINE,
    ):
        rules.append(
            {
                "rule": match.group(1),
                "severity": match.group(2),
                "description": match.group(3).strip(),
                "violations": int(match.group(4)),
            }
        )
    return {
        "violations": int(total.group(1)),
        "errors": sum(item["violations"] for item in rules if item["severity"] == "Error"),
        "critical_warnings": sum(
            item["violations"] for item in rules if item["severity"] == "Critical Warning"
        ),
        "warnings": sum(
            item["violations"] for item in rules if item["severity"] == "Warning"
        ),
        "rules": rules,
    }


def main() -> None:
    args = parse_args()
    raw = json.loads(args.raw_manifest.read_text(encoding="utf-8"))
    requant = json.loads(args.requant_config.read_text(encoding="utf-8"))
    roi = json.loads(args.roi_config.read_text(encoding="utf-8"))
    pc_video = json.loads(args.pc_video_report.read_text(encoding="utf-8"))
    manifest = copy.deepcopy(raw)

    report_dir = args.build_dir / "reports"
    top = args.top
    bitstream = args.build_dir / f"{top}.bit"
    implementation = {
        "tool": "Vivado 2022.2",
        "device": "xc7s25csga324-1",
        "top": top,
        "timing": parse_timing(report_dir / "timing_routed.rpt"),
        "utilization": parse_utilization(report_dir / "utilization_routed.rpt"),
        "power_estimate": parse_power(report_dir / "power_routed.rpt"),
        "drc": parse_drc(report_dir / "drc_routed.rpt"),
        "bitstream": {
            "path": relative(bitstream),
            "bytes": bitstream.stat().st_size,
            "sha256": sha256(bitstream),
        },
    }
    if not implementation["timing"]["constraints_met"]:
        raise RuntimeError("Routed timing constraints are not met")
    if implementation["drc"]["errors"]:
        raise RuntimeError("Routed design has DRC errors")

    fpga = manifest["fpga_core"]
    raw_accumulator = fpga.pop("raw_accumulator")
    output_scale = float(manifest["postprocessing"]["output_requantization"]["scale"])
    shape = list(fpga["output_stream"]["shape_nhwc"])
    values_per_frame = shape[1] * shape[2] * shape[3]
    fpga.update(
        {
            "clock_hz": args.clock_hz,
            "output_kind": "quantized_logits",
            "output_stream": {
                "shape_nhwc": shape,
                "dtype": "INT8",
                "axis_tdata_bits": 8,
                "beats_per_frame": values_per_frame,
                "bytes_per_frame": values_per_frame,
                "tlast": False,
                "serialized_byte_order": "little_endian",
            },
            "quantized_logits": {
                "dtype": "INT8",
                "bits": 8,
                "scale": output_scale,
                "zero_point": 0,
                "formula": "logits=(q-zero_point)*scale",
            },
            "internal_raw_accumulator": raw_accumulator,
            "requantization_rtl": requant,
        }
    )

    manifest["schema_version"] = 2
    input_shape = list(fpga["input_stream"]["shape_nhwc"])
    manifest["name"] = args.name or (
        f"15micro_tiny_detector_{input_shape[2]}x{input_shape[1]}"
        "_w4a6_arty_s7_25_sparse_uart"
    )
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["runtime"] = {
        "roi": roi["roi"],
        "preprocess": {
            "mode": "fixed_roi_grayscale_resize",
            "input_width": input_shape[2],
            "input_height": input_shape[1],
        },
        "uart": {
            "baud": 12_000_000,
            "clock_hz": args.clock_hz,
            "clocks_per_bit": args.clock_hz // 12_000_000,
            "protocol": "RDS1 sparse detector records",
            "record_bytes": 8,
            "maximum_candidates": 256,
        },
        "pc_pre_fpga_video": pc_video,
    }
    manifest["implementation"] = implementation
    manifest["hardware_validation"] = {
        "status": "pending_board_connection",
        "programmed": False,
        "uart_equivalence": "pending",
        "video_run": "pending",
        "truth_boundary": "Bitstream built but not yet executed on the Arty S7-25",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output.resolve()}")
    print(json.dumps(implementation, indent=2))


if __name__ == "__main__":
    main()
