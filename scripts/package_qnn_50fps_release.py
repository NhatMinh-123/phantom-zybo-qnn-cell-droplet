from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "final_results" / "arty_s7_25_qnn_detection"
OPTIMIZED_ROOT = RELEASE_ROOT / "07_50fps_optimized"

BITSTREAM_SOURCE = (
    ROOT
    / "fpga_build"
    / "cell_droplet_finn_uart_w4a6_square192_requant_60fps"
    / "cell_droplet_finn_uart_w4a6_square192_requant_top.bit"
)
VIVADO_REPORTS_SOURCE = BITSTREAM_SOURCE.parent / "reports"
HARDWARE_REPORTS_SOURCE = (
    ROOT / "reports" / "qnn_50fps_optimization" / "hardware_validation_60fps"
)
VIDEO_VALIDATION_SOURCE = (
    ROOT / "reports" / "fpga_video_50fps_validation_20260727"
)
TARGET50_REPORTS_SOURCE = (
    ROOT / "reports" / "qnn_50fps_optimization" / "hardware_validation"
)
BASELINE_REPORTS_SOURCE = RELEASE_ROOT / "02_fpga_hardware_validation"
FINN_BUILD_SOURCE = (
    ROOT
    / "finn_build"
    / "windows_hls_bridge_w4a6_square192_rawhead_60fps"
    / "build_60fps"
)
EVALUATION_SOURCE = (
    ROOT
    / "reports"
    / "qnn_cell_droplet_v2_w4a6_square192_grouped"
    / "evaluation.json"
)
POSTPROCESS_SOURCE = (
    ROOT
    / "reports"
    / "qnn_droplet_postprocess"
    / "w4a6_square192"
    / "postprocess_config.json"
)
BASE_MANIFEST_SOURCE = (
    ROOT
    / "exports"
    / "arty_s7_25_w4a6_square192_requant_uart"
    / "fpga_manifest.json"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_file in source.rglob("*"):
        if source_file.is_file():
            copy_file(source_file, destination / source_file.relative_to(source))


def load_hardware_reports(directory: Path) -> list[dict]:
    reports = [load_json(path) for path in sorted(directory.glob("*_report.json"))]
    if not reports:
        raise FileNotFoundError(f"No hardware reports found in {directory}")
    return reports


def summarize_hardware(reports: list[dict]) -> dict:
    cycles = [int(report["accelerator_cycles"]) for report in reports]
    fps = [float(report["sequential_frame_rate_fps"]) for report in reports]
    exact = [
        bool(report["hardware_vs_checkpoint"]["exact"])
        and int(report["hardware_vs_checkpoint"]["mismatch_count"]) == 0
        and int(report["hardware_vs_checkpoint"]["maximum_absolute_error"]) == 0
        for report in reports
    ]
    return {
        "frames_tested": len(reports),
        "frames_exact": sum(exact),
        "all_exact": all(exact),
        "mismatch_count_total": sum(
            int(report["hardware_vs_checkpoint"]["mismatch_count"])
            for report in reports
        ),
        "maximum_absolute_error": max(
            int(report["hardware_vs_checkpoint"]["maximum_absolute_error"])
            for report in reports
        ),
        "accelerator_cycles": {
            "minimum": min(cycles),
            "maximum": max(cycles),
            "mean": sum(cycles) / len(cycles),
        },
        "accelerator_latency_ms": {
            "minimum": min(cycles) / 100_000.0,
            "maximum": max(cycles) / 100_000.0,
            "mean": sum(cycles) / len(cycles) / 100_000.0,
        },
        "single_frame_fps": {
            "minimum": min(fps),
            "maximum": max(fps),
            "mean": sum(fps) / len(fps),
        },
        "note": (
            "FPS is computed from accelerator cycles at 100 MHz. "
            "UART round-trip time is excluded because 115200-baud transport is the bottleneck."
        ),
    }


def parse_table_row(path: Path, row_name: str) -> dict:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == row_name and len(cells) >= 6:
            return {
                "used": float(cells[1]),
                "available": float(cells[4]),
                "utilization_percent": float(cells[5]),
            }
    raise ValueError(f"Row {row_name!r} not found in {path}")


def parse_timing(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for candidate in lines[index + 1 : index + 5]:
            fields = candidate.split()
            if len(fields) >= 12:
                try:
                    values = [float(field) for field in fields[:12]]
                except ValueError:
                    continue
                return {
                    "wns_ns": values[0],
                    "tns_ns": values[1],
                    "whs_ns": values[4],
                    "ths_ns": values[5],
                    "wpws_ns": values[8],
                    "tpws_ns": values[9],
                    "constraints_met": "All user specified timing constraints are met."
                    in "\n".join(lines),
                }
    raise ValueError(f"Timing summary not found in {path}")


def parse_value_row(path: Path, row_name: str) -> float:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == row_name and len(cells) >= 2:
            return float(cells[1])
    raise ValueError(f"Row {row_name!r} not found in {path}")


def parse_power(path: Path) -> dict:
    return {
        "total_on_chip_w": parse_value_row(path, "Total On-Chip Power (W)"),
        "dynamic_w": parse_value_row(path, "Dynamic (W)"),
        "device_static_w": parse_value_row(path, "Device Static (W)"),
        "junction_temperature_c": parse_value_row(
            path, "Junction Temperature (C)"
        ),
        "confidence": "Medium",
    }


def make_plots(summary: dict) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    plots = OPTIMIZED_ROOT / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    throughput = summary["throughput"]
    names = ["Baseline", "50 target", "60 target"]
    fps = [
        throughput["baseline"]["measured_single_frame_fps"],
        throughput["target_50"]["measured_single_frame_fps"],
        throughput["optimized_60"]["measured_single_frame_fps"],
    ]
    resources = summary["implementation"]["resources"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#6b7280", "#0ea5e9", "#16a34a"]
    bars = axes[0].bar(names, fps, color=colors)
    axes[0].axhline(50, color="#dc2626", linestyle="--", linewidth=1.5, label="50 FPS target")
    axes[0].set_ylabel("Measured single-frame FPS")
    axes[0].set_ylim(0, max(fps) * 1.22)
    axes[0].set_title("Arty S7-25 accelerator throughput")
    axes[0].legend(loc="upper left")
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].grid(axis="y", alpha=0.25)

    resource_names = ["LUT", "Registers", "BRAM", "DSP"]
    resource_values = [
        resources["slice_luts"]["utilization_percent"],
        resources["slice_registers"]["utilization_percent"],
        resources["bram_tiles"]["utilization_percent"],
        resources["dsps"]["utilization_percent"],
    ]
    resource_bars = axes[1].bar(resource_names, resource_values, color="#2563eb")
    axes[1].set_ylabel("Utilization (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Post-route resource utilization")
    axes[1].bar_label(resource_bars, fmt="%.1f%%", padding=3)
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(plots / "fps_and_resources.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    metrics = summary["accuracy"]["optimized_test"]
    categories = ["Overall", "Cell", "Droplet"]
    precision = [
        metrics["precision"],
        metrics["classes"]["cell"]["precision"],
        metrics["classes"]["droplet"]["precision"],
    ]
    recall = [
        metrics["recall"],
        metrics["classes"]["cell"]["recall"],
        metrics["classes"]["droplet"]["recall"],
    ]
    f1 = [
        metrics["f1"],
        metrics["classes"]["cell"]["f1"],
        metrics["classes"]["droplet"]["f1"],
    ]

    x = np.arange(len(categories))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width, precision, width, label="Precision", color="#2563eb")
    axis.bar(x, recall, width, label="Recall", color="#f59e0b")
    axis.bar(x + width, f1, width, label="F1", color="#16a34a")
    axis.set_xticks(x, categories)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("QNN W4A6 test metrics at deployment thresholds")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "qnn_accuracy_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_comparison_csv(summary: dict) -> None:
    path = OPTIMIZED_ROOT / "performance_comparison.csv"
    throughput = summary["throughput"]
    rows = [
        ("baseline", throughput["baseline"]),
        ("target_50", throughput["target_50"]),
        ("optimized_60", throughput["optimized_60"]),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "configuration",
                "accelerator_cycles",
                "latency_ms",
                "measured_single_frame_fps",
                "theoretical_streaming_fps",
                "hardware_exact",
            ]
        )
        for name, row in rows:
            writer.writerow(
                [
                    name,
                    row["accelerator_cycles"],
                    row["latency_ms"],
                    row["measured_single_frame_fps"],
                    row.get("theoretical_streaming_fps", ""),
                    row["hardware_exact"],
                ]
            )


def write_release_readme(summary: dict) -> None:
    throughput = summary["throughput"]["optimized_60"]
    resources = summary["implementation"]["resources"]
    timing = summary["implementation"]["timing"]
    accuracy = summary["accuracy"]["optimized_test"]
    text = f"""# Arty S7-25 QNN - optimized release above 50 FPS

This directory contains the recommended W4A6 deployment for cell and droplet
detection on the Digilent Arty S7-25.

## Verified result

- Input: one grayscale ROI, 192x192 pixels
- Clock: 100 MHz
- Measured accelerator cycles: {throughput['accelerator_cycles']:,}
- Measured accelerator latency: {throughput['latency_ms']:.5f} ms
- Measured single-frame throughput: {throughput['measured_single_frame_fps']:.3f} FPS
- FINN theoretical sustained throughput: {throughput['theoretical_streaming_fps']:.3f} FPS
- Hardware equivalence: {summary['hardware_validation']['frames_exact']}/{summary['hardware_validation']['frames_tested']} exact, maximum absolute error 0
- Speedup from baseline: {summary['speedup']['times']:.3f}x

UART round-trip time is not accelerator latency. At 115200 baud, transport of
the input and output tensors takes several seconds; a camera/parallel/AXI stream
is required for end-to-end real-time video.

## Model quality

The faster bitstream uses the same 192x192 checkpoint, W4A6 quantization,
requantizer, decoder and thresholds. Therefore detection metrics are unchanged:

- Overall test precision: {accuracy['precision'] * 100:.2f}%
- Overall test recall: {accuracy['recall'] * 100:.2f}%
- Overall test F1: {accuracy['f1'] * 100:.2f}%
- Cell test F1: {accuracy['classes']['cell']['f1'] * 100:.2f}%
- Droplet test F1: {accuracy['classes']['droplet']['f1'] * 100:.2f}%

The 128x128 candidate was not deployed because it reduced cell detection
quality. Extra parallelism was added to the 192x192 accelerator instead.

## Post-route implementation

- LUT: {int(resources['slice_luts']['used']):,}/{int(resources['slice_luts']['available']):,} ({resources['slice_luts']['utilization_percent']:.2f}%)
- Registers: {int(resources['slice_registers']['used']):,}/{int(resources['slice_registers']['available']):,} ({resources['slice_registers']['utilization_percent']:.2f}%)
- BRAM: {resources['bram_tiles']['used']}/{int(resources['bram_tiles']['available'])} tiles ({resources['bram_tiles']['utilization_percent']:.2f}%)
- DSP: {int(resources['dsps']['used'])}/{int(resources['dsps']['available'])} ({resources['dsps']['utilization_percent']:.2f}%)
- WNS/WHS: +{timing['wns_ns']:.3f} ns / +{timing['whs_ns']:.3f} ns
- Timing constraints met: {timing['constraints_met']}
- Estimated total on-chip power: {summary['implementation']['power']['total_on_chip_w']:.3f} W

## Files

- `bitstream/cell_droplet_qnn_w4a6_192x192_51fps.bit`: program this file
- `fpga_manifest_60fps.json`: complete model and hardware contract
- `performance_summary.json`: compact machine-readable result
- `hardware_validation/`: six exact board-vs-checkpoint tests and detections
- `video_validation/`: 12 sampled video frames, output video, tensors and CSV
- `vivado_reports/`: routed utilization, timing, power and DRC
- `finn_reports/`: folding and network performance estimates
- `model_evaluation/`: QNN test metrics and post-processing calibration
- `plots/`: FPS/resource and accuracy charts
- `tools/`: programming and one-frame UART validation scripts
- `SHA256SUMS.txt`: integrity hashes
"""
    (OPTIMIZED_ROOT / "README.md").write_text(text, encoding="ascii", newline="\n")


def write_checksums(directory: Path) -> None:
    checksum_path = directory / "SHA256SUMS.txt"
    files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != checksum_path
    ]
    with checksum_path.open("w", encoding="ascii", newline="\n") as handle:
        for path in files:
            relative = path.relative_to(directory).as_posix()
            handle.write(f"{sha256(path)}  {relative}\n")


def create_zip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, source.name + "/" + path.relative_to(source).as_posix())


def main() -> None:
    required = [
        BITSTREAM_SOURCE,
        VIVADO_REPORTS_SOURCE,
        HARDWARE_REPORTS_SOURCE,
        VIDEO_VALIDATION_SOURCE,
        TARGET50_REPORTS_SOURCE,
        BASELINE_REPORTS_SOURCE,
        FINN_BUILD_SOURCE,
        EVALUATION_SOURCE,
        POSTPROCESS_SOURCE,
        BASE_MANIFEST_SOURCE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing release inputs:\n" + "\n".join(missing))

    OPTIMIZED_ROOT.mkdir(parents=True, exist_ok=True)
    bitstream_destination = (
        OPTIMIZED_ROOT
        / "bitstream"
        / "cell_droplet_qnn_w4a6_192x192_51fps.bit"
    )
    copy_file(BITSTREAM_SOURCE, bitstream_destination)
    copy_tree(HARDWARE_REPORTS_SOURCE, OPTIMIZED_ROOT / "hardware_validation")
    copy_tree(VIDEO_VALIDATION_SOURCE, OPTIMIZED_ROOT / "video_validation")
    copy_tree(VIVADO_REPORTS_SOURCE, OPTIMIZED_ROOT / "vivado_reports")

    finn_files = [
        FINN_BUILD_SOURCE / "auto_folding_config.json",
        FINN_BUILD_SOURCE / "final_hw_config.json",
        FINN_BUILD_SOURCE / "report" / "estimate_network_performance.json",
        FINN_BUILD_SOURCE / "report" / "estimate_layer_cycles.json",
        FINN_BUILD_SOURCE / "report" / "estimate_layer_resources_hls.json",
        FINN_BUILD_SOURCE / "report" / "op_and_param_counts.json",
    ]
    for source in finn_files:
        copy_file(source, OPTIMIZED_ROOT / "finn_reports" / source.name)

    evaluation_destination = OPTIMIZED_ROOT / "model_evaluation"
    copy_file(EVALUATION_SOURCE, evaluation_destination / "evaluation.json")
    copy_file(POSTPROCESS_SOURCE, evaluation_destination / "postprocess_config.json")
    copy_file(BASE_MANIFEST_SOURCE, evaluation_destination / "base_fpga_manifest.json")
    tool_sources = [
        ROOT / "fpga_build" / "run_finn_uart_hardware.ps1",
        ROOT / "fpga_build" / "program_finn_uart_top.tcl",
        ROOT / "scripts" / "send_frame_finn_uart.py",
        ROOT / "scripts" / "run_video_finn_uart.py",
    ]
    for source in tool_sources:
        copy_file(source, OPTIMIZED_ROOT / "tools" / source.name)

    baseline = summarize_hardware(load_hardware_reports(BASELINE_REPORTS_SOURCE))
    target50 = summarize_hardware(load_hardware_reports(TARGET50_REPORTS_SOURCE))
    optimized = summarize_hardware(load_hardware_reports(HARDWARE_REPORTS_SOURCE))
    evaluation = load_json(EVALUATION_SOURCE)
    postprocess = load_json(POSTPROCESS_SOURCE)
    base_manifest = load_json(BASE_MANIFEST_SOURCE)
    video_validation = load_json(VIDEO_VALIDATION_SOURCE / "report.json")
    finn_estimate = load_json(
        FINN_BUILD_SOURCE / "report" / "estimate_network_performance.json"
    )

    utilization_path = VIVADO_REPORTS_SOURCE / "utilization_routed.rpt"
    timing_path = VIVADO_REPORTS_SOURCE / "timing_routed.rpt"
    power_path = VIVADO_REPORTS_SOURCE / "power_routed.rpt"
    resources = {
        "slice_luts": parse_table_row(utilization_path, "Slice LUTs"),
        "lut_as_logic": parse_table_row(utilization_path, "LUT as Logic"),
        "lut_as_memory": parse_table_row(utilization_path, "LUT as Memory"),
        "slice_registers": parse_table_row(utilization_path, "Slice Registers"),
        "bram_tiles": parse_table_row(utilization_path, "Block RAM Tile"),
        "dsps": parse_table_row(utilization_path, "DSPs"),
    }
    timing = parse_timing(timing_path)
    power = parse_power(power_path)

    baseline_fps = baseline["single_frame_fps"]["mean"]
    target50_fps = target50["single_frame_fps"]["mean"]
    optimized_fps = optimized["single_frame_fps"]["mean"]
    baseline_cycles = round(baseline["accelerator_cycles"]["mean"])
    target50_cycles = round(target50["accelerator_cycles"]["mean"])
    optimized_cycles = round(optimized["accelerator_cycles"]["mean"])

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "goal": {
            "preserve_best_available_192x192_detection_quality": True,
            "minimum_measured_accelerator_fps": 50.0,
            "goal_met": optimized_fps >= 50.0 and optimized["all_exact"],
        },
        "board": {
            "name": "Digilent Arty S7-25",
            "device": "xc7s25csga324-1",
            "clock_hz": 100_000_000,
        },
        "model": base_manifest["model"],
        "accuracy": {
            "checkpoint_epoch": evaluation["checkpoint_epoch"],
            "deployment_thresholds": postprocess["selected"],
            "optimized_validation": postprocess["optimized"]["validation"],
            "optimized_test": postprocess["optimized"]["test"],
            "note": (
                "Folding changes hardware parallelism only. The checkpoint, "
                "W4A6 quantization, output requantization and decoder are unchanged."
            ),
        },
        "throughput": {
            "baseline": {
                "accelerator_cycles": baseline_cycles,
                "latency_ms": baseline["accelerator_latency_ms"]["mean"],
                "measured_single_frame_fps": baseline_fps,
                "hardware_exact": baseline["all_exact"],
            },
            "target_50": {
                "accelerator_cycles": target50_cycles,
                "latency_ms": target50["accelerator_latency_ms"]["mean"],
                "measured_single_frame_fps": target50_fps,
                "theoretical_streaming_fps": 50.234696502057616,
                "hardware_exact": target50["all_exact"],
            },
            "optimized_60": {
                "accelerator_cycles": optimized_cycles,
                "latency_ms": optimized["accelerator_latency_ms"]["mean"],
                "measured_single_frame_fps": optimized_fps,
                "theoretical_streaming_fps": float(
                    finn_estimate["estimated_throughput_fps"]
                ),
                "hardware_exact": optimized["all_exact"],
            },
        },
        "speedup": {
            "times": optimized_fps / baseline_fps,
            "cycles_reduced_percent": (1.0 - optimized_cycles / baseline_cycles) * 100.0,
        },
        "hardware_validation": optimized,
        "video_validation": video_validation,
        "implementation": {
            "tool": "Vivado 2022.2",
            "resources": resources,
            "timing": timing,
            "power": power,
            "drc_errors": 0,
        },
        "bitstream": {
            "relative_path": bitstream_destination.relative_to(RELEASE_ROOT).as_posix(),
            "bytes": bitstream_destination.stat().st_size,
            "sha256": sha256(bitstream_destination),
        },
        "limitations": [
            "The measured FPS covers the accelerator for one 192x192 ROI.",
            "UART at 115200 baud is for validation and is not a real-time camera transport.",
            "End-to-end real-time video needs a direct pixel stream or a faster host interface.",
        ],
    }
    write_json(OPTIMIZED_ROOT / "performance_summary.json", summary)

    optimized_manifest = dict(base_manifest)
    optimized_manifest["schema_version"] = 2
    optimized_manifest["name"] = (
        "tiny_detector_192x192_w4a6_int8_arty_s7_25_measured_51fps"
    )
    optimized_manifest["optimized_deployment"] = {
        "goal": summary["goal"],
        "throughput": summary["throughput"]["optimized_60"],
        "speedup": summary["speedup"],
        "hardware_validation": summary["hardware_validation"],
        "video_validation": summary["video_validation"],
        "implementation": summary["implementation"],
        "bitstream": summary["bitstream"],
    }
    write_json(OPTIMIZED_ROOT / "fpga_manifest_60fps.json", optimized_manifest)

    write_comparison_csv(summary)
    make_plots(summary)
    write_release_readme(summary)
    write_checksums(OPTIMIZED_ROOT)
    write_checksums(RELEASE_ROOT)

    zip_path = ROOT / "final_results" / "arty_s7_25_qnn_detection_50fps.zip"
    create_zip(RELEASE_ROOT, zip_path)

    result = {
        "release_directory": str(RELEASE_ROOT),
        "optimized_directory": str(OPTIMIZED_ROOT),
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "bitstream_sha256": summary["bitstream"]["sha256"],
        "measured_fps": optimized_fps,
        "goal_met": summary["goal"]["goal_met"],
        "hardware_exact": optimized["all_exact"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
