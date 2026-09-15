#!/usr/bin/env python3
"""Package the 15 um QNN build without claiming pending board results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "09_fpga_ready_sparse_uart_12m"
)


FILES = {
    "bitstream/cell_droplet_15micro_finn_uart_sparse_stream_top.bit": ROOT
    / "fpga_build"
    / "cell_droplet_15micro_finn_uart_sparse_stream_12m"
    / "cell_droplet_15micro_finn_uart_sparse_stream_top.bit",
    "model/best_fixed_f1.pt": ROOT
    / "models"
    / "15micro"
    / "qnn_w4a6_192_v2_from_w8a8"
    / "best_fixed_f1.pt",
    "model/tiny_detector_192x192_w4a6_rawhead.onnx": ROOT
    / "exports"
    / "15micro_qnn_w4a6_192_v2"
    / "tiny_detector_192x192_w4a6_rawhead.onnx",
    "model/tiny_detector_192x192_w4a6_rawhead.json": ROOT
    / "exports"
    / "15micro_qnn_w4a6_192_v2"
    / "tiny_detector_192x192_w4a6_rawhead.json",
    "model/fpga_manifest_sparse_uart.json": ROOT
    / "exports"
    / "15micro_qnn_w4a6_192_v2"
    / "fpga_manifest_sparse_uart.json",
    "config/15micro_yolo11n_roi256_v1.json": ROOT
    / "configs"
    / "15micro_yolo11n_roi256_v1.json",
    "config/postprocess_config.json": ROOT
    / "reports"
    / "15micro_qnn_v1"
    / "final_postprocess_w4a6_192_v2"
    / "postprocess_config.json",
    "evaluation/qnn_evaluation.json": ROOT
    / "reports"
    / "15micro_qnn_v1"
    / "evaluation_w4a6_192_v2_bestfixed"
    / "evaluation.json",
    "evaluation/fpga_requant_equivalence.json": ROOT
    / "reports"
    / "15micro_qnn_v1"
    / "fpga_requant_equivalence"
    / "verification.json",
    "evaluation/pc_qnn_video_report.json": ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "08_pc_qnn_w4a6_roi256_complete"
    / "report.json",
    "finn/estimate_summary.md": ROOT
    / "finn_build"
    / "arty_s7_25_15micro_w4a6_rawhead_estimate"
    / "build_50fps"
    / "report"
    / "estimate_summary.md",
    "finn/synthesis_summary.json": ROOT
    / "finn_build"
    / "windows_hls_bridge_15micro_w4a6_rawhead_50fps"
    / "stitched_windows"
    / "ooc_reports"
    / "synthesis_summary.json",
    "vivado/timing_routed.rpt": ROOT
    / "fpga_build"
    / "cell_droplet_15micro_finn_uart_sparse_stream_12m"
    / "reports"
    / "timing_routed.rpt",
    "vivado/utilization_routed.rpt": ROOT
    / "fpga_build"
    / "cell_droplet_15micro_finn_uart_sparse_stream_12m"
    / "reports"
    / "utilization_routed.rpt",
    "vivado/power_routed.rpt": ROOT
    / "fpga_build"
    / "cell_droplet_15micro_finn_uart_sparse_stream_12m"
    / "reports"
    / "power_routed.rpt",
    "vivado/drc_routed.rpt": ROOT
    / "fpga_build"
    / "cell_droplet_15micro_finn_uart_sparse_stream_12m"
    / "reports"
    / "drc_routed.rpt",
    "rtl/hardware_requant_config.json": ROOT
    / "fpga_rtl"
    / "generated_15micro"
    / "hardware_requant_config.json",
    "rtl/detector_output_thresholds_15micro_pkg.vhd": ROOT
    / "fpga_rtl"
    / "generated_15micro"
    / "detector_output_thresholds_15micro_pkg.vhd",
    "rtl/detector_output_requantizer_15micro.vhd": ROOT
    / "fpga_rtl"
    / "generated_15micro"
    / "detector_output_requantizer_15micro.vhd",
    "rtl/cell_droplet_15micro_finn_uart_sparse_stream_top.vhd": ROOT
    / "fpga_rtl"
    / "generated_15micro"
    / "cell_droplet_15micro_finn_uart_sparse_stream_top.vhd",
    "host/run_15micro_fpga_pipeline.ps1": ROOT
    / "fpga_build"
    / "run_15micro_fpga_pipeline.ps1",
    "host/send_frame_finn_uart_sparse.py": ROOT
    / "scripts"
    / "send_frame_finn_uart_sparse.py",
    "host/run_video_finn_uart_sparse_realtime.py": ROOT
    / "scripts"
    / "run_video_finn_uart_sparse_realtime.py",
}


VIDEOS = {
    "video_pc/pc_qnn_roi256_pre_fpga.mp4": ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "08_pc_qnn_w4a6_roi256_complete"
    / "pc_qnn_roi256_pre_fpga.mp4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(relative: str, source: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = OUTPUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def link_video(relative: str, source: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = OUTPUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    packaged = [copy_file(relative, source) for relative, source in FILES.items()]
    packaged.extend(link_video(relative, source) for relative, source in VIDEOS.items())

    manifest_path = OUTPUT / "model" / "fpga_manifest_sparse_uart.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    implementation = manifest["implementation"]
    evaluation = manifest["reference_evaluation"]["test"]
    roi = manifest["runtime"]["roi"]
    pc_video = manifest["runtime"]["pc_pre_fpga_video"]
    estimate = manifest["synthesis"]["finn_estimate"]
    estimated_108 = float(estimate["estimated_throughput_fps"]) * 1.08

    readme = f"""# 15 um QNN Arty S7-25 release

## Truth boundary

- The bitstream is built and passes routed timing/DRC checks.
- The generated VHDL requantizer matches the QNN checkpoint exactly on 12 test images.
- Hardware programming, UART equivalence, and FPGA video execution are still PENDING.
- The video under `video_pc` is PC GPU QNN output, not FPGA output.

## Fixed ROI and model

- Source reference: 1280x800.
- ROI: x={roi['x']}, y={roi['y']}, width={roi['width']}, height={roi['height']}.
- ROI fraction: 6.4% of full-frame pixels.
- QNN input: 192x192 grayscale.
- Quantization: W4A6, INT8 input/output.
- Test F1 before final postprocess: {evaluation['f1']:.4f}.
- Final calibrated test F1: 0.7214 overall; cell 0.6169; droplet 0.8353.

## PC video benchmark

- PC inference median: {pc_video['latency_ms']['inference_median']:.3f} ms.
- PC inference-only rate: {1000.0 / pc_video['latency_ms']['inference_median']:.2f} FPS.
- PC pipeline excluding video write: {1000.0 / pc_video['latency_ms']['pipeline_excluding_write_mean']:.2f} FPS.
- Export throughput including MP4 write: {pc_video['export_throughput_fps_including_video_write']:.2f} FPS.

## FINN and Vivado

- FINN estimated streaming throughput at 108 MHz: {estimated_108:.2f} FPS (estimate only).
- Routed clock: {implementation['timing']['frequency_mhz']:.0f} MHz.
- Routed WNS/WHS: {implementation['timing']['wns_ns']:+.3f}/{implementation['timing']['whs_ns']:+.3f} ns.
- LUT: {implementation['utilization']['slice_luts']['used']}/{implementation['utilization']['slice_luts']['available']} ({implementation['utilization']['slice_luts']['utilization_percent']:.2f}%).
- BRAM: {implementation['utilization']['bram_tiles']['used']}/{implementation['utilization']['bram_tiles']['available']} ({implementation['utilization']['bram_tiles']['utilization_percent']:.2f}%).
- DSP: {implementation['utilization']['dsps']['used']}/{implementation['utilization']['dsps']['available']} ({implementation['utilization']['dsps']['utilization_percent']:.2f}%).
- DRC errors: {implementation['drc']['errors']}.
- Bitstream SHA-256: `{implementation['bitstream']['sha256']}`.

## Before board programming

Dry-run only; this does not program hardware:

```powershell
powershell -ExecutionPolicy Bypass -File E:\\fpga\\fpga_build\\run_15micro_fpga_pipeline.ps1 -DryRun
```

After the board is connected and the user explicitly confirms, use `-Program`.
The script first checks six labeled images against the checkpoint, then runs the fixed-ROI video through the Arty S7-25 and writes hardware-only results to `final_results/15micro_pipeline_v1/10_fpga_hardware_video`.
"""
    readme_path = OUTPUT / "README.md"
    readme_path.write_text(readme, encoding="ascii")
    packaged.append(readme_path)

    release = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_board_programming",
        "hardware_programmed": False,
        "hardware_video_run": False,
        "files": [
            {
                "path": str(path.relative_to(OUTPUT).as_posix()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(packaged)
        ],
    }
    release_path = OUTPUT / "release_manifest.json"
    release_path.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged {len(packaged)} files under {OUTPUT}")
    print(f"Release manifest: {release_path}")


if __name__ == "__main__":
    main()
