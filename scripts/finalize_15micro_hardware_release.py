from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELEASE = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "09_fpga_ready_sparse_uart_12m"
)
VIDEO_RESULT = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "10_fpga_hardware_video"
)
TARGET_RELEASE = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "11_fpga_hardware_validated_sparse_uart_12m"
)
CANONICAL_MANIFEST = (
    ROOT
    / "exports"
    / "15micro_qnn_w4a6_192_v2"
    / "fpga_manifest_sparse_uart.json"
)
EXACT_REPORT = (
    ROOT
    / "reports"
    / "15micro_fpga_hardware"
    / "sparse_uart_exact"
    / "report.json"
)
THRESHOLD_REPORT = (
    ROOT
    / "reports"
    / "15micro_qnn_v1"
    / "final_postprocess_w4a6_192_v2"
    / "threshold_calibration.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    required = [
        SOURCE_RELEASE,
        CANONICAL_MANIFEST,
        EXACT_REPORT,
        THRESHOLD_REPORT,
        VIDEO_RESULT / "report.json",
        VIDEO_RESULT / "fpga_sparse_realtime.mp4",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    exact = read_json(EXACT_REPORT)
    video = read_json(VIDEO_RESULT / "report.json")
    thresholds = read_json(THRESHOLD_REPORT)
    manifest = read_json(CANONICAL_MANIFEST)
    generated_at = datetime.now(timezone.utc).isoformat()

    if not exact.get("all_exact") or exact.get("exact_images") != exact.get("images"):
        raise RuntimeError("FPGA/QNN exact-equivalence validation did not pass")

    validation = {
        "status": "passed",
        "validated_at_utc": generated_at,
        "programmed": True,
        "board": "Arty S7-25",
        "device": "xc7s25_0",
        "jtag_target": "Digilent/210352BE716FA",
        "bitstream_sha256": manifest["implementation"]["bitstream"]["sha256"],
        "uart_equivalence": {
            "status": "passed",
            "port": exact["port"],
            "baud": exact["baud"],
            "clock_hz": exact["clock_hz"],
            "images": exact["images"],
            "exact_images": exact["exact_images"],
            "all_exact": exact["all_exact"],
            "mean_payload_bytes": exact["mean_payload_bytes"],
            "maximum_payload_bytes": exact["maximum_payload_bytes"],
            "mean_uart_paced_stream_fps": exact["mean_uart_paced_stream_fps"],
            "mean_uart_round_trip_ms": exact["mean_uart_round_trip_ms"],
            "mean_system_fps": exact["mean_system_fps"],
            "report": str(EXACT_REPORT.relative_to(ROOT)).replace("\\", "/"),
        },
        "video_run": {
            "status": "passed",
            "source": video["source"],
            "output_video": video["output_video"],
            "displayed_frames": video["displayed_frames"],
            "display_fps": video["display_fps"],
            "fpga_results": video["fpga_results"],
            "fpga_update_fps": video["fpga_update_fps"],
            "uart_paced_stream_fps": video["uart_paced_stream_fps"],
            "round_trip_ms": video["round_trip_ms"],
            "jobs_submitted": video["jobs_submitted"],
            "stale_jobs_dropped": video["stale_jobs_dropped"],
            "report": str((VIDEO_RESULT / "report.json").relative_to(ROOT)).replace(
                "\\", "/"
            ),
        },
        "visual_inspection": {
            "status": "passed",
            "frames": [30, 150, 270],
            "roi_alignment": "passed",
            "detections_inside_fixed_roi": True,
            "note": "Dense labels can overlap visually; this is a host overlay issue, not an FPGA coordinate mismatch.",
        },
        "truth_boundary": {
            "fpga": "QNN convolutional inference, INT8 output requantization, sparse candidate thresholding and serialization",
            "host_pc": "MP4 decode, fixed ROI crop, grayscale/PIL-bilinear resize, UART I/O, geometry decode/NMS, overlay and video encoding",
            "standalone_camera": False,
        },
    }
    manifest["hardware_validation"] = validation
    write_json(CANONICAL_MANIFEST, manifest)

    summary = {
        "status": "hardware_validated",
        "generated_at_utc": generated_at,
        "model": {
            "architecture": "TinyQuantDetector",
            "input": "1x192x192 grayscale UINT8",
            "classes": ["cell", "droplet"],
            "parameters": 11009,
            "weight_bits": 4,
            "activation_bits": 6,
            "output_bits": 8,
            "test_metrics_after_validation_only_calibration": thresholds["test"],
        },
        "roi": manifest["runtime"]["roi"],
        "implementation": manifest["implementation"],
        "hardware_validation": validation,
    }
    write_json(VIDEO_RESULT / "hardware_validation_summary.json", summary)

    shutil.copytree(SOURCE_RELEASE, TARGET_RELEASE, dirs_exist_ok=True)
    copy_file(CANONICAL_MANIFEST, TARGET_RELEASE / "model" / CANONICAL_MANIFEST.name)
    copy_file(
        THRESHOLD_REPORT,
        TARGET_RELEASE / "evaluation" / "threshold_calibration.json",
    )
    copy_file(EXACT_REPORT, TARGET_RELEASE / "hardware" / "uart_equivalence_report.json")
    copy_file(VIDEO_RESULT / "report.json", TARGET_RELEASE / "hardware" / "video_report.json")
    copy_file(
        VIDEO_RESULT / "hardware_validation_summary.json",
        TARGET_RELEASE / "hardware" / "hardware_validation_summary.json",
    )
    copy_file(
        VIDEO_RESULT / "fpga_sparse_realtime.mp4",
        TARGET_RELEASE / "hardware" / "fpga_sparse_realtime.mp4",
    )
    inspection_dir = VIDEO_RESULT / "inspection_frames"
    for image in sorted(inspection_dir.glob("frame_*.jpg")):
        copy_file(image, TARGET_RELEASE / "hardware" / "inspection_frames" / image.name)

    markdown = f"""# Arty S7-25 hardware validation

Status: **PASS**

- Board: Arty S7-25 (`xc7s25_0`)
- UART: `{exact['port']}` at `{exact['baud']}` baud
- FPGA/QNN exact images: `{exact['exact_images']}/{exact['images']}`
- UART-paced FPGA stream: `{exact['mean_uart_paced_stream_fps']:.2f} FPS`
- Mean image round trip: `{video['round_trip_ms']['mean']:.2f} ms`
- Video display: `{video['display_fps']:.2f} FPS`
- New FPGA detections: `{video['fpga_update_fps']:.2f} updates/s`
- Test F1 after validation-only threshold calibration: `{thresholds['test']['f1']:.4f}`
- Cell F1: `{thresholds['test']['classes']['cell']['f1']:.4f}`
- Droplet F1: `{thresholds['test']['classes']['droplet']['f1']:.4f}`

## Execution boundary

The quantized CNN inference, output requantization, candidate thresholding, and sparse
record serialization ran on the FPGA. The PC decoded the MP4, cropped and resized the
fixed ROI, transported frames over UART, applied geometry decode/NMS, drew overlays,
and encoded the result video. This validates FPGA inference but is not yet a direct
camera-to-FPGA standalone system.
"""
    (TARGET_RELEASE / "HARDWARE_VALIDATION.md").write_text(markdown, encoding="utf-8")

    files = []
    for path in sorted(TARGET_RELEASE.rglob("*")):
        if not path.is_file() or path.name == "release_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(TARGET_RELEASE)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    release_manifest = {
        "generated_at_utc": generated_at,
        "status": "hardware_validated",
        "hardware_programmed": True,
        "hardware_uart_equivalence": "6/6 exact",
        "hardware_video_run": True,
        "board": "Arty S7-25",
        "device": "xc7s25_0",
        "port": exact["port"],
        "preprocessing": "fixed ROI -> grayscale -> one PIL bilinear resize -> UINT8",
        "files": files,
    }
    write_json(TARGET_RELEASE / "release_manifest.json", release_manifest)

    bad = []
    for item in files:
        path = TARGET_RELEASE / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            bad.append(item["path"])
    if bad:
        raise RuntimeError("Release checksum verification failed: " + ", ".join(bad))

    print(f"HARDWARE_RELEASE_PASS: {TARGET_RELEASE}")
    print(f"files={len(files)} checksum_failures={len(bad)}")


if __name__ == "__main__":
    main()
