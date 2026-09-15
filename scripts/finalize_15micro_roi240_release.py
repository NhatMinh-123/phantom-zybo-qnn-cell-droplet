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
    / "11_fpga_hardware_validated_sparse_uart_12m"
)
TARGET_RELEASE = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "13_fpga_roi240_validated_release"
)
VIDEO_RESULT = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "12_fpga_roi240_no_motion"
)
CANONICAL_MANIFEST = (
    ROOT
    / "exports"
    / "15micro_qnn_w4a6_192_v2"
    / "fpga_manifest_sparse_uart.json"
)
ROI_CONFIG = ROOT / "configs" / "15micro_qnn_roi240_center_v2.json"
LAUNCHER = ROOT / "fpga_build" / "run_15micro_fpga_roi240.ps1"
AB_REPORT_DIR = ROOT / "reports" / "15micro_roi_ab"


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
        VIDEO_RESULT / "fpga_sparse_realtime.mp4",
        VIDEO_RESULT / "report.json",
        CANONICAL_MANIFEST,
        ROI_CONFIG,
        LAUNCHER,
        AB_REPORT_DIR / "summary.json",
        AB_REPORT_DIR / "README.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing artifacts:\n" + "\n".join(missing))

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = read_json(CANONICAL_MANIFEST)
    roi_config = read_json(ROI_CONFIG)
    video = read_json(VIDEO_RESULT / "report.json")
    ab_summary = read_json(AB_REPORT_DIR / "summary.json")

    manifest["runtime"]["active_profile"] = "roi240_center_no_motion_v2"
    manifest["runtime"]["roi"] = roi_config["roi"]
    manifest["runtime"]["motion_compensation"] = False
    manifest["runtime"]["source_roi_area_reduction_percent_vs_roi256"] = 12.109375
    manifest["runtime"]["resource_note"] = (
        "Source ROI reduction does not change FPGA resources because the QNN input remains 192x192."
    )
    manifest.setdefault("hardware_validation", {})["roi240_runtime"] = {
        "status": "passed",
        "validated_at_utc": generated_at,
        "port": video["port"],
        "roi_config": str(ROI_CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "motion_compensation": False,
        "displayed_frames": video["displayed_frames"],
        "display_fps": video["display_fps"],
        "fpga_results": video["fpga_results"],
        "fpga_update_fps": video["fpga_update_fps"],
        "uart_paced_stream_fps": video["uart_paced_stream_fps"],
        "round_trip_ms": video["round_trip_ms"],
        "mean_display_lag_frames": video["motion_compensation"][
            "mean_display_lag_frames"
        ],
        "p95_display_lag_frames": video["motion_compensation"][
            "p95_display_lag_frames"
        ],
        "output_video": str(
            (VIDEO_RESULT / "fpga_sparse_realtime.mp4").relative_to(ROOT)
        ).replace("\\", "/"),
        "box_geometry": ab_summary["box_geometry"],
    }
    write_json(CANONICAL_MANIFEST, manifest)

    shutil.copytree(SOURCE_RELEASE, TARGET_RELEASE, dirs_exist_ok=True)
    copy_file(CANONICAL_MANIFEST, TARGET_RELEASE / "model" / CANONICAL_MANIFEST.name)
    copy_file(ROI_CONFIG, TARGET_RELEASE / "config" / ROI_CONFIG.name)
    copy_file(LAUNCHER, TARGET_RELEASE / "host" / LAUNCHER.name)
    copy_file(
        AB_REPORT_DIR / "summary.json",
        TARGET_RELEASE / "evaluation" / "roi240_ab_summary.json",
    )
    copy_file(
        AB_REPORT_DIR / "README.md",
        TARGET_RELEASE / "evaluation" / "ROI240_AB.md",
    )
    copy_file(
        VIDEO_RESULT / "report.json",
        TARGET_RELEASE / "hardware_roi240" / "video_report.json",
    )
    copy_file(
        VIDEO_RESULT / "fpga_sparse_realtime.mp4",
        TARGET_RELEASE / "hardware_roi240" / "fpga_sparse_realtime.mp4",
    )
    for image in sorted((VIDEO_RESULT / "inspection_frames").glob("frame_*.jpg")):
        copy_file(
            image,
            TARGET_RELEASE / "hardware_roi240" / "inspection_frames" / image.name,
        )

    readme = f"""# ROI 240 FPGA validation

Status: **PASS**

- Source ROI: `(568,350,240,240)` at 1280x800.
- Source ROI area reduction versus 256x256: `12.11%`.
- QNN input: `192x192`; FPGA LUT/BRAM/DSP usage is unchanged.
- Display: `{video['display_fps']:.2f} FPS` over `{video['displayed_frames']}` frames.
- FPGA detection updates: `{video['fpga_update_fps']:.2f}/s`.
- UART-paced FPGA stream: `{video['uart_paced_stream_fps']['mean']:.2f} FPS`.
- Mean round trip: `{video['round_trip_ms']['mean']:.2f} ms`.
- Display motion prediction: disabled.
- Static box center calibration: x=0, y=0; droplet width scale=1.275.

The smaller source ROI reduces background and host-side crop area. A real FPGA resource
reduction requires training a separate 160x160 QNN and rebuilding FINN/Vivado.
"""
    (TARGET_RELEASE / "ROI240_HARDWARE_VALIDATION.md").write_text(
        readme, encoding="utf-8"
    )

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
        "status": "hardware_validated_roi240_no_motion",
        "hardware_programmed": True,
        "hardware_uart_equivalence": "6/6 exact",
        "hardware_video_run": True,
        "active_runtime_profile": "roi240_center_no_motion_v2",
        "board": "Arty S7-25",
        "device": "xc7s25_0",
        "port": video["port"],
        "files": files,
    }
    write_json(TARGET_RELEASE / "release_manifest.json", release_manifest)

    failures = []
    for item in files:
        path = TARGET_RELEASE / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise RuntimeError("Checksum failure: " + ", ".join(failures))

    print(f"ROI240_RELEASE_PASS: {TARGET_RELEASE}")
    print(f"files={len(files)} checksum_failures={len(failures)}")


if __name__ == "__main__":
    main()
