#!/usr/bin/env python3
"""Add exact-preprocess hardware runners and refresh release checksums."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "final_results"
    / "15micro_pipeline_v1"
    / "09_fpga_ready_sparse_uart_12m"
)


EXTRAS = {
    "host/run_15micro_fpga.ps1": ROOT / "fpga_build" / "run_15micro_fpga.ps1",
    "host/run_15micro_fpga_video_sparse.py": ROOT
    / "scripts"
    / "run_15micro_fpga_video_sparse.py",
    "evaluation/host_contract_verification.json": ROOT
    / "reports"
    / "15micro_qnn_v1"
    / "fpga_requant_equivalence"
    / "host_contract_verification.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not RELEASE.is_dir():
        raise FileNotFoundError(RELEASE)
    for relative, source in EXTRAS.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = RELEASE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    runbook = """# Chay QNN 15 um tren Arty S7-25

Trang thai hien tai: bitstream da build, chua program board.

Tien xu ly runtime duoc khoa giong luc train:

1. Cat dung ROI x=560, y=342, w=256, h=256 tren frame 1280x800.
2. Chuyen ROI sang grayscale.
3. Resize PIL bilinear mot lan xuong 192x192.
4. Luong tu UINT8 va gui vao FINN core.

Dry-run, khong nap board:

```powershell
powershell -ExecutionPolicy Bypass -File E:\\fpga\\fpga_build\\run_15micro_fpga.ps1 -DryRun
```

Chi sau khi da cam board va xac nhan COM12:

```powershell
powershell -ExecutionPolicy Bypass -File E:\\fpga\\fpga_build\\run_15micro_fpga.ps1 -Port COM12 -Program -DurationSec 10
```

Lenh tren program bitstream, doi chieu 6 anh test voi checkpoint, sau do moi
chay video ROI qua FPGA. Ket qua board se duoc ghi vao thu muc buoc 10; ket qua
PC trong release khong duoc coi la ket qua FPGA.
"""
    (RELEASE / "RUN_HARDWARE.md").write_text(runbook, encoding="ascii")

    files = []
    for path in sorted(RELEASE.rglob("*")):
        if not path.is_file() or path.name == "release_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(RELEASE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_board_programming",
        "hardware_programmed": False,
        "hardware_video_run": False,
        "preprocessing": "fixed ROI -> grayscale -> one PIL bilinear resize -> UINT8",
        "files": files,
    }
    (RELEASE / "release_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Refreshed release: {len(files)} files under {RELEASE}")


if __name__ == "__main__":
    main()
