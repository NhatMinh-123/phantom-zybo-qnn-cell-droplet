from pathlib import Path

import pytest

from finn.fix_hls_readmem_paths import rewrite_readmem_paths


def test_rewrite_readmem_paths_uses_windows_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    rom_path = project_root / "build" / "threshold.dat"
    rom_path.parent.mkdir(parents=True)
    rom_path.write_text("0011\n", encoding="ascii")

    rtl_root = tmp_path / "rtl"
    rtl_root.mkdir()
    verilog_path = rtl_root / "threshold_rom.v"
    verilog_path.write_text(
        '$readmemh("/workspace/fpga/build/threshold.dat", rom0);\n',
        encoding="ascii",
    )

    result = rewrite_readmem_paths(rtl_root, project_root=project_root)

    assert result.changed_files == 1
    assert result.changed_references == 1
    assert rom_path.as_posix() in verilog_path.read_text(encoding="ascii")


def test_rewrite_readmem_paths_rejects_missing_rom(tmp_path: Path) -> None:
    verilog_path = tmp_path / "threshold_rom.v"
    verilog_path.write_text(
        '$readmemh("/workspace/fpga/build/missing.dat", rom0);\n',
        encoding="ascii",
    )

    with pytest.raises(FileNotFoundError, match="missing.dat"):
        rewrite_readmem_paths(tmp_path, project_root=tmp_path / "project")
