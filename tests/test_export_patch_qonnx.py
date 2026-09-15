from pathlib import Path

import numpy as np
from PIL import Image

from qnn.export_patch_qonnx import load_patch, verification_patches


def test_load_patch_returns_normalized_nchw(tmp_path: Path) -> None:
    path = tmp_path / "patch.png"
    Image.fromarray(np.full((16, 16), 128, dtype=np.uint8)).save(path)

    patch = load_patch(path, 32, "raw")

    assert patch.shape == (1, 1, 32, 32)
    assert patch.dtype == np.float32
    assert np.allclose(patch, np.float32(128 / 255))


def test_verification_patches_balances_classes(tmp_path: Path) -> None:
    for class_name in ("background", "particle"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(3):
            Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(
                class_dir / f"{index}.png"
            )

    paths = verification_patches(tmp_path, limit_per_class=2)

    assert len(paths) == 4
    assert sum("background" in path.parts for path in paths) == 2
    assert sum("particle" in path.parts for path in paths) == 2
