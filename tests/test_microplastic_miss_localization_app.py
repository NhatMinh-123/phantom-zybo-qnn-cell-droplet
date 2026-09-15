import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_microplastic_miss_localization_app.py"
SPEC = importlib.util.spec_from_file_location("miss_localization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_centered_crop_preserves_requested_center() -> None:
    image = np.zeros((80, 90), dtype=np.uint8)
    image[37, 43] = 255
    crop = MODULE.centered_crop(image, 43, 37, 32)
    assert crop.shape == (32, 32)
    assert crop[16, 16] == 255


def test_localization_app_exports_click_points_and_32_pixel_boxes() -> None:
    html = MODULE.HTML
    assert "microplastic_missed_particle_locations.csv" in html
    assert "points_json" in html
    assert "strokeRect(x-16,y-16,32,32)" in html
    assert "Confirm boxes" in html
    assert "t0 - click every particle" in html
