#!/usr/bin/env python3
"""Direct-entry wrapper for the final small-ROI feature-ML video runner."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_feature_ml_small_roi_motion_v2 import main  # noqa: E402


if __name__ == "__main__":
    main()
