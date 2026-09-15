"""Stable entry point for the one-droplet QNN video pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_microplastic_one_droplet_qnn as application


if __name__ == "__main__":
    output = application.output_from_argv()
    print(
        f"QNN classifier: {application.CHECKPOINT_PATH}; "
        f"device={application.DEVICE}; "
        f"gate={application.GATE_THRESHOLD:.3f}"
    )
    application.pipeline.main()
    application.enrich_summary(output)
