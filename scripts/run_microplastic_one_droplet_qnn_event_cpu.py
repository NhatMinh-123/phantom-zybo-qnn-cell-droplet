"""CPU event-gated QNN entry point for low-jitter PC video inference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


os.environ["MICROPLASTIC_QNN_DEVICE"] = "cpu"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_microplastic_one_droplet_qnn_event_gated as event_pipeline


def warm_up(iterations: int = 24) -> None:
    application = event_pipeline.application
    inputs = torch.zeros(
        4,
        1,
        application.INPUT_SIZE,
        application.INPUT_SIZE,
        device=application.DEVICE,
    )
    with torch.inference_mode():
        for _ in range(iterations):
            application.MODEL(inputs)
    application.CLASSIFIER_CALL_MS.clear()
    application.CLASSIFIER_PATCHES = 0
    application.CLASSIFIER_ACCEPTED = 0


if __name__ == "__main__":
    application = event_pipeline.application
    warm_up()
    output = application.output_from_argv()
    print(
        f"CPU event-gated QNN: {application.CHECKPOINT_PATH}; "
        f"gate={application.GATE_THRESHOLD:.3f}"
    )
    application.pipeline.main()
    application.enrich_summary(output)
    if output is not None:
        summary_path = output / "summary.json"
        payload = application.json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        payload["pipeline"] = "one_droplet_event_gated_qnn_cpu_w4a6_v1"
        payload["qnn_classifier"]["invocation_policy"] = (
            "one cached CPU inference after temporal min-hits gate"
        )
        payload["qnn_classifier"]["warmup_iterations"] = 24
        summary_path.write_text(
            application.json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
