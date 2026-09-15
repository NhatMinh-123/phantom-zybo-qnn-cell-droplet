"""One-droplet QNN pipeline with scale-normalized candidate crops.

The classifier training set crops each annotated object with context and then
resizes it to 32x32. This entry point applies the same rule at inference:
candidate crop size is proportional to the connected-component size instead
of being a fixed 32-pixel window.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_microplastic_one_droplet_qnn as application


CONTEXT_SCALE = 2.4
MIN_SOURCE_SIDE = 8
MAX_SOURCE_SIDE = 48


def adaptive_candidate_classifier(
    gray_patch,
    aligned_previous,
    **kwargs,
):
    (
        candidates,
        blackhat,
        candidate_mask,
        blackhat_threshold,
        temporal_threshold,
    ) = application.ORIGINAL_CANDIDATE_FINDER(
        gray_patch,
        aligned_previous,
        **kwargs,
    )
    if not candidates:
        return (
            candidates,
            blackhat,
            candidate_mask,
            blackhat_threshold,
            temporal_threshold,
        )

    patches = []
    for candidate in candidates:
        source_side = int(
            round(max(candidate.width, candidate.height) * CONTEXT_SCALE)
        )
        source_side = int(
            np.clip(source_side, MIN_SOURCE_SIDE, MAX_SOURCE_SIDE)
        )
        crop = application.pipeline.centered_crop(
            gray_patch,
            candidate.x,
            candidate.y,
            source_side,
        )
        patches.append(
            cv2.resize(
                crop,
                (application.INPUT_SIZE, application.INPUT_SIZE),
                interpolation=(
                    cv2.INTER_AREA
                    if source_side >= application.INPUT_SIZE
                    else cv2.INTER_CUBIC
                ),
            )
        )

    batch = (
        torch.from_numpy(
            np.stack(patches).astype(np.float32)[:, None] / 255.0
        )
        .to(
            application.DEVICE,
            non_blocking=application.DEVICE.type == "cuda",
        )
    )
    application.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.sigmoid(
            application.MODEL(batch)
        ).flatten()
    application.synchronize()
    application.CLASSIFIER_CALL_MS.append(
        (time.perf_counter() - start) * 1000.0
    )
    scores = probabilities.detach().cpu().numpy()
    application.CLASSIFIER_PATCHES += len(candidates)

    accepted = []
    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)
        if candidate.score >= application.GATE_THRESHOLD:
            accepted.append(candidate)
    accepted.sort(key=lambda item: item.score, reverse=True)
    application.CLASSIFIER_ACCEPTED += len(accepted)
    return (
        accepted,
        blackhat,
        candidate_mask,
        blackhat_threshold,
        temporal_threshold,
    )


application.pipeline.find_particle_candidates = adaptive_candidate_classifier


if __name__ == "__main__":
    output = application.output_from_argv()
    print(
        f"Adaptive QNN classifier: {application.CHECKPOINT_PATH}; "
        f"device={application.DEVICE}; "
        f"gate={application.GATE_THRESHOLD:.3f}; "
        f"context={CONTEXT_SCALE:.1f}x"
    )
    application.pipeline.main()
    application.enrich_summary(output)
    if output is not None:
        summary_path = output / "summary.json"
        payload = application.json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        payload["pipeline"] = "one_droplet_hybrid_qnn_adaptive_w4a6_v1"
        payload["qnn_classifier"]["candidate_crop"] = {
            "mode": "component_adaptive",
            "context_scale": CONTEXT_SCALE,
            "minimum_source_side": MIN_SOURCE_SIDE,
            "maximum_source_side": MAX_SOURCE_SIDE,
            "network_input_size": application.INPUT_SIZE,
        }
        summary_path.write_text(
            application.json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
