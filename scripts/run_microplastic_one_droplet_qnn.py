"""Run the FPGA-fast one-droplet pipeline with the tiny QNN classifier.

Environment overrides:

    MICROPLASTIC_QNN_CHECKPOINT
    MICROPLASTIC_QNN_DEVICE          (auto, cpu, cuda)
    MICROPLASTIC_QNN_GATE_THRESHOLD (defaults to checkpoint threshold)

The existing command-line interface is inherited from
run_microplastic_one_droplet_hybrid.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

import run_microplastic_one_droplet_hybrid as pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "models" / "qnn_microplastic_patch32_w4a6_v1" / "best.pt"
)


def resolve_device() -> torch.device:
    requested = os.environ.get("MICROPLASTIC_QNN_DEVICE", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MICROPLASTIC_QNN_DEVICE requests unavailable CUDA")
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CHECKPOINT_PATH = Path(
    os.environ.get("MICROPLASTIC_QNN_CHECKPOINT", DEFAULT_CHECKPOINT)
).expanduser().resolve()
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

DEVICE = resolve_device()
CHECKPOINT = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

from qnn.patch_classifier import (
    TinyQuantPatchClassifier,
    patch_config_from_dict,
)
from qnn.patch_preprocess import preprocess_patch


MODEL = TinyQuantPatchClassifier(
    patch_config_from_dict(CHECKPOINT["config"])
).to(DEVICE)
MODEL.load_state_dict(CHECKPOINT["model_state"])
MODEL.eval()
INPUT_SIZE = MODEL.config.input_size
GATE_THRESHOLD = float(
    os.environ.get(
        "MICROPLASTIC_QNN_GATE_THRESHOLD",
        CHECKPOINT.get("threshold", 0.5),
    )
)
if not 0 <= GATE_THRESHOLD <= 1:
    raise ValueError("MICROPLASTIC_QNN_GATE_THRESHOLD must be in [0, 1]")

ORIGINAL_CANDIDATE_FINDER = pipeline.find_particle_candidates
CLASSIFIER_CALL_MS: list[float] = []
CLASSIFIER_PATCHES = 0
CLASSIFIER_ACCEPTED = 0


def synchronize() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def classify_candidate_patches(
    gray_patch,
    aligned_previous,
    **kwargs,
):
    global CLASSIFIER_PATCHES, CLASSIFIER_ACCEPTED
    (
        candidates,
        blackhat,
        candidate_mask,
        blackhat_threshold,
        temporal_threshold,
    ) = ORIGINAL_CANDIDATE_FINDER(
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

    patches = [
        pipeline.centered_crop(
            gray_patch,
            candidate.x,
            candidate.y,
            INPUT_SIZE,
        )
        for candidate in candidates
    ]
    patches = [
        preprocess_patch(patch, MODEL.config.input_transform)
        for patch in patches
    ]
    batch = (
        torch.from_numpy(
            np.stack(patches).astype(np.float32)[:, None] / 255.0
        )
        .to(DEVICE, non_blocking=DEVICE.type == "cuda")
    )
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.sigmoid(MODEL(batch)).flatten()
    synchronize()
    CLASSIFIER_CALL_MS.append((time.perf_counter() - start) * 1000.0)
    scores = probabilities.detach().cpu().numpy()
    CLASSIFIER_PATCHES += len(candidates)

    accepted = []
    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)
        if candidate.score >= GATE_THRESHOLD:
            accepted.append(candidate)
    accepted.sort(key=lambda item: item.score, reverse=True)
    CLASSIFIER_ACCEPTED += len(accepted)
    return (
        accepted,
        blackhat,
        candidate_mask,
        blackhat_threshold,
        temporal_threshold,
    )


def use_previous_patch_without_phase_alignment(
    previous,
    current,
    hanning_window,
):
    del current, hanning_window
    return previous, (0.0, 0.0), 0.0


def output_from_argv() -> Path | None:
    try:
        index = sys.argv.index("--output")
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[index + 1]).expanduser().resolve()


def enrich_summary(output: Path | None) -> None:
    if output is None:
        return
    summary_path = output / "summary.json"
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    model_summary_path = CHECKPOINT_PATH.parent / "summary.json"
    model_summary = (
        json.loads(model_summary_path.read_text(encoding="utf-8"))
        if model_summary_path.exists()
        else {}
    )
    payload["pipeline"] = "one_droplet_hybrid_qnn_w4a6_v1"
    payload["accuracy_status"] = (
        "Patch-level QNN metrics are measured on the grouped bootstrap test "
        "set. End-to-end video precision/recall still requires manual review "
        "of candidate tracks from independent videos."
    )
    payload["qnn_classifier"] = {
        "checkpoint": str(CHECKPOINT_PATH),
        "sha256": sha256_file(CHECKPOINT_PATH),
        "device": str(DEVICE),
        "config": MODEL.config.to_dict(),
        "gate_threshold": GATE_THRESHOLD,
        "selected_validation_metrics": CHECKPOINT.get(
            "validation_metrics",
            {},
        ),
        "bootstrap_test_metrics": model_summary.get("test", {}),
        "candidate_patches_evaluated": CLASSIFIER_PATCHES,
        "candidate_patches_accepted": CLASSIFIER_ACCEPTED,
        "candidate_acceptance_ratio": (
            CLASSIFIER_ACCEPTED / max(CLASSIFIER_PATCHES, 1)
        ),
        "calls": len(CLASSIFIER_CALL_MS),
        "mean_call_ms": (
            statistics.fmean(CLASSIFIER_CALL_MS)
            if CLASSIFIER_CALL_MS
            else 0.0
        ),
        "p95_call_ms": (
            float(np.percentile(CLASSIFIER_CALL_MS, 95))
            if CLASSIFIER_CALL_MS
            else 0.0
        ),
    }
    payload["artifacts"]["qnn_checkpoint"] = str(CHECKPOINT_PATH)
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


pipeline.align_previous_patch = use_previous_patch_without_phase_alignment
pipeline.find_particle_candidates = classify_candidate_patches


if __name__ == "__main__":
    output = output_from_argv()
    print(
        f"QNN classifier: {CHECKPOINT_PATH}; device={DEVICE}; "
        f"gate={GATE_THRESHOLD:.3f}"
    )
    pipeline.main()
    enrich_summary(output)
