"""Event-gated QNN verification after cheap temporal candidate tracking.

This ordering is optimized for the Arty S7-25:

    classical candidate -> temporal gate -> one QNN call per mature track

The QNN is no longer called for every blob in every frame. A track must first
survive the configured ``--particle-min-hits`` gate. Its best 32x32 patch is
then classified once and the probability is cached for the rest of the track.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_microplastic_one_droplet_qnn as application


BaseTracker = application.pipeline.TemporalParticleTracker


class EventGatedQNNTracker(BaseTracker):
    def _qnn_probability(self, track) -> float | None:
        value = getattr(track, "qnn_probability", None)
        return float(value) if value is not None else None

    def _track_confidence(self, track) -> float:
        qnn_probability = self._qnn_probability(track)
        if qnn_probability is None:
            return super()._track_confidence(track)
        persistence = min(
            track.hits / max(self.min_hits + 1, 1),
            1.0,
        )
        return float(
            np.clip(
                0.80 * qnn_probability + 0.20 * persistence,
                0,
                1,
            )
        )

    def is_confirmed(self, track) -> bool:
        qnn_probability = self._qnn_probability(track)
        return (
            track.hits >= self.min_hits
            and qnn_probability is not None
            and qnn_probability >= application.GATE_THRESHOLD
            and self._track_confidence(track) >= self.confidence_threshold
        )

    def _classify_new_mature_tracks(self) -> None:
        tracks = [
            track
            for track in self.tracks
            if track.hits >= self.min_hits
            and self._qnn_probability(track) is None
            and track.best_patch is not None
        ]
        if not tracks:
            return
        patches = np.stack([track.best_patch for track in tracks])
        batch = (
            torch.from_numpy(
                patches.astype(np.float32)[:, None] / 255.0
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
        application.CLASSIFIER_PATCHES += len(tracks)
        for track, probability in zip(tracks, scores):
            track.qnn_probability = float(probability)
            if probability >= application.GATE_THRESHOLD:
                application.CLASSIFIER_ACCEPTED += 1

    def update(self, candidates, gray_patch, frame_index):
        visible, finished = super().update(
            candidates,
            gray_patch,
            frame_index,
        )
        self._classify_new_mature_tracks()
        tracks_by_id = {
            track.track_id: track
            for track in self.tracks
        }
        for item in visible:
            track = tracks_by_id[item.track_id]
            item.confidence = self._track_confidence(track)
            item.confirmed = self.is_confirmed(track)
        return visible, finished


application.pipeline.find_particle_candidates = (
    application.ORIGINAL_CANDIDATE_FINDER
)
application.pipeline.TemporalParticleTracker = EventGatedQNNTracker


if __name__ == "__main__":
    output = application.output_from_argv()
    print(
        f"Event-gated QNN: {application.CHECKPOINT_PATH}; "
        f"device={application.DEVICE}; "
        f"gate={application.GATE_THRESHOLD:.3f}"
    )
    application.pipeline.main()
    application.enrich_summary(output)
    if output is not None:
        summary_path = output / "summary.json"
        payload = application.json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        payload["pipeline"] = "one_droplet_event_gated_qnn_w4a6_v1"
        payload["qnn_classifier"]["invocation_policy"] = (
            "one cached inference after temporal min-hits gate"
        )
        summary_path.write_text(
            application.json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
