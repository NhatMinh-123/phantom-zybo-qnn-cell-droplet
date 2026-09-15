#!/usr/bin/env python3
"""Small-ROI runner using a max-history temporal background."""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

from scripts import run_feature_ml_small_roi_motion_test as v1
from scripts.feature_ml_motion_candidates import (
    extract_motion_fused_candidates,
)


class MaxHistoryMotionExtractor(v1.LaggedMotionExtractor):
    def __init__(self, *, lag: int, dark_sigma: float) -> None:
        super().__init__(lag=lag, dark_sigma=dark_sigma)
        self.history: deque[np.ndarray] = deque(maxlen=self.lag)

    def __call__(
        self,
        gray_patch: np.ndarray,
        previous_patch: np.ndarray | None,
        **kwargs,
    ):
        if previous_patch is None:
            self.history.clear()
        reference = (
            np.max(np.stack(self.history), axis=0).astype(np.uint8)
            if len(self.history) == self.lag
            else None
        )
        candidates, diagnostics = extract_motion_fused_candidates(
            gray_patch,
            reference,
            dark_sigma=self.dark_sigma,
            **kwargs,
        )
        self.history.append(gray_patch.copy())
        for candidate in candidates:
            source = str(candidate.row.get("proposal_source", "unknown"))
            self.source_counts[source] += 1
            self.frame_counts[int(candidate.row["frame_index"])] += 1
        return candidates, diagnostics


def enrich_summary(
    output: Path,
    *,
    args,
    extractor,
) -> None:
    v1.update_summary_original(
        output,
        args=args,
        extractor=extractor,
    )
    summary_path = output.resolve() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["proposal_generation"]["temporal_reference"] = (
        "per-pixel maximum of the previous four registered droplet patches"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    if "--motion-lag" not in sys.argv:
        sys.argv.extend(["--motion-lag", "4"])
    v1.LaggedMotionExtractor = MaxHistoryMotionExtractor
    v1.update_summary_original = v1.update_summary
    v1.update_summary = enrich_summary
    v1.main()


if __name__ == "__main__":
    main()
