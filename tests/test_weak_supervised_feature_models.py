from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.train_weak_supervised_feature_models import (
    create_weak_labels,
    metrics_at_threshold,
    split_groups,
)


class WeakSupervisedFeatureModelTests(unittest.TestCase):
    def test_weak_labels_keep_ambiguous_tracks_out_of_training(self) -> None:
        tracks = pd.DataFrame(
            [
                {
                    "video_slug": "video_a",
                    "droplet_sequence": 1,
                    "hits": 4,
                    "best_proposal_score": 0.70,
                    "radial_distance_norm_mean": 0.30,
                    "local_contrast_gray_mean": 9.0,
                    "pixel_area_mean": 12.0,
                },
                {
                    "video_slug": "video_a",
                    "droplet_sequence": 2,
                    "hits": 1,
                    "best_proposal_score": 0.20,
                    "radial_distance_norm_mean": 0.40,
                    "local_contrast_gray_mean": 3.0,
                    "pixel_area_mean": 3.0,
                },
                {
                    "video_slug": "video_a",
                    "droplet_sequence": 3,
                    "hits": 2,
                    "best_proposal_score": 0.42,
                    "radial_distance_norm_mean": 0.40,
                    "local_contrast_gray_mean": 7.0,
                    "pixel_area_mean": 8.0,
                },
            ]
        )

        labeled = create_weak_labels(
            tracks,
            minimum_positive_hits=3,
            positive_score=0.45,
            negative_score=0.38,
        )

        self.assertEqual(labeled["weak_label"].tolist(), [1, 0, -1])
        self.assertEqual(
            labeled["weak_label_name"].tolist(),
            ["particle", "background", "ambiguous"],
        )
        self.assertTrue(
            labeled["split_group"].str.startswith("video_a_sequence_").all()
        )

    def test_group_split_has_no_sequence_leakage(self) -> None:
        rows = []
        for sequence in range(1, 61):
            label = sequence % 2
            for track_index in range(3):
                rows.append(
                    {
                        "video_slug": f"video_{sequence % 4}",
                        "droplet_sequence": sequence,
                        "weak_label": label,
                        "track_index": track_index,
                    }
                )
        tracks = pd.DataFrame(rows)
        tracks["split_group"] = (
            tracks["video_slug"]
            + "_sequence_"
            + tracks["droplet_sequence"].astype(str)
        )

        mapping = split_groups(tracks, seed=42)
        tracks["split"] = tracks["split_group"].map(mapping)

        self.assertEqual(
            set(tracks["split"].unique()),
            {"train", "validation", "test"},
        )
        self.assertTrue(
            (tracks.groupby("split_group")["split"].nunique() == 1).all()
        )
        for split in ("train", "validation", "test"):
            self.assertEqual(
                set(tracks.loc[tracks["split"] == split, "weak_label"]),
                {0, 1},
            )

    def test_threshold_metrics_match_known_confusion_matrix(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        probabilities = np.array([0.10, 0.70, 0.80, 0.20])

        metrics = metrics_at_threshold(labels, probabilities, 0.50)

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertAlmostEqual(float(metrics["precision"]), 0.5)
        self.assertAlmostEqual(float(metrics["recall"]), 0.5)
        self.assertAlmostEqual(float(metrics["f1"]), 0.5)


if __name__ == "__main__":
    unittest.main()
