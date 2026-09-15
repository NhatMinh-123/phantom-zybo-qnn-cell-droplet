#!/usr/bin/env python3
"""Train feature-based ML models from conservative image-processing labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURES = [
    "bbox_width",
    "bbox_height",
    "pixel_area",
    "contour_area",
    "perimeter",
    "equivalent_diameter",
    "aspect_ratio",
    "circularity",
    "solidity",
    "extent",
    "object_mean_gray",
    "object_std_gray",
    "local_background_mean_gray",
    "local_contrast_gray",
    "blackhat_mean",
    "blackhat_max",
    "blackhat_ratio",
    "temporal_mean",
    "temporal_max",
    "temporal_ratio",
    "gradient_mean",
    "gradient_max",
    "local_entropy",
    "local_laplacian_variance",
    "relative_x",
    "relative_y",
    "radial_distance_norm",
    "droplet_radius_px",
    "core_radius_px",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--minimum-positive-hits", type=int, default=3)
    parser.add_argument("--positive-score", type=float, default=0.45)
    parser.add_argument("--negative-score", type=float, default=0.38)
    return parser.parse_args()


def prepare_candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["blackhat_ratio"] = prepared["blackhat_mean"] / np.maximum(
        prepared["blackhat_threshold"],
        1.0,
    )
    prepared["temporal_ratio"] = prepared["temporal_mean"] / np.maximum(
        prepared["temporal_threshold"],
        1.0,
    )
    return prepared


def create_weak_labels(
    tracks: pd.DataFrame,
    *,
    minimum_positive_hits: int,
    positive_score: float,
    negative_score: float,
) -> pd.DataFrame:
    labeled = tracks.copy()
    positive = (
        (labeled["hits"] >= minimum_positive_hits)
        & (labeled["best_proposal_score"] >= positive_score)
        & (labeled["radial_distance_norm_mean"] <= 0.90)
    )
    negative = (
        (labeled["hits"] == 1)
        & (labeled["best_proposal_score"] <= negative_score)
        & (labeled["local_contrast_gray_mean"] <= 6.0)
        & (labeled["pixel_area_mean"] <= 5.0)
    )
    labeled["weak_label"] = -1
    labeled.loc[negative, "weak_label"] = 0
    labeled.loc[positive, "weak_label"] = 1
    labeled["weak_label_name"] = "ambiguous"
    labeled.loc[negative, "weak_label_name"] = "background"
    labeled.loc[positive, "weak_label_name"] = "particle"
    labeled["weak_label_source"] = "image_processing_teacher_v1"
    labeled["weak_label_confidence"] = 0.0
    labeled.loc[positive, "weak_label_confidence"] = (
        0.70
        + 0.15
        * np.clip(
            (labeled.loc[positive, "hits"] - minimum_positive_hits) / 5.0,
            0.0,
            1.0,
        )
        + 0.15
        * np.clip(
            (
                labeled.loc[positive, "best_proposal_score"]
                - positive_score
            )
            / max(1.0 - positive_score, 1e-6),
            0.0,
            1.0,
        )
    )
    labeled.loc[negative, "weak_label_confidence"] = (
        0.75
        + 0.25
        * np.clip(
            (
                negative_score
                - labeled.loc[negative, "best_proposal_score"]
            )
            / max(negative_score, 1e-6),
            0.0,
            1.0,
        )
    )
    labeled["split_group"] = (
        labeled["video_slug"]
        + "_sequence_"
        + labeled["droplet_sequence"].astype(int).astype(str)
    )
    return labeled


def split_groups(
    tracks: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, str]:
    labeled = tracks[tracks["weak_label"] >= 0].copy()
    best: tuple[float, dict[str, str]] | None = None
    overall_ratio = float(labeled["weak_label"].mean())
    for candidate_seed in range(seed, seed + 120):
        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=0.30,
            random_state=candidate_seed,
        )
        train_index, temporary_index = next(
            outer.split(
                labeled,
                labeled["weak_label"],
                labeled["split_group"],
            )
        )
        temporary = labeled.iloc[temporary_index]
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=candidate_seed + 1000,
        )
        validation_local, test_local = next(
            inner.split(
                temporary,
                temporary["weak_label"],
                temporary["split_group"],
            )
        )
        subsets = {
            "train": labeled.iloc[train_index],
            "validation": temporary.iloc[validation_local],
            "test": temporary.iloc[test_local],
        }
        if any(
            subset["weak_label"].nunique() < 2
            for subset in subsets.values()
        ):
            continue
        score = 0.0
        targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
        for name, subset in subsets.items():
            score += abs(len(subset) / len(labeled) - targets[name])
            score += 0.8 * abs(float(subset["weak_label"].mean()) - overall_ratio)
        mapping: dict[str, str] = {}
        for name, subset in subsets.items():
            for group in subset["split_group"].unique():
                mapping[str(group)] = name
        if best is None or score < best[0]:
            best = (score, mapping)
    if best is None:
        raise RuntimeError("Could not create grouped train/validation/test split")
    return best[1]


def make_pipeline(classifier, *, scale: bool) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(
            ("scaler", RobustScaler(quantile_range=(10, 90)))
        )
    preprocessing = ColumnTransformer(
        [("numeric", Pipeline(numeric_steps), FEATURES)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def model_candidates(seed: int) -> dict[str, list[Pipeline]]:
    return {
        "logistic_regression": [
            make_pipeline(
                LogisticRegression(
                    C=value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
                scale=True,
            )
            for value in (0.1, 0.5, 1.0, 2.0, 10.0)
        ],
        "decision_tree": [
            make_pipeline(
                DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    class_weight="balanced",
                    random_state=seed,
                ),
                scale=False,
            )
            for depth in (3, 4, 5, 6, 8)
            for leaf in (5, 12)
        ],
        "random_forest": [
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed,
                ),
                scale=False,
            )
            for depth in (6, 9, 12)
            for leaf in (3, 8)
        ],
        "rbf_svm": [
            make_pipeline(
                SVC(
                    C=value,
                    gamma=gamma,
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
                scale=True,
            )
            for value in (0.5, 1.0, 2.0, 5.0)
            for gamma in ("scale", 0.1)
        ],
    }


def aggregate_track_predictions(
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    predicted = candidates[
        [
            "video_slug",
            "track_id",
            "droplet_sequence",
            "weak_label",
            "weak_label_name",
            "split",
        ]
    ].copy()
    predicted["probability"] = probabilities
    return (
        predicted.groupby(
            [
                "video_slug",
                "track_id",
                "droplet_sequence",
                "weak_label",
                "weak_label_name",
                "split",
            ],
            as_index=False,
        )
        .agg(
            probability_mean=("probability", "mean"),
            probability_max=("probability", "max"),
            observations=("probability", "size"),
        )
    )


def metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    accuracy = float(np.mean(predictions == labels))
    specificity = float(tn / max(tn + fp, 1))
    return {
        "threshold": float(threshold),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": accuracy,
        "specificity": specificity,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best: tuple[float, float] | None = None
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = metrics_at_threshold(labels, probabilities, float(threshold))
        recall_penalty = max(0.0, 0.90 - float(metrics["recall"]))
        score = (
            float(metrics["f1"])
            + 0.08 * float(metrics["recall"])
            - 0.50 * recall_penalty
        )
        if best is None or score > best[0]:
            best = (score, float(threshold))
    assert best is not None
    return best[1]


def track_predictions_for_split(
    model: Pipeline,
    candidates: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    selected = candidates[candidates["split"] == split]
    probabilities = model.predict_proba(selected[FEATURES])[:, 1]
    return aggregate_track_predictions(selected, probabilities)


def benchmark_model(model: Pipeline, sample: pd.DataFrame) -> dict[str, float]:
    one = sample.iloc[[0]][FEATURES]
    batch = sample.head(min(256, len(sample)))[FEATURES]
    for _ in range(40):
        model.predict_proba(one)
    iterations = 1200
    start = time.perf_counter()
    for _ in range(iterations):
        model.predict_proba(one)
    one_ms = (time.perf_counter() - start) * 1000.0 / iterations
    start = time.perf_counter()
    loops = 100
    for _ in range(loops):
        model.predict_proba(batch)
    batch_seconds = time.perf_counter() - start
    rows_per_second = len(batch) * loops / max(batch_seconds, 1e-9)
    return {
        "single_candidate_mean_ms": float(one_ms),
        "batch_rows": len(batch),
        "batch_candidates_per_second": float(rows_per_second),
    }


def make_metric_plot(path: Path, comparison: pd.DataFrame) -> None:
    metrics = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(comparison))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#1971c2", "#f59f00", "#12b886", "#845ef7"]
    for index, metric in enumerate(metrics):
        ax.bar(
            x + (index - 1.5) * width,
            comparison[metric],
            width,
            label=metric,
            color=colors[index],
        )
    ax.set_xticks(x, comparison["model"], rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weak-label test score")
    ax.set_title("Feature-based ML comparison on held-out droplet sequences")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_confusion_plot(
    path: Path,
    result_frames: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
) -> None:
    names = list(result_frames)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.flat, names):
        frame = result_frames[name]
        labels = frame["weak_label"].to_numpy(dtype=np.int64)
        predictions = (
            frame["probability_mean"].to_numpy() >= thresholds[name]
        ).astype(np.int64)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=(
                        "white"
                        if matrix[row, column] > matrix.max() * 0.55
                        else "black"
                    ),
                )
        ax.set_title(name)
        ax.set_xticks([0, 1], ["background", "particle"])
        ax.set_yticks([0, 1], ["background", "particle"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Weak label")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Track-level weak-label test confusion matrices")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_pr_plot(
    path: Path,
    result_frames: dict[str, pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, frame in result_frames.items():
        precision, recall, _ = precision_recall_curve(
            frame["weak_label"],
            frame["probability_mean"],
        )
        average_precision = average_precision_score(
            frame["weak_label"],
            frame["probability_mean"],
        )
        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{name}, AP={average_precision:.3f}",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Track-level precision-recall, weak-label test")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def classifier_feature_importance(model: Pipeline) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        return np.asarray(classifier.feature_importances_, dtype=np.float64)
    if hasattr(classifier, "coef_"):
        return np.abs(np.asarray(classifier.coef_[0], dtype=np.float64))
    return np.zeros(len(FEATURES), dtype=np.float64)


def make_importance_plot(
    path: Path,
    model: Pipeline,
    *,
    title: str,
) -> None:
    importance = classifier_feature_importance(model)
    order = np.argsort(importance)[-15:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [FEATURES[index] for index in order],
        importance[order],
        color="#0b7285",
    )
    ax.set_title(title)
    ax.set_xlabel("Model importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    model_dir = output / "models"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(
        dataset / "analysis_unsupervised_v1" / "track_clusters.csv"
    )
    candidates = prepare_candidate_features(
        pd.read_csv(dataset / "particle_candidate_features.csv")
    )
    weak_tracks = create_weak_labels(
        tracks,
        minimum_positive_hits=args.minimum_positive_hits,
        positive_score=args.positive_score,
        negative_score=args.negative_score,
    )
    split_mapping = split_groups(weak_tracks, seed=args.random_seed)
    weak_tracks["split"] = weak_tracks["split_group"].map(split_mapping)
    weak_tracks.loc[weak_tracks["weak_label"] < 0, "split"] = "not_used"
    weak_tracks.to_csv(output / "weak_labeled_tracks.csv", index=False)

    join_columns = [
        "video_slug",
        "track_id",
        "weak_label",
        "weak_label_name",
        "weak_label_confidence",
        "split_group",
        "split",
    ]
    learning_candidates = candidates.merge(
        weak_tracks[join_columns],
        on=["video_slug", "track_id"],
        how="inner",
        validate="many_to_one",
    )
    learning_candidates = learning_candidates[
        learning_candidates["weak_label"] >= 0
    ].copy()
    observation_counts = learning_candidates.groupby(
        ["video_slug", "track_id"]
    )["candidate_id"].transform("size")
    learning_candidates["sample_weight"] = (
        learning_candidates["weak_label_confidence"]
        / np.maximum(observation_counts, 1)
    )
    learning_candidates.to_csv(
        output / "weak_labeled_candidate_observations.csv",
        index=False,
    )

    train = learning_candidates[learning_candidates["split"] == "train"]
    validation = learning_candidates[
        learning_candidates["split"] == "validation"
    ]
    test = learning_candidates[learning_candidates["split"] == "test"]
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("A learning split is empty")

    best_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    validation_results: dict[str, dict[str, float | int]] = {}
    test_results: dict[str, dict[str, float | int]] = {}
    test_prediction_frames: dict[str, pd.DataFrame] = {}
    comparison_rows: list[dict[str, object]] = []

    for model_name, candidates_for_model in model_candidates(
        args.random_seed
    ).items():
        best_selection: tuple[
            float,
            Pipeline,
            float,
            dict[str, float | int],
        ] | None = None
        for candidate_model in candidates_for_model:
            model = clone(candidate_model)
            model.fit(
                train[FEATURES],
                train["weak_label"],
                classifier__sample_weight=train["sample_weight"],
            )
            validation_tracks = track_predictions_for_split(
                model,
                validation,
                "validation",
            )
            labels = validation_tracks["weak_label"].to_numpy(dtype=np.int64)
            probabilities = validation_tracks[
                "probability_mean"
            ].to_numpy()
            threshold = select_threshold(labels, probabilities)
            metrics = metrics_at_threshold(labels, probabilities, threshold)
            score = float(metrics["f1"]) + 0.08 * float(metrics["recall"])
            if best_selection is None or score > best_selection[0]:
                best_selection = (score, model, threshold, metrics)
        assert best_selection is not None
        _, best_model, threshold, validation_metrics = best_selection
        best_models[model_name] = best_model
        thresholds[model_name] = threshold
        validation_results[model_name] = validation_metrics

        test_tracks = track_predictions_for_split(best_model, test, "test")
        test_metrics = metrics_at_threshold(
            test_tracks["weak_label"].to_numpy(dtype=np.int64),
            test_tracks["probability_mean"].to_numpy(),
            threshold,
        )
        test_results[model_name] = test_metrics
        test_prediction_frames[model_name] = test_tracks
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(best_model, model_path)
        runtime = benchmark_model(best_model, test)
        comparison_rows.append(
            {
                "model": model_name,
                **test_metrics,
                **runtime,
                "serialized_bytes": model_path.stat().st_size,
                "validation_f1": validation_metrics["f1"],
                "validation_recall": validation_metrics["recall"],
            }
        )
        print(
            f"{model_name}: val_f1={validation_metrics['f1']:.3f} "
            f"test_f1={test_metrics['f1']:.3f} "
            f"test_recall={test_metrics['recall']:.3f}"
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    best_pc_name = str(
        comparison.sort_values(
            ["validation_f1", "validation_recall"],
            ascending=False,
        ).iloc[0]["model"]
    )
    fpga_candidates = comparison[
        comparison["model"].isin(["decision_tree", "logistic_regression"])
    ]
    best_fpga_name = str(
        fpga_candidates.sort_values(
            ["validation_f1", "serialized_bytes"],
            ascending=[False, True],
        ).iloc[0]["model"]
    )

    joblib.dump(best_models[best_pc_name], output / "best_pc_model.joblib")
    joblib.dump(
        best_models[best_fpga_name],
        output / "recommended_fpga_feature_model.joblib",
    )
    tree_classifier = best_models["decision_tree"].named_steps["classifier"]
    (output / "decision_tree_rules.txt").write_text(
        export_text(tree_classifier, feature_names=FEATURES),
        encoding="utf-8",
    )

    all_probabilities_pc = best_models[best_pc_name].predict_proba(
        candidates[FEATURES]
    )[:, 1]
    all_probabilities_fpga = best_models[best_fpga_name].predict_proba(
        candidates[FEATURES]
    )[:, 1]
    full = candidates[
        [
            "video_slug",
            "source_video",
            "frame_index",
            "droplet_sequence",
            "track_id",
            "candidate_id",
        ]
    ].copy()
    full["pc_probability"] = all_probabilities_pc
    full["fpga_probability"] = all_probabilities_fpga
    aggregated = (
        full.groupby(
            [
                "video_slug",
                "source_video",
                "droplet_sequence",
                "track_id",
            ],
            as_index=False,
        )
        .agg(
            first_frame=("frame_index", "min"),
            last_frame=("frame_index", "max"),
            observations=("candidate_id", "size"),
            pc_probability=("pc_probability", "mean"),
            fpga_probability=("fpga_probability", "mean"),
        )
    )
    aggregated["pc_predicted_particle"] = (
        aggregated["pc_probability"] >= thresholds[best_pc_name]
    ).astype(int)
    aggregated["fpga_predicted_particle"] = (
        aggregated["fpga_probability"] >= thresholds[best_fpga_name]
    ).astype(int)
    aggregated = aggregated.merge(
        weak_tracks[
            [
                "video_slug",
                "track_id",
                "weak_label",
                "weak_label_name",
                "split",
            ]
        ],
        on=["video_slug", "track_id"],
        how="left",
        validate="one_to_one",
    )
    aggregated.to_csv(output / "all_track_predictions.csv", index=False)
    sequence_counts = (
        aggregated.groupby(
            ["video_slug", "source_video", "droplet_sequence"],
            as_index=False,
        )
        .agg(
            candidate_tracks=("track_id", "size"),
            pc_predicted_particles=("pc_predicted_particle", "sum"),
            fpga_predicted_particles=("fpga_predicted_particle", "sum"),
            mean_pc_probability=("pc_probability", "mean"),
            mean_fpga_probability=("fpga_probability", "mean"),
        )
    )
    sequence_counts.to_csv(
        output / "droplet_sequence_ml_counts.csv",
        index=False,
    )

    make_metric_plot(output / "model_metric_comparison.png", comparison)
    make_confusion_plot(
        output / "test_confusion_matrices.png",
        test_prediction_frames,
        thresholds,
    )
    make_pr_plot(
        output / "test_precision_recall_curves.png",
        test_prediction_frames,
    )
    make_importance_plot(
        output / "fpga_model_feature_importance.png",
        best_models[best_fpga_name],
        title=f"Feature importance: {best_fpga_name}",
    )

    split_counts = (
        weak_tracks[weak_tracks["weak_label"] >= 0]
        .groupby(["split", "weak_label_name"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    summary = {
        "experiment": "weak_supervised_feature_ml_v1",
        "independent_from_roboflow": True,
        "accuracy_interpretation": (
            "Metrics measure agreement with conservative image-processing "
            "weak labels on held-out droplet sequences, not ground-truth "
            "microplastic accuracy."
        ),
        "weak_label_policy": {
            "positive": {
                "minimum_hits": args.minimum_positive_hits,
                "minimum_best_proposal_score": args.positive_score,
                "maximum_radial_distance_norm": 0.90,
            },
            "negative": {
                "hits": 1,
                "maximum_best_proposal_score": args.negative_score,
                "maximum_local_contrast_gray": 6.0,
                "maximum_pixel_area": 5.0,
            },
        },
        "counts": {
            "tracks_total": len(weak_tracks),
            "weak_positive_tracks": int(
                (weak_tracks["weak_label"] == 1).sum()
            ),
            "weak_negative_tracks": int(
                (weak_tracks["weak_label"] == 0).sum()
            ),
            "ambiguous_tracks_not_used": int(
                (weak_tracks["weak_label"] < 0).sum()
            ),
            "candidate_observations_used": len(learning_candidates),
        },
        "split_track_counts": split_counts,
        "features": FEATURES,
        "best_pc_model": best_pc_name,
        "recommended_fpga_model": best_fpga_name,
        "thresholds": thresholds,
        "validation_metrics": validation_results,
        "test_metrics": test_results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        (
            "# Weak-supervised feature ML experiment\n\n"
            "This is experiment 1 and is independent from Roboflow. Labels are "
            "generated conservatively by temporal image processing. Reported "
            "metrics are weak-label agreement, not real microplastic accuracy. "
            "Roboflow will later form experiment 2 for ground-truth comparison.\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"], indent=2))
    print(f"Best PC model: {best_pc_name}")
    print(f"Recommended FPGA model: {best_fpga_name}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
