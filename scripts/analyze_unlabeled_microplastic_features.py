#!/usr/bin/env python3
"""Cluster unlabeled candidate tracks and package feature diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURES = [
    "hits",
    "duration_frames",
    "persistence_ratio",
    "mean_speed_px_per_frame",
    "path_length_px",
    "proposal_score_mean",
    "proposal_score_max",
    "pixel_area_mean",
    "circularity_mean",
    "solidity_mean",
    "local_contrast_gray_mean",
    "blackhat_mean_mean",
    "temporal_mean_mean",
    "gradient_mean_mean",
    "local_entropy_mean",
    "local_laplacian_variance_mean",
    "radial_distance_norm_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--review-per-cluster", type=int, default=80)
    return parser.parse_args()


def percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average").fillna(0.5)


def make_cluster_plot(
    path: Path,
    tracks: pd.DataFrame,
    cluster_count: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, cluster_count))
    for cluster_id in range(cluster_count):
        selected = tracks[tracks["cluster_id"] == cluster_id]
        ax.scatter(
            selected["pca_1"],
            selected["pca_2"],
            s=np.clip(10 + selected["hits"] * 4, 10, 55),
            alpha=0.52,
            color=colors[cluster_id],
            label=f"Cluster {cluster_id} (n={len(selected)})",
            edgecolors="none",
        )
    persistent = tracks[tracks["hits"] >= 3]
    ax.scatter(
        persistent["pca_1"],
        persistent["pca_2"],
        s=34,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.55,
        alpha=0.55,
        label=f"Persistent tracks (n={len(persistent)})",
    )
    ax.set_title("Unsupervised candidate-track clusters")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_feature_plot(path: Path, tracks: pd.DataFrame) -> None:
    selected_features = [
        ("hits", "Temporal hits"),
        ("proposal_score_max", "Max proposal score"),
        ("pixel_area_mean", "Mean component area"),
        ("local_contrast_gray_mean", "Mean local contrast"),
        ("blackhat_mean_mean", "Mean black-hat response"),
        ("mean_speed_px_per_frame", "Mean speed"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (feature, title) in zip(axes.flat, selected_features):
        values = tracks[feature].replace([np.inf, -np.inf], np.nan).dropna()
        upper = float(values.quantile(0.99)) if len(values) else 1.0
        values = values.clip(upper=upper)
        ax.hist(values, bins=35, color="#1971c2", alpha=0.82)
        ax.set_title(title)
        ax.set_xlabel(feature)
        ax.set_ylabel("Track count")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Candidate-track feature distributions", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_droplet_plot(
    path: Path,
    droplets: pd.DataFrame,
    sequences: pd.DataFrame,
) -> None:
    video_slugs = list(droplets["video_slug"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    radius_values = [
        droplets.loc[droplets["video_slug"] == slug, "radius_px"].to_numpy()
        for slug in video_slugs
    ]
    axes[0].boxplot(radius_values, tick_labels=video_slugs, showfliers=False)
    axes[0].set_title("Detected droplet radius by video")
    axes[0].set_ylabel("Radius (pixels)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].scatter(
        sequences["frames_observed"],
        sequences["candidate_tracks"],
        c=sequences["persistent_candidate_tracks"],
        cmap="viridis",
        s=28,
        alpha=0.75,
    )
    axes[1].set_title("Candidate tracks per droplet sequence")
    axes[1].set_xlabel("Observed frames")
    axes[1].set_ylabel("Candidate tracks")
    axes[1].grid(alpha=0.2)
    colorbar = fig.colorbar(axes[1].collections[0], ax=axes[1])
    colorbar.set_label("Persistent candidate tracks")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_correlation_plot(path: Path, tracks: pd.DataFrame) -> None:
    abbreviated = {
        "hits": "hits",
        "persistence_ratio": "persist",
        "mean_speed_px_per_frame": "speed",
        "proposal_score_max": "score",
        "pixel_area_mean": "area",
        "circularity_mean": "circular",
        "local_contrast_gray_mean": "contrast",
        "blackhat_mean_mean": "blackhat",
        "temporal_mean_mean": "temporal",
        "radial_distance_norm_mean": "radius",
    }
    correlation = tracks[list(abbreviated)].rename(
        columns=abbreviated
    ).corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=45)
    ax.set_yticks(range(len(correlation)), correlation.index)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            ax.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if abs(correlation.iloc[row, column]) > 0.55
                    else "black"
                ),
            )
    ax.set_title("Feature correlation, unlabeled tracks")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(
    dataset: Path,
    selected: pd.DataFrame,
    output: Path,
    *,
    title: str,
) -> None:
    columns = 8
    rows = 5
    tile = 112
    label_height = 32
    header = 42
    sheet = np.full(
        (header + rows * (tile + label_height), columns * tile, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(
        sheet,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    for index, (_, row) in enumerate(selected.head(columns * rows).iterrows()):
        patch_path = dataset / str(row["best_patch"])
        patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
        if patch is None:
            continue
        patch = cv2.resize(patch, (tile, tile), interpolation=cv2.INTER_NEAREST)
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        grid_row, grid_column = divmod(index, columns)
        x = grid_column * tile
        y = header + grid_row * (tile + label_height)
        sheet[y : y + tile, x : x + tile] = patch
        cv2.putText(
            sheet,
            f"h={int(row['hits'])} s={row['best_proposal_score']:.2f}",
            (x + 3, y + tile + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"f={int(row['best_frame'])}",
            (x + 3, y + tile + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    sheets_dir = output / "cluster_contact_sheets"
    model_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(dataset / "particle_track_features.csv")
    droplets = pd.read_csv(dataset / "droplet_frame_features.csv")
    sequences = pd.read_csv(dataset / "droplet_sequence_features.csv")

    feature_frame = tracks[FEATURES].replace([np.inf, -np.inf], np.nan)
    preprocessing = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10, 90))),
        ]
    )
    matrix = preprocessing.fit_transform(feature_frame)
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(matrix)
    kmeans = KMeans(
        n_clusters=args.clusters,
        random_state=42,
        n_init=20,
    )
    cluster_ids = kmeans.fit_predict(matrix)
    distances = np.min(kmeans.transform(matrix), axis=1)
    isolation = IsolationForest(
        n_estimators=250,
        contamination=0.12,
        random_state=42,
        n_jobs=-1,
    )
    isolation.fit(matrix)
    anomaly_scores = -isolation.score_samples(matrix)

    clustered = tracks.copy()
    clustered["cluster_id"] = cluster_ids
    clustered["pca_1"] = embedding[:, 0]
    clustered["pca_2"] = embedding[:, 1]
    clustered["cluster_distance"] = distances
    clustered["anomaly_score"] = anomaly_scores
    clustered["is_anomaly"] = (isolation.predict(matrix) == -1).astype(int)
    clustered["review_priority"] = (
        0.32 * clustered["best_proposal_score"].clip(0, 1)
        + 0.25 * (clustered["hits"].clip(0, 5) / 5.0)
        + 0.18 * clustered["persistence_ratio"].clip(0, 1)
        + 0.15 * percentile_rank(clustered["anomaly_score"])
        + 0.10 * percentile_rank(clustered["cluster_distance"])
    )
    clustered["ground_truth_class"] = ""
    clustered["label_status"] = "unlabeled"
    clustered.to_csv(output / "track_clusters.csv", index=False)

    profile_columns = [
        "hits",
        "persistence_ratio",
        "mean_speed_px_per_frame",
        "best_proposal_score",
        "pixel_area_mean",
        "circularity_mean",
        "local_contrast_gray_mean",
        "blackhat_mean_mean",
        "temporal_mean_mean",
        "radial_distance_norm_mean",
        "anomaly_score",
    ]
    profiles = (
        clustered.groupby("cluster_id")[profile_columns]
        .agg(["count", "mean", "median", "std"])
    )
    profiles.columns = [
        f"{feature}_{statistic}"
        for feature, statistic in profiles.columns
    ]
    profiles.reset_index().to_csv(
        output / "cluster_profiles.csv",
        index=False,
    )

    review_groups: list[pd.DataFrame] = []
    for cluster_id in range(args.clusters):
        group = clustered[clustered["cluster_id"] == cluster_id].copy()
        persistent = group[group["hits"] >= 3].sort_values(
            "review_priority",
            ascending=False,
        )
        transient = group[group["hits"] < 3].sort_values(
            "review_priority",
            ascending=False,
        )
        half = args.review_per_cluster // 2
        selected = pd.concat(
            [
                persistent.head(half),
                transient.head(args.review_per_cluster - min(half, len(persistent))),
            ]
        ).drop_duplicates("track_key")
        if len(selected) < args.review_per_cluster:
            remainder = group[
                ~group["track_key"].isin(selected["track_key"])
            ].sort_values("review_priority", ascending=False)
            selected = pd.concat(
                [
                    selected,
                    remainder.head(args.review_per_cluster - len(selected)),
                ]
            )
        selected = selected.head(args.review_per_cluster).copy()
        selected["reviewed_label"] = ""
        selected["review_status"] = "pending"
        review_groups.append(selected)

        representative = group.assign(
            centroid_distance=np.linalg.norm(
                embedding[group.index]
                - np.mean(embedding[group.index], axis=0),
                axis=1,
            )
        ).sort_values(
            ["centroid_distance", "hits"],
            ascending=[True, False],
        )
        make_contact_sheet(
            dataset,
            representative,
            sheets_dir / f"cluster_{cluster_id}_representative.jpg",
            title=f"Cluster {cluster_id}: representative unlabeled tracks",
        )
        make_contact_sheet(
            dataset,
            group.sort_values("review_priority", ascending=False),
            sheets_dir / f"cluster_{cluster_id}_priority.jpg",
            title=f"Cluster {cluster_id}: high review priority",
        )

    review = pd.concat(review_groups, ignore_index=True)
    review.to_csv(output / "active_learning_priority.csv", index=False)

    make_cluster_plot(
        output / "candidate_clusters_pca.png",
        clustered,
        args.clusters,
    )
    make_feature_plot(
        output / "candidate_feature_distributions.png",
        clustered,
    )
    make_droplet_plot(
        output / "droplet_feature_overview.png",
        droplets,
        sequences,
    )
    make_correlation_plot(
        output / "candidate_feature_correlation.png",
        clustered,
    )

    joblib.dump(preprocessing, model_dir / "feature_preprocessing.joblib")
    joblib.dump(pca, model_dir / "pca.joblib")
    joblib.dump(kmeans, model_dir / "kmeans.joblib")
    joblib.dump(isolation, model_dir / "isolation_forest.joblib")

    cluster_counts = {
        str(cluster_id): int(count)
        for cluster_id, count in clustered["cluster_id"].value_counts().sort_index().items()
    }
    summary = {
        "dataset": str(dataset),
        "output": str(output),
        "status": (
            "Unsupervised analysis only. Clusters and anomaly scores are not "
            "particle/background labels."
        ),
        "features": FEATURES,
        "tracks": len(clustered),
        "persistent_tracks": int((clustered["hits"] >= 3).sum()),
        "clusters": args.clusters,
        "cluster_counts": cluster_counts,
        "pca_explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "anomalies": int(clustered["is_anomaly"].sum()),
        "active_learning_rows": len(review),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        (
            "# Unsupervised feature analysis\n\n"
            "KMeans clusters, PCA coordinates, Isolation Forest scores, and "
            "review priorities are exploratory outputs. They are not ground "
            "truth labels. Use `active_learning_priority.csv` to review diverse "
            "tracks while waiting for Roboflow, or match Roboflow boxes later.\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Analysis: {output}")


if __name__ == "__main__":
    main()
