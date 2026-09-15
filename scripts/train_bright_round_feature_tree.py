#!/usr/bin/env python3
"""Train a compact tree to reproduce the strict bright-round feature target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bright_round_feature_gate import (  # noqa: E402
    BrightRoundGateConfig,
    evaluate_bright_round_candidate,
)
from scripts.train_weak_supervised_feature_models import (  # noqa: E402
    FEATURES,
    prepare_candidate_features,
)


DEFAULT_INPUT = (
    ROOT
    / "dataset"
    / "droplet_microplastic_features_unlabeled_v1"
    / "particle_candidate_features.csv"
)
DEFAULT_OUTPUT = ROOT / "models" / "bright_round_feature_tree_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        prediction,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(
        target,
        prediction,
        labels=[0, 1],
    ).ravel()
    return {
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(target, prediction)),
    }


def split_indices(
    data: pd.DataFrame,
    target: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = (
        data["video_slug"].astype(str)
        + "_s"
        + data["droplet_sequence"].astype(str)
    )
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_valid, test = next(outer.split(data, target, groups))
    inner_groups = groups.iloc[train_valid]
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed + 1,
    )
    train_local, valid_local = next(
        inner.split(data.iloc[train_valid], target[train_valid], inner_groups)
    )
    return train_valid[train_local], train_valid[valid_local], test


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    gate_config = BrightRoundGateConfig()
    target = np.asarray(
        [
            int(evaluate_bright_round_candidate(row, gate_config).passed)
            for row in data.to_dict("records")
        ],
        dtype=np.uint8,
    )
    prepared = prepare_candidate_features(data)
    train, valid, test = split_indices(data, target, seed=args.seed)

    evaluation_model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
    )
    evaluation_model.fit(prepared.iloc[train][FEATURES], target[train])
    validation_prediction = evaluation_model.predict(
        prepared.iloc[valid][FEATURES]
    )
    test_prediction = evaluation_model.predict(prepared.iloc[test][FEATURES])

    final_model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
    )
    final_model.fit(prepared[FEATURES], target)
    model_path = output / "recommended_fpga_feature_model.joblib"
    joblib.dump(final_model, model_path)
    (output / "tree.txt").write_text(
        export_text(final_model, feature_names=FEATURES),
        encoding="utf-8",
    )

    summary = {
        "experiment": "bright_round_feature_tree_v1",
        "input": str(input_path),
        "target_definition": (
            "Compact, approximately round candidate with a strong black-hat "
            "response; dim, diffuse, elongated, rim-like, and off-core "
            "responses are background."
        ),
        "accuracy_interpretation": (
            "Metrics measure agreement with the explicit feature gate on "
            "held-out droplet sequences, not biological ground-truth accuracy."
        ),
        "counts": {
            "all": int(len(data)),
            "particle": int(target.sum()),
            "background": int(len(target) - target.sum()),
            "train": int(len(train)),
            "validation": int(len(valid)),
            "test": int(len(test)),
        },
        "gate": vars(gate_config),
        "features": FEATURES,
        "model": {
            "type": "DecisionTreeClassifier",
            "max_depth": int(final_model.get_depth()),
            "node_count": int(final_model.tree_.node_count),
            "min_samples_leaf": args.min_samples_leaf,
            "threshold": 0.5,
        },
        "thresholds": {"decision_tree": 0.5},
        "validation_metrics": metrics(target[valid], validation_prediction),
        "test_metrics": metrics(target[test], test_prediction),
        "artifacts": {
            "model": str(model_path),
            "tree": str(output / "tree.txt"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
