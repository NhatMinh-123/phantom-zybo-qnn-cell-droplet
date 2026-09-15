"""Build a leakage-safe 32x32 QNN dataset from reviewed candidate labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = {"background", "particle"}
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "microplastic_patch32_reviewed_v2",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_labels(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Review CSV is empty")
    identifiers = [row["review_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Review CSV contains duplicate review_id values")
    counts: dict[str, int] = defaultdict(int)
    selected: list[dict[str, str]] = []
    for row in rows:
        counts[f"{row.get('item_type', '')}:{row.get('review_status', '')}:{row.get('reviewed_label', '')}"] += 1
        if row.get("item_type") != "candidate":
            continue
        if row.get("review_status") != "reviewed":
            continue
        if row.get("reviewed_label") not in VALID_LABELS:
            continue
        patch = Path(row["source_patch"])
        if not patch.is_file():
            raise FileNotFoundError(patch)
        image = cv2.imread(str(patch), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (32, 32):
            raise RuntimeError(f"Expected a readable 32x32 patch: {patch}")
        selected.append(row)
    if not selected:
        raise RuntimeError("No reviewed particle/background candidates found")
    return selected, dict(counts)


def group_statistics(rows: list[dict[str, str]]) -> dict[int, dict[str, int]]:
    groups: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "particle": 0, "background": 0}
    )
    for row in rows:
        sequence = int(row["droplet_sequence"])
        label = row["reviewed_label"]
        groups[sequence]["total"] += 1
        groups[sequence][label] += 1
    return dict(groups)


def choose_grouped_split(
    groups: dict[int, dict[str, int]],
    *,
    train_ratio: float,
    valid_ratio: float,
    trials: int,
    seed: int,
) -> dict[int, str]:
    test_ratio = 1.0 - train_ratio - valid_ratio
    ratios = {"train": train_ratio, "valid": valid_ratio, "test": test_ratio}
    if min(ratios.values()) <= 0:
        raise ValueError("Split ratios must all be positive")
    sequences = sorted(groups)
    count = len(sequences)
    valid_groups = max(1, round(count * valid_ratio))
    test_groups = max(1, round(count * test_ratio))
    train_groups = count - valid_groups - test_groups
    if train_groups <= 0:
        raise ValueError("Not enough groups for the requested split")
    group_counts = {"train": train_groups, "valid": valid_groups, "test": test_groups}
    totals = {
        key: sum(group[key] for group in groups.values())
        for key in ("total", "particle", "background")
    }
    targets = {
        split: {key: totals[key] * ratios[split] for key in totals}
        for split in SPLITS
    }
    rng = random.Random(seed)
    best_score = float("inf")
    best: dict[int, str] | None = None
    for _ in range(trials):
        order = sequences.copy()
        rng.shuffle(order)
        assignment: dict[int, str] = {}
        offset = 0
        for split in SPLITS:
            for sequence in order[offset : offset + group_counts[split]]:
                assignment[sequence] = split
            offset += group_counts[split]
        actual = {
            split: {key: 0 for key in totals}
            for split in SPLITS
        }
        for sequence, split in assignment.items():
            for key in totals:
                actual[split][key] += groups[sequence][key]
        if any(actual[split]["particle"] == 0 for split in SPLITS):
            continue
        score = 0.0
        for split in SPLITS:
            for key, weight in (("total", 1.0), ("particle", 4.0), ("background", 1.0)):
                target = max(targets[split][key], 1.0)
                score += weight * ((actual[split][key] - target) / target) ** 2
        if score < best_score:
            best_score = score
            best = assignment
    if best is None:
        raise RuntimeError("Could not create a grouped split containing particles in every split")
    return best


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    labels = args.labels.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")
    rows, review_counts = read_labels(labels)
    groups = group_statistics(rows)
    assignment = choose_grouped_split(
        groups,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        trials=args.trials,
        seed=args.seed,
    )
    for split in SPLITS:
        for label in sorted(VALID_LABELS):
            (output / split / label).mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    counts = {
        split: {"total": 0, "particle": 0, "background": 0, "sequences": set()}
        for split in SPLITS
    }
    for row in sorted(rows, key=lambda item: item["review_id"]):
        sequence = int(row["droplet_sequence"])
        split = assignment[sequence]
        label = row["reviewed_label"]
        source = Path(row["source_patch"])
        name = f"{row['review_id']}_seq{sequence:04d}_frame{int(row['best_frame']):05d}.png"
        destination = output / split / label / name
        shutil.copy2(source, destination)
        counts[split]["total"] += 1
        counts[split][label] += 1
        counts[split]["sequences"].add(sequence)
        manifest.append(
            {
                **row,
                "split": split,
                "dataset_path": destination.relative_to(output).as_posix(),
            }
        )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    shutil.copy2(labels, output / "review_labels_source.csv")
    serializable_counts = {
        split: {
            **{key: value for key, value in values.items() if key != "sequences"},
            "sequence_count": len(values["sequences"]),
            "sequence_ids": sorted(values["sequences"]),
        }
        for split, values in counts.items()
    }
    summary = {
        "name": "microplastic_patch32_reviewed_v2",
        "purpose": "Manually reviewed particle/background patches for Tiny QNN",
        "source_labels": str(labels),
        "source_labels_sha256": sha256(labels),
        "image_size": [32, 32],
        "group_key": "droplet_sequence",
        "seed": args.seed,
        "trials": args.trials,
        "counts": serializable_counts,
        "review_csv_counts": review_counts,
        "excluded": {
            "uncertain_candidates": review_counts.get("candidate:reviewed:uncertain", 0),
            "pending_audits": sum(
                value for key, value in review_counts.items() if key.startswith("audit:pending:")
            ),
        },
        "limitation": "All reviewed patches originate from one video; independent-video testing is still required.",
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
