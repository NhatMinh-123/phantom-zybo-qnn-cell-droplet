from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "valid", "test")
SOURCE_PATTERN = re.compile(r"_src(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a source-frame grouped YOLO split")
    parser.add_argument("--source", type=Path, default=Path("dataset/cell_droplet_roi384"))
    parser.add_argument(
        "--output", type=Path, default=Path("dataset/cell_droplet_roi384_grouped")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=5000)
    return parser.parse_args()


def source_id(path: Path) -> str:
    match = SOURCE_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Cannot extract source frame from {path.name}")
    return match.group(1)


def label_counts(path: Path) -> tuple[int, int]:
    counts = [0, 0]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts[0], counts[1]


def collect_groups(root: Path) -> dict[str, list[tuple[Path, Path]]]:
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for split in SPLITS:
        for image in (root / split / "images").glob("*.*"):
            label = root / split / "labels" / f"{image.stem}.txt"
            if label.exists():
                groups[source_id(image)].append((image, label))
    return dict(groups)


def choose_split(
    groups: dict[str, list[tuple[Path, Path]]], seed: int, trials: int
) -> dict[str, list[str]]:
    ids = sorted(groups)
    counts = {
        group_id: tuple(
            sum(values)
            for values in zip(*(label_counts(label) for _image, label in groups[group_id]))
        )
        for group_id in ids
    }
    target_sizes = (round(len(ids) * 0.70), round(len(ids) * 0.15))
    target_sizes = (*target_sizes, len(ids) - sum(target_sizes))
    total = tuple(sum(counts[group_id][class_id] for group_id in ids) for class_id in range(2))
    ratios = (0.70, 0.15, 0.15)
    best_score = float("inf")
    best: dict[str, list[str]] | None = None
    rng = random.Random(seed)
    for _ in range(trials):
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        boundaries = (target_sizes[0], target_sizes[0] + target_sizes[1])
        candidates = (
            shuffled[: boundaries[0]],
            shuffled[boundaries[0] : boundaries[1]],
            shuffled[boundaries[1] :],
        )
        score = 0.0
        for split_index, group_ids in enumerate(candidates):
            for class_id in range(2):
                actual = sum(counts[group_id][class_id] for group_id in group_ids)
                expected = total[class_id] * ratios[split_index]
                score += ((actual - expected) / max(expected, 1)) ** 2
        if score < best_score:
            best_score = score
            best = dict(zip(SPLITS, (sorted(value) for value in candidates)))
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    groups = collect_groups(args.source)
    assignment = choose_split(groups, args.seed, args.trials)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    summary: dict[str, object] = {"seed": args.seed, "splits": {}}
    for split, group_ids in assignment.items():
        image_dir = args.output / split / "images"
        label_dir = args.output / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        class_counts = [0, 0]
        image_count = 0
        for group_id in group_ids:
            for image, label in groups[group_id]:
                shutil.copy2(image, image_dir / image.name)
                shutil.copy2(label, label_dir / label.name)
                values = label_counts(label)
                class_counts = [left + right for left, right in zip(class_counts, values)]
                image_count += 1
        summary["splits"][split] = {
            "source_frames": group_ids,
            "images": image_count,
            "cell_boxes": class_counts[0],
            "droplet_boxes": class_counts[1],
        }
    source_yaml = (args.source / "data.yaml").read_text(encoding="utf-8")
    yaml_lines = source_yaml.splitlines()
    yaml_lines[0] = f"path: {args.output.resolve().as_posix()}"
    (args.output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (args.output / "split_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
