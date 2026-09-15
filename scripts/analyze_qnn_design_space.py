from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = ("train", "valid", "test")
CLASSES = ("cell", "droplet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FPGA detector design choices")
    parser.add_argument(
        "--data", type=Path, default=Path("dataset/cell_droplet_roi384")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/fpga_algorithm_review")
    )
    return parser.parse_args()


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = line.split()
        labels.append((int(class_id), float(x), float(y), float(width), float(height)))
    return labels


def audit_dataset(
    root: Path,
    grids: tuple[int, ...],
    rectangular_grids: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    result: dict[str, object] = {"splits": {}, "box_statistics": {}}
    sizes = {class_id: [] for class_id in range(len(CLASSES))}
    for split in SPLITS:
        label_dir = root / split / "labels"
        labels_by_image = [read_labels(path) for path in sorted(label_dir.glob("*.txt"))]
        collisions = {grid: 0 for grid in grids}
        rectangular_collisions = {f"{width}x{height}": 0 for width, height in rectangular_grids}
        counts = [0] * len(CLASSES)
        for labels in labels_by_image:
            for class_id, _x, _y, width, height in labels:
                counts[class_id] += 1
                sizes[class_id].append((width, height))
            for grid in grids:
                occupied: set[tuple[int, int, int]] = set()
                for class_id, x, y, _width, _height in labels:
                    key = (class_id, min(int(y * grid), grid - 1), min(int(x * grid), grid - 1))
                    collisions[grid] += key in occupied
                    occupied.add(key)
            for grid_width, grid_height in rectangular_grids:
                occupied = set()
                for class_id, x, y, _width, _height in labels:
                    key = (
                        class_id,
                        min(int(y * grid_height), grid_height - 1),
                        min(int(x * grid_width), grid_width - 1),
                    )
                    rectangular_collisions[f"{grid_width}x{grid_height}"] += key in occupied
                    occupied.add(key)
        result["splits"][split] = {
            "images": len(labels_by_image),
            "boxes": sum(counts),
            "boxes_per_class": dict(zip(CLASSES, counts)),
            "collisions_per_grid": collisions,
            "collisions_per_rectangular_grid": rectangular_collisions,
        }

    for class_id, class_name in enumerate(CLASSES):
        widths = sorted(item[0] for item in sizes[class_id])
        heights = sorted(item[1] for item in sizes[class_id])
        result["box_statistics"][class_name] = {
            "count": len(widths),
            "width_normalized": percentiles(widths),
            "height_normalized": percentiles(heights),
        }
    return result


def percentiles(values: list[float]) -> dict[str, float]:
    def at(fraction: float) -> float:
        return values[round((len(values) - 1) * fraction)] if values else 0.0

    return {"min": at(0), "p10": at(0.1), "median": at(0.5), "p90": at(0.9), "max": at(1)}


def estimate_designs() -> list[dict[str, int | str | float]]:
    designs = [
        ("current_256_s4", 256, 256, 4),
        ("candidate_192_s8", 192, 192, 8),
        ("candidate_192x144_s8", 192, 144, 8),
        ("candidate_160x128_s8", 160, 128, 8),
    ]
    channels = (8, 12, 16, 16)
    rows = []
    for name, width, height, downsample in designs:
        shapes = []
        in_channels = 1
        current_width, current_height = width, height
        macs = 0
        strides = (2, 2, 1 if downsample == 4 else 2, 1)
        for out_channels, stride in zip(channels, strides):
            current_width = (current_width + stride - 1) // stride
            current_height = (current_height + stride - 1) // stride
            macs += current_width * current_height * in_channels * out_channels * 9
            shapes.append(f"{current_width}x{current_height}x{out_channels}")
            in_channels = out_channels
        output_values = current_width * current_height * len(CLASSES) * 5
        macs += current_width * current_height * channels[-1] * len(CLASSES) * 5
        rows.append(
            {
                "design": name,
                "input": f"{width}x{height}",
                "grid": f"{current_width}x{current_height}",
                "feature_shapes": ";".join(shapes),
                "macs": macs,
                "macs_million": round(macs / 1_000_000, 3),
                "output_values": output_values,
                "output_bytes_int8": output_values,
                "relative_macs": 0.0,
            }
        )
    baseline = int(rows[0]["macs"])
    for row in rows:
        row["relative_macs"] = round(int(row["macs"]) / baseline, 3)
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    designs = estimate_designs()
    audit = audit_dataset(
        args.data,
        (16, 20, 24, 32, 48, 64),
        ((24, 18), (20, 16)),
    )
    payload = {"dataset": str(args.data.resolve()), "audit": audit, "designs": designs}
    (args.output / "design_space.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output / "design_space.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=designs[0].keys())
        writer.writeheader()
        writer.writerows(designs)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
