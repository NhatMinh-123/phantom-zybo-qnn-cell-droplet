from __future__ import annotations

import argparse
import json
from pathlib import Path

from qnn.dataset import audit_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO labels for the tiny QNN detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/cell_droplet_yolo_grouped"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qnn_cell_droplet/dataset_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [audit_split(args.data, split) for split in ("train", "valid", "test")]
    payload = {"class_names": ["cell", "droplet"], "splits": [a.to_dict() for a in audits]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for audit in audits:
        print(
            f"{audit.split}: images={audit.images} boxes={audit.boxes} "
            f"class_boxes={audit.boxes_per_class} collisions={audit.collisions_per_grid}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

