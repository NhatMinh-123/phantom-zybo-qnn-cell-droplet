"""Package temporal candidate patches into numbered review sheets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a human-review package from candidate track CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "final_results" / "microplastic_active_learning_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "final_results"
        / "microplastic_active_learning_v1"
        / "review_package",
    )
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=96)
    return parser.parse_args()


def draw_label(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def make_tile(
    patch_path: Path,
    *,
    review_id: str,
    confidence: float,
    hits: int,
    suggested: str,
    size: int,
) -> np.ndarray:
    patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
    if patch is None:
        raise RuntimeError(f"Could not read review patch: {patch_path}")
    header_height = 34
    resized = cv2.resize(
        patch,
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )
    tile = np.zeros((size + header_height, size, 3), dtype=np.uint8)
    tile[header_height:] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    color = (0, 220, 0) if suggested == "likely_particle" else (0, 180, 255)
    cv2.rectangle(
        tile,
        (0, header_height),
        (size - 1, size + header_height - 1),
        color,
        2,
    )
    draw_label(
        tile,
        review_id,
        (3, 13),
        color=color,
        scale=0.38,
    )
    draw_label(
        tile,
        f"h{hits} c{confidence:.2f}",
        (3, 29),
        color=(230, 230, 230),
        scale=0.34,
    )
    return tile


def main() -> None:
    args = parse_args()
    input_directory = args.input.expanduser().resolve()
    output_directory = args.output.expanduser().resolve()
    tracks_path = input_directory / "particle_tracks.csv"
    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Review output must be new or empty: {output_directory}"
        )
    if args.columns < 1 or args.rows < 1:
        raise ValueError("columns and rows must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    patch_directory = output_directory / "patches"
    sheet_directory = output_directory / "contact_sheets"
    patch_directory.mkdir()
    sheet_directory.mkdir()

    with tracks_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("review_patch")]
    rows.sort(
        key=lambda row: (
            -int(row["confirmed"]),
            -float(row["confidence"]),
            -int(row["hits"]),
        )
    )

    counters = {"likely_particle": 0, "uncertain": 0}
    review_rows: list[dict[str, object]] = []
    tiles: list[tuple[str, np.ndarray]] = []
    for row in rows:
        suggested = (
            "likely_particle"
            if int(row["confirmed"])
            else "uncertain"
        )
        counters[suggested] += 1
        prefix = "P" if suggested == "likely_particle" else "U"
        review_id = f"{prefix}{counters[suggested]:04d}"
        source_patch = Path(row["review_patch"])
        suffix = source_patch.suffix.lower() or ".png"
        packaged_patch = patch_directory / f"{review_id}{suffix}"
        shutil.copy2(source_patch, packaged_patch)
        confidence = float(row["confidence"])
        hits = int(row["hits"])
        review_rows.append(
            {
                "review_id": review_id,
                "suggested_label": suggested,
                "reviewed_label": "",
                "review_status": "pending",
                "droplet_sequence": row["droplet_sequence"],
                "particle_track": row["particle_track"],
                "first_frame": row["first_frame"],
                "last_frame": row["last_frame"],
                "hits": hits,
                "mean_score": row["mean_score"],
                "max_score": row["max_score"],
                "temporal_confidence": confidence,
                "patch": str(packaged_patch.resolve()),
                "source_patch": str(source_patch.resolve()),
            }
        )
        tiles.append(
            (
                suggested,
                make_tile(
                    packaged_patch,
                    review_id=review_id,
                    confidence=confidence,
                    hits=hits,
                    suggested=suggested,
                    size=args.tile_size,
                ),
            )
        )

    per_page = args.columns * args.rows
    tile_height = args.tile_size + 34
    for page_index in range(math.ceil(len(tiles) / per_page)):
        page_tiles = tiles[
            page_index * per_page : (page_index + 1) * per_page
        ]
        canvas = np.full(
            (
                args.rows * tile_height,
                args.columns * args.tile_size,
                3,
            ),
            32,
            dtype=np.uint8,
        )
        for tile_index, (_, tile) in enumerate(page_tiles):
            row = tile_index // args.columns
            column = tile_index % args.columns
            y = row * tile_height
            x = column * args.tile_size
            canvas[y : y + tile_height, x : x + args.tile_size] = tile
        cv2.imwrite(
            str(
                sheet_directory
                / f"review_page_{page_index + 1:03d}.jpg"
            ),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )

    manifest_path = output_directory / "review_manifest.csv"
    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(review_rows[0]),
        )
        writer.writeheader()
        writer.writerows(review_rows)

    readme = """# Microplastic patch review

This package is an active-learning queue, not ground truth.

1. Open the numbered pages in `contact_sheets`.
2. Enter one value in `reviewed_label` for every useful row:
   `particle`, `background`, or `ignore`.
3. Set `review_status` to `reviewed`.
4. When a patch is ambiguous, inspect the source frame interval shown in the
   manifest before assigning a label.

`likely_particle` means only that the classical response persisted through
multiple frames. It must still be reviewed before QNN fine-tuning.
"""
    (output_directory / "README.md").write_text(
        readme,
        encoding="utf-8",
    )
    summary = {
        "source_tracks": str(tracks_path.resolve()),
        "total_patches": len(review_rows),
        "likely_particle": counters["likely_particle"],
        "uncertain": counters["uncertain"],
        "pages": math.ceil(len(tiles) / per_page),
        "manifest": str(manifest_path.resolve()),
        "policy": (
            "No suggested label is used for training until reviewed_label "
            "and review_status are completed."
        ),
    }
    (output_directory / "review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
