#!/usr/bin/env python3
"""Create 900 flat ROI images for annotation before any train/valid/test split."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from prepare_15um_selected9_labeling_dataset import (
    OUTPUT_SIZE,
    SOURCE_ROOT,
    TILE_SIZE,
    TILE_XS,
    TILE_Y,
    VIDEO_NAMES,
    apply_profile,
    metrics,
    rng_for,
)


OUTPUT_ROOT = Path(r"E:\fpga\roboflow_upload\cell_droplet_15um_selected9_label_all_900_v1")
FRAMES_PER_VIDEO = 20
PROFILES = ("native", "low_light", "high_light", "low_contrast", "soft_blur", "resolution_loss", "sensor_jpeg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_even_frames(video: Path, count: int) -> tuple[list[int], float, int]:
    """Use widely separated frame times; image conditions are augmented separately."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = min(total - 1, max(0, round(2.0 * fps)))
    last = max(first + 1, min(total - 1, round(total - 2.0 * fps)))
    cap.release()
    return np.linspace(first, last, count, dtype=int).tolist(), fps, total

def make_contact_sheets(paths: list[Path], profiles: dict[str, str], out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    columns, rows, side, caption = 6, 4, 160, 26
    per_page = columns * rows
    pages = math.ceil(len(paths) / per_page)
    for page_index in range(pages):
        page = np.full((rows * (side + caption), columns * side, 3), 245, np.uint8)
        for index, path in enumerate(paths[page_index * per_page:(page_index + 1) * per_page]):
            image = cv2.imread(str(path))
            row, col = divmod(index, columns)
            x, y = col * side, row * (side + caption)
            page[y:y + side, x:x + side] = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
            cv2.putText(page, profiles[path.name], (x + 3, y + side + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (18, 18, 18), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"all_images_page_{page_index + 1:03d}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return pages


def main() -> None:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing folder: {OUTPUT_ROOT}")
    videos = [SOURCE_ROOT / name for name in VIDEO_NAMES]
    missing = [str(path) for path in videos if not path.is_file()]
    if missing:
        raise SystemExit("Missing videos:\n" + "\n".join(missing))

    images_root = OUTPUT_ROOT / "images_to_label"
    references_root = OUTPUT_ROOT / "native_reference_do_not_upload"
    full_root = OUTPUT_ROOT / "selected_full_frames"
    for folder in (images_root, references_root, full_root):
        folder.mkdir(parents=True, exist_ok=True)

    # 180 source-frame groups: 60 native and 20 groups per other profile.
    groups = [(video.name, frame_ordinal) for video in videos for frame_ordinal in range(1, FRAMES_PER_VIDEO + 1)]
    profile_list = ["native"] * 60 + [profile for profile in PROFILES if profile != "native" for _ in range(20)]
    group_rng = rng_for("flat_900_profile_assignment")
    group_rng.shuffle(groups)
    group_rng.shuffle(profile_list)
    profile_assignment = dict(zip(groups, profile_list))

    rows, frame_rows, paths, profiles = [], [], [], {}
    for video in videos:
        selected, fps, total = select_even_frames(video, count=FRAMES_PER_VIDEO)
        cap = cv2.VideoCapture(str(video))
        for ordinal, frame_index in enumerate(selected, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read {video} frame {frame_index + 1}")
            group_id = f"v{video.stem.replace('.', '_')}_src{frame_index + 1:06d}"
            profile = profile_assignment[(video.name, ordinal)]
            full_path = full_root / f"{group_id}.jpg"
            cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_rows.append({"group_id": group_id, "source_video": video.name, "source_frame": frame_index + 1, "source_time_s": round(frame_index / fps, 5), "total_frames": total, "quality_profile": profile})
            for tile_index, x in enumerate(TILE_XS, 1):
                native = frame[TILE_Y:TILE_Y + TILE_SIZE, x:x + TILE_SIZE]
                native = cv2.resize(native, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_CUBIC)
                transformed, params = apply_profile(native, profile, f"flat900:{group_id}:tile{tile_index}:{profile}")
                filename = f"{group_id}_tile{tile_index:02d}_x{x:04d}_y{TILE_Y:04d}_{profile}.jpg"
                destination, reference = images_root / filename, references_root / filename
                if not cv2.imwrite(str(destination), transformed, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Cannot write {destination}")
                cv2.imwrite(str(reference), native, [cv2.IMWRITE_JPEG_QUALITY, 95])
                paths.append(destination)
                profiles[filename] = profile
                rows.append({"file": filename, "group_id": group_id, "source_video": video.name, "source_frame": frame_index + 1, "source_time_s": round(frame_index / fps, 5), "tile_index": tile_index, "crop_x": x, "crop_y": TILE_Y, "crop_width": TILE_SIZE, "crop_height": TILE_SIZE, "output_width": OUTPUT_SIZE, "output_height": OUTPUT_SIZE, "quality_profile": profile, "quality_parameters": json.dumps(params, sort_keys=True), "upload_sha256": sha256(destination), "upload_path": str(destination), "native_reference_path": str(reference), **{f"native_{key}": value for key, value in metrics(native).items()}, **{f"upload_{key}": value for key, value in metrics(transformed).items()}})
        cap.release()

    with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUTPUT_ROOT / "selected_source_frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0])); writer.writeheader(); writer.writerows(frame_rows)
    summary = {"particle_size_um": 15, "source_videos": list(VIDEO_NAMES), "source_frame_groups": len(frame_rows), "images_total": len(rows), "quality_profile_counts": dict(Counter(row["quality_profile"] for row in rows)), "geometry": {"tile_xs": list(TILE_XS), "tile_y": TILE_Y, "tile_size": TILE_SIZE, "output_size": OUTPUT_SIZE}, "duplicate_file_hashes": len(rows) - len({row["upload_sha256"] for row in rows}), "split_status": "Not split. Label all images first; split later by complete source video.", "label_classes": ["droplet", "cell"]}
    summary["contact_sheet_pages"] = make_contact_sheets(paths, profiles, OUTPUT_ROOT / "contact_sheets")
    (OUTPUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "README_LABELING.md").write_text("""# Dataset 15 um - label truoc, chia split sau

Tat ca 900 anh can label nam trong `images_to_label`. Upload ca thu muc nay vao mot annotation job Roboflow; chua can chon Train/Valid/Test o buoc nay.

Nhan: `droplet` cho toan bo giot, box du ra 2-4 pixel ngoai vien de khong bo sot hat bam sat bien; `cell` cho tung hat 15 um ro rang trong giot. Khong label thanh kenh, bong phan xa, vet xuoc hay nhieu nen. Duoc phep co box cell nam trong box droplet.

Thu muc `native_reference_do_not_upload` la ban goc cua cung ROI, chi dung doi chieu khi anh da bi thay doi sang/toi/mo. Sau khi label xong, dung `manifest.csv` de chia theo ca video nguon, tranh frame cung video nam ca train va test.
""", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
