#!/usr/bin/env python3
"""Create a compact, quality-diverse 15 um ROI dataset from nine selected videos.

Every source frame produces five overlapping 256x256 channel tiles, enlarged to
640x640 for annotation. A single mild camera-quality profile is applied to all
five tiles from a frame, avoiding near-duplicate annotation work.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SOURCE_ROOT = Path(r"E:\fpga\data\raw\09_07_2026\09_07_2026")
OUTPUT_ROOT = Path(r"E:\fpga\roboflow_upload\cell_droplet_15um_selected9_diverse_v1")
VIDEO_NAMES = ("4.4.mp4", "2.5.mp4", "3.1.mp4", "3.2.mp4", "3.3.mp4", "3.4.mp4", "3.5.mp4", "4.1.mp4", "4.3.mp4")
SPLITS = {
    "train": {"4.4.mp4", "3.1.mp4", "3.2.mp4", "3.4.mp4", "3.5.mp4", "4.1.mp4", "4.3.mp4"},
    "valid": {"2.5.mp4"},
    "test": {"3.3.mp4"},
}
TILE_XS = (170, 365, 560, 755, 950)
TILE_Y, TILE_SIZE, OUTPUT_SIZE = 342, 256, 640
PROFILES = ("native", "low_light", "high_light", "low_contrast", "soft_blur", "resolution_loss", "sensor_jpeg")


def rng_for(token: str) -> np.random.Generator:
    data = hashlib.sha256(("15072026:" + token).encode("ascii")).digest()
    return np.random.default_rng(int.from_bytes(data[:8], "little"))


def metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": round(float(gray.mean()), 4),
        "contrast": round(float(gray.std()), 4),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
    }


def source_score(frame: np.ndarray) -> float:
    band = frame[TILE_Y:TILE_Y + TILE_SIZE, min(TILE_XS):max(TILE_XS) + TILE_SIZE]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) * max(0.15, 1.0 - float(np.mean(gray >= 252)))


def select_frames(video: Path, count: int = 4) -> tuple[list[int], float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first, last = min(total - 1, round(fps * 2)), max(1, min(total - 1, round(total - fps * 2)))
    selected, used = [], set()
    for target in np.linspace(first, last, count, dtype=int):
        best: tuple[float, int] | None = None
        for index in sorted({max(first, min(last, target + delta)) for delta in range(-18, 19, 3)} | {int(target)}):
            if index in used:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                item = (source_score(frame), index)
                if best is None or item > best:
                    best = item
        if best is None:
            raise RuntimeError(f"No readable frame in {video}")
        selected.append(best[1])
        used.add(best[1])
    cap.release()
    return selected, fps, total


def apply_profile(image: np.ndarray, profile: str, token: str) -> tuple[np.ndarray, dict[str, float | int]]:
    rng = rng_for(token)
    if profile == "native":
        return image.copy(), {}
    if profile in {"low_light", "high_light"}:
        gamma = float(rng.uniform(1.08, 1.22) if profile == "low_light" else rng.uniform(0.86, 0.96))
        gain = float(rng.uniform(0.90, 0.98) if profile == "low_light" else rng.uniform(1.00, 1.06))
        offset = float(rng.uniform(-4.0, 0.0) if profile == "low_light" else rng.uniform(0.0, 5.0))
        out = ((image.astype(np.float32) / 255.0) ** gamma) * 255.0 * gain + offset
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), {"gamma": round(gamma, 4), "gain": round(gain, 4), "offset": round(offset, 3)}
    if profile == "low_contrast":
        factor = float(rng.uniform(0.84, 0.94))
        center = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        out = (image.astype(np.float32) - center) * factor + center
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), {"factor": round(factor, 4)}
    if profile == "soft_blur":
        sigma = float(rng.uniform(0.35, 0.66))
        return cv2.GaussianBlur(image, (3, 3), sigmaX=sigma), {"sigma": round(sigma, 4)}
    if profile == "resolution_loss":
        scale = float(rng.uniform(0.72, 0.88))
        side = max(64, round(OUTPUT_SIZE * scale))
        down = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR), {"scale": round(scale, 4)}
    if profile == "sensor_jpeg":
        sigma, quality = float(rng.uniform(1.5, 3.8)), int(rng.integers(74, 91))
        noisy = np.clip(np.rint(image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)), 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR), {"noise_sigma": round(sigma, 4), "jpeg_quality": quality}
    raise ValueError(profile)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def contact_sheets(paths: list[Path], profile_names: dict[str, str], output: Path, split: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    columns, rows, side, caption = 5, 4, 180, 30
    pages = math.ceil(len(paths) / (columns * rows))
    for page in range(pages):
        sheet = np.full((rows * (side + caption), columns * side, 3), 245, np.uint8)
        for index, path in enumerate(paths[page * columns * rows:(page + 1) * columns * rows]):
            image = cv2.imread(str(path))
            row, col = divmod(index, columns)
            x, y = col * side, row * (side + caption)
            sheet[y:y + side, x:x + side] = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
            cv2.putText(sheet, profile_names[path.name], (x + 3, y + side + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
        cv2.imwrite(str(output / f"{split}_page_{page + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return pages


def main() -> None:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {OUTPUT_ROOT}")
    video_paths = [SOURCE_ROOT / name for name in VIDEO_NAMES]
    missing = [str(path) for path in video_paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing videos:\n" + "\n".join(missing))
    for split in SPLITS:
        (OUTPUT_ROOT / "images_to_label" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "native_reference_do_not_upload" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "selected_full_frames").mkdir(parents=True, exist_ok=True)
    train_groups = [(name, group) for name in SPLITS["train"] for group in range(1, 5)]
    train_profiles = ["native"] * 4 + [profile for profile in PROFILES if profile != "native" for _ in range(4)]
    # 28 train groups: 4 native groups and 4 of each of six quality conditions.
    profile_rng = rng_for("profile_assignment")
    profile_rng.shuffle(train_groups)
    profile_rng.shuffle(train_profiles)
    assignments = dict(zip(train_groups, train_profiles))
    manifest, frame_rows = [], []
    paths_by_split: dict[str, list[Path]] = defaultdict(list)
    profiles_by_name: dict[str, str] = {}
    for video in video_paths:
        split = next(name for name, files in SPLITS.items() if video.name in files)
        selected, fps, total = select_frames(video)
        cap = cv2.VideoCapture(str(video))
        for ordinal, frame_index in enumerate(selected, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read {video} frame {frame_index + 1}")
            token = f"v{video.stem.replace('.', '_')}_src{frame_index + 1:06d}"
            profile = assignments.get((video.name, ordinal), "native")
            params_token = f"{token}:{profile}"
            full_path = OUTPUT_ROOT / "selected_full_frames" / f"{split}_{token}.jpg"
            cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_rows.append({"group_id": token, "split": split, "source_video": video.name, "source_frame": frame_index + 1, "source_time_s": round(frame_index / fps, 5), "total_frames": total, "profile": profile, "source_sharpness": metrics(frame[TILE_Y:TILE_Y + TILE_SIZE, min(TILE_XS):max(TILE_XS) + TILE_SIZE])["sharpness"]})
            for tile_index, x in enumerate(TILE_XS, 1):
                native = cv2.resize(frame[TILE_Y:TILE_Y + TILE_SIZE, x:x + TILE_SIZE], (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_CUBIC)
                transformed, params = apply_profile(native, profile, f"{params_token}:tile{tile_index}")
                file_name = f"{split}_{token}_tile{tile_index:02d}_x{x:04d}_y{TILE_Y:04d}_{profile}.jpg"
                upload = OUTPUT_ROOT / "images_to_label" / split / file_name
                reference = OUTPUT_ROOT / "native_reference_do_not_upload" / split / file_name
                if not cv2.imwrite(str(upload), transformed, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Cannot write {upload}")
                cv2.imwrite(str(reference), native, [cv2.IMWRITE_JPEG_QUALITY, 95])
                paths_by_split[split].append(upload)
                profiles_by_name[file_name] = profile
                manifest.append({"file": file_name, "group_id": token, "split": split, "source_video": video.name, "source_frame": frame_index + 1, "source_time_s": round(frame_index / fps, 5), "tile_index": tile_index, "crop_x": x, "crop_y": TILE_Y, "crop_width": TILE_SIZE, "crop_height": TILE_SIZE, "output_width": OUTPUT_SIZE, "output_height": OUTPUT_SIZE, "quality_profile": profile, "quality_parameters": json.dumps(params, sort_keys=True), "upload_sha256": sha256(upload), "upload_path": str(upload), "native_reference_path": str(reference), **{f"native_{key}": value for key, value in metrics(native).items()}, **{f"upload_{key}": value for key, value in metrics(transformed).items()}})
        cap.release()
    with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    with (OUTPUT_ROOT / "selected_source_frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0])); writer.writeheader(); writer.writerows(frame_rows)
    sheet_counts = {split: contact_sheets(paths, profiles_by_name, OUTPUT_ROOT / "contact_sheets", split) for split, paths in paths_by_split.items()}
    summary = {"particle_size_um": 15, "source_videos": list(VIDEO_NAMES), "videos": len(video_paths), "source_frame_groups": len(frame_rows), "images_total": len(manifest), "split_counts": dict(Counter(row["split"] for row in manifest)), "quality_profile_counts": dict(Counter(row["quality_profile"] for row in manifest)), "geometry": {"tile_xs": list(TILE_XS), "tile_y": TILE_Y, "tile_size": TILE_SIZE, "output_size": OUTPUT_SIZE}, "contact_sheet_pages": sheet_counts, "duplicate_file_hashes": len(manifest) - len({row["upload_sha256"] for row in manifest}), "label_classes": ["droplet", "cell"], "label_class_note": "Use cell for 15 um particle to retain compatibility with earlier project labels."}
    (OUTPUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "README_LABELING.md").write_text("""# Bo du lieu 15 um - 9 video duoc chon

## Upload len Roboflow

Tai rieng tung thu muc trong `images_to_label`: `train` (140 anh), `valid` (20 anh), `test` (20 anh). Khong tai `native_reference_do_not_upload`; day la anh chuan de doi chieu neu anh bien the kho nhin.

## Nhan

- `droplet`: mot box cho moi giot. Box bao tron giot va duoc du ra ngoai vien 2-4 pixel, de khong bo sot particle bam gan vien.
- `cell`: hat 15 um / te bao nhin ro trong giot. Khoanh sat tung hat; khong khoanh vet xuoc, bong phan xa, thanh kenh, bui nen.

Mot giot va particle ben trong no co the chong box. Neu khong chac chan doi tuong la hat, danh dau review/uncertain thay vi doan. Anh khong co doi tuong hop le phai de background/null, khong xoa.

`manifest.csv` luu dung video, frame, ROI, profile va split. Khong doi ten file hoac trao doi anh giua cac split.
""", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
