#!/usr/bin/env python3
"""Prepare a grouped, quality-diverse 15 um dataset for Roboflow labeling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TILE_XS = (170, 365, 560, 755, 950)
TILE_Y = 342
TILE_SIZE = 256
OUTPUT_SIZE = 640
TRAIN_VIDEOS = {
    "1.1.mp4",
    "2.1.mp4",
    "2.2.mp4",
    "2.3.mp4",
    "3.1.mp4",
    "3.2.mp4",
    "3.4.mp4",
    "4.1.mp4",
    "4.2.mp4",
    "4.3.mp4",
    "4.4.mp4",
}
VALID_VIDEOS = {"2.5.mp4", "3.5.mp4"}
TEST_VIDEOS = {"2.4.mp4", "3.3.mp4", "4.5.mp4"}
EXTERNAL_TEST_VIDEOS = {"test.mp4"}
PROFILE_COUNTS_PER_SOURCE_FRAME = {
    "native": 20,
    "low_light": 4,
    "high_light": 4,
    "low_contrast": 4,
    "soft_blur": 4,
    "motion_blur": 4,
    "sensor_jpeg": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--search-window", type=int, default=18)
    parser.add_argument("--candidate-step", type=int, default=3)
    parser.add_argument("--seed", type=int, default=15072026)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def split_for_video(name: str) -> str:
    if name in TRAIN_VIDEOS:
        return "train"
    if name in VALID_VIDEOS:
        return "valid"
    if name in TEST_VIDEOS:
        return "test"
    if name in EXTERNAL_TEST_VIDEOS:
        return "external_test"
    raise ValueError(f"Video {name} has no grouped split assignment")


def stable_rng(seed: int, token: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{token}".encode("ascii")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "clipped_dark": float(np.mean(gray <= 3)),
        "clipped_bright": float(np.mean(gray >= 252)),
    }


def selection_quality(frame: np.ndarray) -> dict[str, float]:
    roi = frame[
        TILE_Y : TILE_Y + TILE_SIZE,
        min(TILE_XS) : max(TILE_XS) + TILE_SIZE,
    ]
    metrics = quality_metrics(roi)
    metrics["score"] = metrics["sharpness"] * max(
        0.15, 1.0 - metrics["clipped_dark"] - metrics["clipped_bright"]
    )
    return metrics


def select_frames(
    video_path: Path,
    count: int,
    search_window: int,
    candidate_step: int,
) -> tuple[list[tuple[int, dict[str, float]]], float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first = min(total_frames - 1, max(0, round(2.0 * fps)))
    last = max(first + 1, min(total_frames - 1, round(total_frames - 2.0 * fps)))
    targets = np.linspace(first, last, count, dtype=int).tolist()
    selected: list[tuple[int, dict[str, float]]] = []
    used: set[int] = set()
    for target in targets:
        candidates = sorted(
            {
                min(last, max(first, target + offset))
                for offset in range(-search_window, search_window + 1, candidate_step)
            }
            | {target}
        )
        scored: list[tuple[float, int, dict[str, float]]] = []
        for frame_index in candidates:
            if frame_index in used:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            metrics = selection_quality(frame)
            scored.append((metrics["score"], frame_index, metrics))
        if not scored:
            raise RuntimeError(f"No readable candidates around frame {target + 1} in {video_path}")
        _, best_index, best_metrics = max(scored, key=lambda item: item[0])
        selected.append((best_index, best_metrics))
        used.add(best_index)
    capture.release()
    return selected, fps, total_frames


def clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def gamma_adjust(image: np.ndarray, gamma: float) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    return np.power(normalized, gamma) * 255.0


def profile_parameters(profile: str, rng: np.random.Generator) -> dict[str, Any]:
    if profile == "native":
        return {}
    if profile == "low_light":
        return {
            "gamma": float(rng.uniform(1.08, 1.22)),
            "gain": float(rng.uniform(0.90, 0.98)),
            "offset": float(rng.uniform(-4.0, 0.0)),
        }
    if profile == "high_light":
        return {
            "gamma": float(rng.uniform(0.86, 0.96)),
            "gain": float(rng.uniform(1.00, 1.06)),
            "offset": float(rng.uniform(0.0, 5.0)),
        }
    if profile == "low_contrast":
        return {
            "factor": float(rng.uniform(0.82, 0.93)),
            "gamma": float(rng.uniform(0.97, 1.05)),
        }
    if profile == "soft_blur":
        return {"sigma": float(rng.uniform(0.35, 0.68))}
    if profile == "motion_blur":
        return {"length": 3, "angle": float(rng.uniform(-12.0, 12.0))}
    if profile == "sensor_jpeg":
        return {
            "noise_sigma": float(rng.uniform(1.5, 4.0)),
            "jpeg_quality": int(rng.integers(72, 91)),
        }
    raise ValueError(f"Unknown quality profile {profile}")


def motion_kernel(length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(
        (length / 2 - 0.5, length / 2 - 0.5), angle, 1.0
    )
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    return kernel / max(float(kernel.sum()), 1e-9)


def jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def apply_profile(
    image: np.ndarray,
    profile: str,
    parameters: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    if profile == "native":
        return image.copy()
    if profile in {"low_light", "high_light"}:
        adjusted = gamma_adjust(image, float(parameters["gamma"]))
        adjusted = adjusted * float(parameters["gain"]) + float(parameters["offset"])
        return clip_uint8(adjusted)
    if profile == "low_contrast":
        center = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        adjusted = (image.astype(np.float32) - center) * float(parameters["factor"]) + center
        return clip_uint8(gamma_adjust(clip_uint8(adjusted), float(parameters["gamma"])))
    if profile == "soft_blur":
        return cv2.GaussianBlur(image, (3, 3), sigmaX=float(parameters["sigma"]))
    if profile == "motion_blur":
        return cv2.filter2D(
            image,
            -1,
            motion_kernel(int(parameters["length"]), float(parameters["angle"])),
            borderType=cv2.BORDER_REPLICATE,
        )
    if profile == "sensor_jpeg":
        noise = rng.normal(0.0, float(parameters["noise_sigma"]), image.shape)
        noisy = clip_uint8(image.astype(np.float32) + noise)
        return jpeg_round_trip(noisy, int(parameters["jpeg_quality"]))
    raise ValueError(f"Unknown quality profile {profile}")


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read source frame {frame_index + 1}")
    return frame


def make_contact_pages(
    paths: list[Path],
    profile_by_name: dict[str, str],
    output: Path,
    prefix: str,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    columns = 5
    rows = 4
    tile = 210
    caption_height = 34
    per_page = columns * rows
    page_count = math.ceil(len(paths) / per_page)
    for page_index in range(page_count):
        page_paths = paths[page_index * per_page : (page_index + 1) * per_page]
        sheet = np.full(
            (rows * (tile + caption_height), columns * tile, 3), 245, dtype=np.uint8
        )
        for index, path in enumerate(page_paths):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read {path}")
            thumb = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
            row, column = divmod(index, columns)
            x = column * tile
            y = row * (tile + caption_height)
            sheet[y : y + tile, x : x + tile] = thumb
            caption = f"{path.stem[:27]} | {profile_by_name[path.name]}"
            cv2.putText(
                sheet,
                caption,
                (x + 3, y + tile + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(
            str(output / f"{prefix}_page_{page_index + 1:03d}.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
    return page_count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_train_profiles(
    videos: list[Path], frames_per_video: int, seed: int
) -> dict[tuple[str, int], str]:
    groups = [
        (video.name, selection_index)
        for video in videos
        if split_for_video(video.name) == "train"
        for selection_index in range(1, frames_per_video + 1)
    ]
    profiles = [
        profile
        for profile, count in PROFILE_COUNTS_PER_SOURCE_FRAME.items()
        for _ in range(count)
    ]
    if len(groups) != len(profiles):
        raise ValueError(
            "Profile distribution expects exactly 44 train source-frame groups; "
            f"found {len(groups)}"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    rng.shuffle(profiles)
    return dict(zip(groups, profiles))


def main() -> None:
    args = parse_args()
    videos_root = args.videos.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    upload_root = output / "images_to_upload"
    native_root = output / "native_reference_do_not_upload"
    full_root = output / "selected_full_frames"
    contact_root = output / "contact_sheets"
    for split in ("train", "valid", "test", "external_test"):
        (upload_root / split).mkdir(parents=True, exist_ok=True)
        (native_root / split).mkdir(parents=True, exist_ok=True)
    full_root.mkdir(parents=True, exist_ok=True)
    contact_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_root.glob("*.mp4"))
    expected = TRAIN_VIDEOS | VALID_VIDEOS | TEST_VIDEOS | EXTERNAL_TEST_VIDEOS
    if {path.name for path in videos} != expected:
        raise ValueError("Video set does not match the expected 17-file archive")
    profile_assignments = assign_train_profiles(videos, args.frames_per_video, args.seed)
    manifest: list[dict[str, object]] = []
    frame_manifest: list[dict[str, object]] = []
    profile_by_name: dict[str, str] = {}
    upload_paths: dict[str, list[Path]] = defaultdict(list)
    encode = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    for video_path in videos:
        split = split_for_video(video_path.name)
        selected, fps, total_frames = select_frames(
            video_path,
            args.frames_per_video,
            args.search_window,
            args.candidate_step,
        )
        capture = cv2.VideoCapture(str(video_path))
        video_token = video_path.stem.replace(".", "_")
        for selection_index, (frame_index, frame_metrics) in enumerate(selected, start=1):
            frame = read_frame(capture, frame_index)
            group_id = f"v{video_token}_src{frame_index + 1:06d}"
            profile = (
                profile_assignments[(video_path.name, selection_index)]
                if split == "train"
                else "native"
            )
            profile_rng = stable_rng(args.seed, f"{group_id}:{profile}:parameters")
            parameters = profile_parameters(profile, profile_rng)
            full_path = full_root / f"{split}_{group_id}.jpg"
            cv2.imwrite(str(full_path), frame, encode)
            frame_manifest.append(
                {
                    "group_id": group_id,
                    "split": split,
                    "source_video": video_path.name,
                    "source_frame": frame_index + 1,
                    "source_time_s": frame_index / fps,
                    "video_total_frames": total_frames,
                    "quality_profile": profile,
                    "quality_parameters": json.dumps(parameters, sort_keys=True),
                    **{f"source_{key}": value for key, value in frame_metrics.items()},
                }
            )
            for tile_index, tile_x in enumerate(TILE_XS, start=1):
                native = frame[
                    TILE_Y : TILE_Y + TILE_SIZE,
                    tile_x : tile_x + TILE_SIZE,
                ]
                native = cv2.resize(
                    native,
                    (OUTPUT_SIZE, OUTPUT_SIZE),
                    interpolation=cv2.INTER_CUBIC,
                )
                tile_rng = stable_rng(args.seed, f"{group_id}:tile{tile_index}:{profile}")
                upload = apply_profile(native, profile, parameters, tile_rng)
                filename = (
                    f"{split}_{group_id}_tile{tile_index:02d}_"
                    f"x{tile_x:04d}_y{TILE_Y:04d}_{profile}.jpg"
                )
                upload_path = upload_root / split / filename
                native_path = native_root / split / filename
                if not cv2.imwrite(str(upload_path), upload, encode):
                    raise RuntimeError(f"Could not write {upload_path}")
                if not cv2.imwrite(str(native_path), native, encode):
                    raise RuntimeError(f"Could not write {native_path}")
                upload_paths[split].append(upload_path)
                profile_by_name[filename] = profile
                native_metrics = quality_metrics(native)
                upload_metrics = quality_metrics(upload)
                manifest.append(
                    {
                        "file": filename,
                        "group_id": group_id,
                        "split": split,
                        "source_video": video_path.name,
                        "source_frame": frame_index + 1,
                        "source_time_s": frame_index / fps,
                        "selection_index": selection_index,
                        "tile_index": tile_index,
                        "crop_x": tile_x,
                        "crop_y": TILE_Y,
                        "crop_width": TILE_SIZE,
                        "crop_height": TILE_SIZE,
                        "output_width": OUTPUT_SIZE,
                        "output_height": OUTPUT_SIZE,
                        "quality_profile": profile,
                        "quality_parameters": json.dumps(parameters, sort_keys=True),
                        **{f"native_{key}": value for key, value in native_metrics.items()},
                        **{f"upload_{key}": value for key, value in upload_metrics.items()},
                        "upload_sha256": sha256(upload_path),
                        "upload_path": str(upload_path),
                        "native_reference_path": str(native_path),
                    }
                )
        capture.release()

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (output / "selected_source_frames.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_manifest[0]))
        writer.writeheader()
        writer.writerows(frame_manifest)

    contact_pages = {
        split: make_contact_pages(paths, profile_by_name, contact_root, split)
        for split, paths in upload_paths.items()
    }
    split_counts = Counter(str(row["split"]) for row in manifest)
    profile_counts = Counter(str(row["quality_profile"]) for row in manifest)
    video_counts = Counter(str(row["source_video"]) for row in manifest)
    hashes = [str(row["upload_sha256"]) for row in manifest]
    duplicate_hashes = len(hashes) - len(set(hashes))
    summary = {
        "particle_size_um": 15,
        "images_total": len(manifest),
        "source_frame_groups": len(frame_manifest),
        "videos": len(videos),
        "split_counts": dict(split_counts),
        "quality_profile_counts": dict(profile_counts),
        "video_counts": dict(video_counts),
        "contact_sheet_pages": contact_pages,
        "geometry": {
            "tile_xs": list(TILE_XS),
            "tile_y": TILE_Y,
            "tile_size": TILE_SIZE,
            "output_size": OUTPUT_SIZE,
        },
        "duplicate_file_hashes": duplicate_hashes,
        "split_policy": "Grouped by complete source video; test.mp4 is external test only.",
        "augmentation_policy": "Mild deterministic quality variants on train only.",
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "README_LABELING.md").write_text(
        """# Hướng dẫn gán nhãn bộ 15 µm

## Thư mục cần tải lên Roboflow

- `images_to_upload/train`: chọn đưa toàn bộ vào Train.
- `images_to_upload/valid`: chọn đưa toàn bộ vào Valid.
- `images_to_upload/test`: chọn đưa toàn bộ vào Test.
- `images_to_upload/external_test`: chỉ dùng đánh giá độc lập, không dùng huấn luyện.
- Không tải thư mục `native_reference_do_not_upload`; đây chỉ là ảnh tham chiếu khi ảnh biến thể khó nhìn.

## Hai lớp nhãn

1. `droplet`: một box cho mỗi giọt có đường biên ngoài khép kín và nhìn rõ. Box phủ hết viền tối, có thể dư 2-4 pixel nhưng không gộp hai giọt khác nhau.
2. `cell`: box sát từng hạt/tế bào 15 µm nhìn thấy rõ. Không khoanh bụi, vết xước, thành kênh hoặc bóng phản xạ.

## Quy tắc quan trọng

- Một giọt và cell bên trong được phép có box chồng nhau.
- Giọt không chứa cell vẫn phải gán `droplet`.
- Nếu hạt bị mờ đến mức không chắc chắn, dùng Review/Uncertain thay vì đoán.
- Ảnh không có đối tượng hợp lệ phải để Null/Background; không xóa ảnh đó.
- Không sửa crop, không xoay ảnh và không đổi tên file.
- Gán nhãn nhất quán cả viền trái/phải; đối tượng bị cắt quá nửa ở mép ảnh thì bỏ qua.

`manifest.csv` giữ quan hệ video, frame, tile, profile chất lượng và split. Sau khi xuất COCO/YOLO, dùng manifest này để kiểm tra không rò rỉ giữa các split.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
