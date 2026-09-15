import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_int_list(value):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select sharp video frames and export square downstream-channel tiles for labeling."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="New output directory.")
    parser.add_argument("--frame-count", type=int, default=40, help="Number of source frames to select.")
    parser.add_argument("--start-sec", type=float, default=2.0, help="Ignore video before this time.")
    parser.add_argument("--end-sec", type=float, default=None, help="Ignore video after this time.")
    parser.add_argument(
        "--search-window",
        type=int,
        default=15,
        help="Search this many source frames around each evenly spaced target.",
    )
    parser.add_argument(
        "--candidate-step",
        type=int,
        default=3,
        help="Evaluate every Nth frame inside each search window.",
    )
    parser.add_argument("--tile-xs", type=parse_int_list, default=[170, 365, 560, 755, 950])
    parser.add_argument("--tile-y", type=int, default=342)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=640)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def evenly_spaced_indices(first, last, count):
    if count <= 0:
        raise ValueError("frame-count must be positive")
    if count == 1:
        return [(first + last) // 2]
    return [int(round(value)) for value in np.linspace(first, last, count)]


def quality_metrics(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    clipped = float(np.mean((gray <= 4) | (gray >= 251)))
    score = sharpness * max(0.1, 1.0 - clipped)
    return {
        "score": score,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "clipped_fraction": clipped,
    }


def make_contact_sheet(paths, output_path, columns, rows, tile_size):
    if not paths:
        return
    count = min(len(paths), columns * rows)
    picks = evenly_spaced_indices(0, len(paths) - 1, count)
    selected = [paths[index] for index in picks]
    label_height = 18
    pad = 4
    canvas = Image.new(
        "RGB",
        (columns * tile_size + (columns + 1) * pad, rows * (tile_size + label_height) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, path in enumerate(selected):
        row = index // columns
        col = index % columns
        x = pad + col * (tile_size + pad)
        y = pad + row * (tile_size + label_height + pad)
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        px = x + (tile_size - image.width) // 2
        py = y + (tile_size - image.height) // 2
        canvas.paste(image, (px, py))
        draw.text((x, y + tile_size + 2), path.stem[:44], fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def validate_geometry(width, height, tile_xs, tile_y, tile_size):
    if tile_y < 0 or tile_y + tile_size > height:
        raise SystemExit("Tile y range lies outside the video frame.")
    for x in tile_xs:
        if x < 0 or x + tile_size > width:
            raise SystemExit(f"Tile x range {x}:{x + tile_size} lies outside the video frame.")


def select_frames(cap, targets, first_frame, last_frame, args, quality_bounds):
    target_candidates = []
    all_candidates = set()
    for target in targets:
        candidates = sorted(
            {
                min(last_frame, max(first_frame, target + offset))
                for offset in range(-args.search_window, args.search_window + 1, args.candidate_step)
            }
            | {target}
        )
        target_candidates.append(candidates)
        all_candidates.update(candidates)

    metrics = {}
    source_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if source_index in all_candidates:
            metrics[source_index] = quality_metrics(frame, *quality_bounds)
        if source_index > max(all_candidates):
            break
        source_index += 1

    selected = []
    used = set()
    for target, candidates in zip(targets, target_candidates):
        available = [index for index in candidates if index in metrics and index not in used]
        if not available:
            raise SystemExit(f"No readable candidate frames around source frame {target + 1}.")
        best = max(available, key=lambda index: metrics[index]["score"])
        selected.append((target, best, metrics[best]))
        used.add(best)
    return selected


def read_frame(cap, zero_based_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based_index)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read source frame {zero_based_index + 1}.")
    return frame


def main():
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "selected_frames"
    images_dir = output_dir / "roboflow_images"
    previews_dir = output_dir / "previews"

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        raise SystemExit("Video metadata is incomplete.")
    validate_geometry(width, height, args.tile_xs, args.tile_y, args.tile_size)

    duration = total_frames / fps
    end_sec = args.end_sec if args.end_sec is not None else max(args.start_sec, duration - 2.0)
    first_frame = max(0, int(round(args.start_sec * fps)))
    last_frame = min(total_frames - 1, int(round(end_sec * fps)))
    if first_frame >= last_frame:
        raise SystemExit("The selected time range is empty.")

    targets = evenly_spaced_indices(first_frame, last_frame, args.frame_count)
    quality_bounds = (
        min(args.tile_xs),
        args.tile_y,
        max(args.tile_xs) + args.tile_size,
        args.tile_y + args.tile_size,
    )
    selected = select_frames(cap, targets, first_frame, last_frame, args, quality_bounds)
    cap.release()

    prefix = args.prefix or video_path.stem.replace(".", "_")
    frame_rows = []
    tile_rows = []
    frame_paths = []
    tile_paths = []
    cap = cv2.VideoCapture(str(video_path))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality]
    for selection_index, (target, source_index, metrics) in enumerate(selected, start=1):
        frame = read_frame(cap, source_index)
        frame_name = f"{prefix}_src{source_index + 1:06d}_sel{selection_index:03d}.jpg"
        frame_path = frames_dir / frame_name
        if not cv2.imwrite(str(frame_path), frame, encode_params):
            raise SystemExit(f"Failed to write {frame_path}")
        frame_paths.append(frame_path)
        frame_rows.append(
            {
                "selection_index": selection_index,
                "target_frame": target + 1,
                "source_frame": source_index + 1,
                "time_sec": f"{source_index / fps:.6f}",
                **{key: f"{value:.6f}" for key, value in metrics.items()},
                "file": frame_name,
            }
        )

        for tile_index, tile_x in enumerate(args.tile_xs, start=1):
            crop = frame[
                args.tile_y : args.tile_y + args.tile_size,
                tile_x : tile_x + args.tile_size,
            ]
            output = cv2.resize(
                crop,
                (args.output_size, args.output_size),
                interpolation=cv2.INTER_CUBIC,
            )
            tile_name = (
                f"{prefix}_src{source_index + 1:06d}_tile{tile_index:02d}"
                f"_x{tile_x:04d}_y{args.tile_y:04d}.jpg"
            )
            tile_path = images_dir / tile_name
            if not cv2.imwrite(str(tile_path), output, encode_params):
                raise SystemExit(f"Failed to write {tile_path}")
            tile_paths.append(tile_path)
            tile_rows.append(
                {
                    "source_frame": source_index + 1,
                    "time_sec": f"{source_index / fps:.6f}",
                    "tile_index": tile_index,
                    "crop_x": tile_x,
                    "crop_y": args.tile_y,
                    "crop_width": args.tile_size,
                    "crop_height": args.tile_size,
                    "output_width": args.output_size,
                    "output_height": args.output_size,
                    "file": tile_name,
                }
            )
    cap.release()

    with (output_dir / "selected_frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    with (output_dir / "tiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tile_rows[0]))
        writer.writeheader()
        writer.writerows(tile_rows)

    make_contact_sheet(frame_paths, previews_dir / "selected_frames_overview.jpg", 5, 4, 256)
    make_contact_sheet(tile_paths, previews_dir / "roboflow_tiles_overview.jpg", 5, 4, 256)
    (output_dir / "README.txt").write_text(
        "Roboflow upload folder: roboflow_images\n"
        f"Selected source frames: {len(frame_paths)}\n"
        f"Training tiles: {len(tile_paths)}\n"
        "Project type: Object Detection\n"
        "Classes: droplet, cell\n"
        "Draw one droplet box around each distinct outer droplet boundary.\n"
        "Draw one tight cell box around every clearly visible cell.\n"
        "A droplet without a cell still needs a droplet box.\n"
        "Do not label channel edges or debris. Keep true empty tiles as null/background images.\n",
        encoding="utf-8",
    )

    sharpness_values = [float(row["sharpness"]) for row in frame_rows]
    print(f"video: {video_path}")
    print(f"source: {width}x{height}, fps={fps:.3f}, duration={duration:.2f}s")
    print(f"selected frames: {len(frame_paths)} -> {frames_dir.resolve()}")
    print(f"Roboflow tiles: {len(tile_paths)} -> {images_dir.resolve()}")
    print(
        "selected sharpness: "
        f"min={min(sharpness_values):.2f}, median={float(np.median(sharpness_values)):.2f}, "
        f"max={max(sharpness_values):.2f}"
    )
    print(f"previews: {previews_dir.resolve()}")


if __name__ == "__main__":
    main()
