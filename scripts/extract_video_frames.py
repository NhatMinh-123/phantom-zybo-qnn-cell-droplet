import argparse
import csv
from pathlib import Path

import cv2


def parse_size(value):
    if value is None:
        return None
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must look like WIDTHxHEIGHT, for example 160x120.")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive.")
    return width, height


def parse_args():
    parser = argparse.ArgumentParser(description="Extract standard image frames from a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Output folder for extracted images.")
    parser.add_argument("--metadata", default=None, help="Optional metadata CSV path.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Evenly sample up to this many frames. Use 0 to export every selected frame.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Only consider every Nth source frame before max-frame sampling.",
    )
    parser.add_argument("--resize", type=parse_size, default=None, help="Optional output size, for example 160x120.")
    parser.add_argument("--gray", action="store_true", help="Save grayscale images.")
    parser.add_argument("--ext", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument("--clean", action="store_true", help="Delete existing images in output-dir first.")
    return parser.parse_args()


def choose_indices(total_frames, every_n, max_frames):
    candidates = list(range(0, total_frames, every_n))
    if max_frames <= 0 or max_frames >= len(candidates):
        return candidates
    if max_frames == 1:
        return [candidates[len(candidates) // 2]]

    last = len(candidates) - 1
    selected = []
    seen = set()
    for i in range(max_frames):
        index = candidates[round(i * last / (max_frames - 1))]
        if index not in seen:
            selected.append(index)
            seen.add(index)
    return selected


def clean_output(output_dir, ext):
    for path in output_dir.glob(f"*.{ext}"):
        path.unlink()


def main():
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    metadata_path = Path(args.metadata) if args.metadata else output_dir / "metadata.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise SystemExit("Could not read frame count from video.")
    if args.every_n <= 0:
        raise SystemExit("--every-n must be >= 1")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_output(output_dir, args.ext)

    prefix = args.prefix or video_path.stem.replace(".", "_")
    indices = choose_indices(total_frames, args.every_n, args.max_frames)
    index_set = set(indices)
    rows = []
    written = 0
    source_index = 0

    encode_params = []
    if args.ext == "jpg":
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if source_index in index_set:
            output = frame
            if args.gray:
                output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            if args.resize:
                output = cv2.resize(output, args.resize, interpolation=cv2.INTER_AREA)

            written += 1
            output_name = f"{prefix}_src{source_index + 1:06d}_img{written:04d}.{args.ext}"
            output_path = output_dir / output_name
            if not cv2.imwrite(str(output_path), output, encode_params):
                raise SystemExit(f"Failed to write {output_path}")

            out_height, out_width = output.shape[:2]
            rows.append(
                {
                    "output_file": output_name,
                    "source_video": str(video_path),
                    "source_frame": source_index + 1,
                    "time_sec": f"{source_index / fps:.6f}" if fps > 0 else "",
                    "source_width": source_width,
                    "source_height": source_height,
                    "output_width": out_width,
                    "output_height": out_height,
                    "gray": int(args.gray),
                }
            )

            if written % 50 == 0 or written == len(indices):
                print(f"written {written}/{len(indices)}")

        source_index += 1

    cap.release()

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "output_file",
            "source_video",
            "source_frame",
            "time_sec",
            "source_width",
            "source_height",
            "output_width",
            "output_height",
            "gray",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"video: {video_path}")
    print(f"source: {source_width}x{source_height}, fps={fps:.3f}, frames={total_frames}")
    print(f"images: {written} -> {output_dir.resolve()}")
    print(f"metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
