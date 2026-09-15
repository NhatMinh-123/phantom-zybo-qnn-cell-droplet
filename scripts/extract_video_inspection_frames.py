import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selected frames from a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = set(args.frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    saved = []
    frame_index = 0
    while targets:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in targets:
            output = args.output_dir / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"Cannot write frame: {output}")
            saved.append(output)
            targets.remove(frame_index)
        frame_index += 1
    capture.release()

    if targets:
        missing = ", ".join(str(index) for index in sorted(targets))
        raise RuntimeError(f"Video ended before frames: {missing}")

    for output in saved:
        print(output)


if __name__ == "__main__":
    main()
