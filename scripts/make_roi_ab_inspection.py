import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side ROI overlay crops.")
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, default=(500, 300, 860, 620))
    parser.add_argument("--scale", type=float, default=0.85)
    return parser.parse_args()


def load_crop(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    x1, y1, x2, y2 = crop
    return image[y1:y2, x1:x2]


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (12, 12, 12), -1)
    cv2.putText(
        output,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop = tuple(args.crop)

    for frame_index in args.frames:
        name = f"frame_{frame_index:06d}.jpg"
        left = add_title(load_crop(args.left_dir / name, crop), "ROI 256, no motion")
        right = add_title(load_crop(args.right_dir / name, crop), "ROI 240, no motion")
        comparison = np.hstack((left, right))
        if args.scale != 1.0:
            comparison = cv2.resize(
                comparison,
                None,
                fx=args.scale,
                fy=args.scale,
                interpolation=cv2.INTER_AREA,
            )
        output = args.output_dir / name
        if not cv2.imwrite(
            str(output), comparison, [cv2.IMWRITE_JPEG_QUALITY, 70]
        ):
            raise RuntimeError(f"Cannot write comparison: {output}")
        print(output)


if __name__ == "__main__":
    main()
