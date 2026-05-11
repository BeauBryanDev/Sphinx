#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
TARGET_SIZE = 640


def load_image(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def detect_background_color(image: np.ndarray) -> tuple[int, int, int]:
    border_pixels = np.concatenate(
        [
            image[0, :].reshape(-1, 3),
            image[-1, :].reshape(-1, 3),
            image[:, 0].reshape(-1, 3),
            image[:, -1].reshape(-1, 3),
        ]
    )
    return tuple(int(value) for value in np.median(border_pixels, axis=0))


def resize_with_padding(image: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    height, width = image.shape[:2]

    if height == 0 or width == 0:
        raise ValueError("Invalid image dimensions")

    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    background = detect_background_color(image)
    canvas = np.full((target_size, target_size, 3), background, dtype=np.uint8)

    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    return canvas


def process_folder(input_dir: Path, output_dir: Path, target_size: int) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "saved": 0, "errors": 0}

    for image_path in sorted(input_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        stats["total"] += 1
        image = load_image(image_path)
        if image is None:
            stats["errors"] += 1
            continue

        try:
            resized = resize_with_padding(image, target_size=target_size)
            output_path = output_dir / f"{image_path.stem}.jpg"
            cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["saved"] += 1
        except Exception:
            stats["errors"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize all images in unas_petties to 640x640 with padding."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("unas_petties"),
        help="Input folder containing the source images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unas_petties_640"),
        help="Output folder for the resized images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=TARGET_SIZE,
        help="Final square image size in pixels.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"ERROR: input folder does not exist: {args.input}")

    stats = process_folder(args.input, args.output, args.size)

    print(
        f"Done: {stats['saved']} images saved to {args.output} "
        f"({stats['errors']} errors, {stats['total']} seen)."
    )


if __name__ == "__main__":
    main()
