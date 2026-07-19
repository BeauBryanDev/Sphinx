#!/usr/bin/env python3
"""
Create random mosaic images from a YOLO train split.

Default behavior:
  - source: SphinxProjectV2/images/train
  - output: SphinxProjectV2/mosaic/images
  - grid:   5x5
  - canvas: 1280x1280

Because 5 * 224 = 1120, the script resizes each tile to 256x256 so the
final canvas is exactly 1280x1280.

Dependencies:
  pip install opencv-python pillow numpy
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(source: Path) -> list[Path]:
    """Collect image files recursively from source."""
    return sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXT
    )


def load_image(path: Path) -> np.ndarray | None:
    """Load an image with OpenCV in BGR format."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def resize_tile(img: np.ndarray, tile_size: int) -> np.ndarray:
    """Resize one tile to tile_size x tile_size."""
    return cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)


def build_mosaic(image_paths: list[Path], grid: int, tile_size: int, rng: random.Random) -> Image.Image:
    """Build one mosaic image from random source files."""
    canvas_size = grid * tile_size
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

    chosen = [rng.choice(image_paths) for _ in range(grid * grid)]

    for idx, img_path in enumerate(chosen):
        img = load_image(img_path)
        if img is None:
            continue

        tile = resize_tile(img, tile_size)
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile_img = Image.fromarray(tile)

        row = idx // grid
        col = idx % grid
        x = col * tile_size
        y = row * tile_size
        canvas.paste(tile_img, (x, y))

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 1280x1280 mosaic images from random train images"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./SphinxProjectV2/images/train",
        help="Input directory containing train images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./SphinxProjectV2/mosaic/images",
        help="Output directory for generated mosaic images"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of mosaic images to create"
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=5,
        help="Grid size for the mosaic, default: 5"
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=1280,
        help="Final canvas size in pixels, default: 1280"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible mosaics"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="mosaic",
        help="Filename prefix for output images"
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise SystemExit(f"ERROR: source directory not found: {source}")

    if args.canvas_size % args.grid != 0:
        raise SystemExit("ERROR: canvas-size must be divisible by grid")

    image_paths = collect_images(source)
    if not image_paths:
        raise SystemExit(f"ERROR: no images found in {source}")

    tile_size = args.canvas_size // args.grid
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print(f"Source      : {source}")
    print(f"Output      : {output}")
    print(f"Images found: {len(image_paths)}")
    print(f"Grid        : {args.grid}x{args.grid}")
    print(f"Tile size   : {tile_size}x{tile_size}")
    print(f"Canvas      : {args.canvas_size}x{args.canvas_size}")
    print(f"Count       : {args.count}")
    print(f"Seed        : {args.seed}")

    for idx in range(1, args.count + 1):
        mosaic = build_mosaic(image_paths, args.grid, tile_size, rng)
        out_path = output / f"{args.prefix}_{idx:04d}.png"
        mosaic.save(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
