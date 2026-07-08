#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_pairs(source_images: Path, source_labels: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    for image_path in sorted(source_images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"SKIP: missing label for {image_path.name}")
            continue

        pairs.append((image_path, label_path))

    return pairs


def rename_and_export(
    pairs: list[tuple[Path, Path]],
    dest_images: Path,
    dest_labels: Path,
    run_id: int,
    prefix: str,
    start_index: int,
    move_files: bool,
    dry_run: bool,
) -> int:
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    copied = 0

    for offset, (image_path, label_path) in enumerate(pairs, start=start_index):
        suffix = image_path.suffix.lower()
        new_stem = f"{prefix}_run{run_id}_{offset:04d}"
        new_image_path = dest_images / f"{new_stem}{suffix}"
        new_label_path = dest_labels / f"{new_stem}.txt"

        if new_image_path.exists() or new_label_path.exists():
            raise SystemExit(
                f"ERROR: destination already exists for {new_stem}"
            )

        print(f"{image_path.name} -> {new_image_path.name}")
        print(f"{label_path.name} -> {new_label_path.name}")

        if dry_run:
            copied += 1
            continue

        if move_files:
            shutil.move(str(image_path), str(new_image_path))
            shutil.move(str(label_path), str(new_label_path))
        else:
            shutil.copy2(str(image_path), str(new_image_path))
            shutil.copy2(str(label_path), str(new_label_path))

        copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely rename mosaic image/label pairs to avoid collisions."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("mosaics"),
        help="Source folder containing images/ and labels/ subfolders.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("TrainingV2"),
        help="Destination dataset folder containing images/ and labels/ subfolders.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Run number to embed in filenames, e.g. 2 -> mosaic_rare_run2_0001.jpg",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="mosaic_rare",
        help="Filename prefix before run id.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for the renamed files.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned renames without writing files.",
    )
    args = parser.parse_args()

    source_images = args.source / "images"
    source_labels = args.source / "labels"
    dest_images = args.dest / "images"
    dest_labels = args.dest / "labels"

    if not source_images.exists():
        raise SystemExit(f"ERROR: source images folder not found: {source_images}")
    if not source_labels.exists():
        raise SystemExit(f"ERROR: source labels folder not found: {source_labels}")
    if not args.dest.exists():
        raise SystemExit(f"ERROR: destination folder not found: {args.dest}")

    pairs = collect_pairs(source_images, source_labels)
    if not pairs:
        raise SystemExit("ERROR: no valid image/label pairs found.")

    print(f"Source      : {args.source}")
    print(f"Destination : {args.dest}")
    print(f"Run ID      : {args.run_id}")
    print(f"Pairs found : {len(pairs)}")
    print(f"Mode        : {'MOVE' if args.move else 'COPY'}")
    print(f"Dry run     : {args.dry_run}")

    exported = rename_and_export(
        pairs=pairs,
        dest_images=dest_images,
        dest_labels=dest_labels,
        run_id=args.run_id,
        prefix=args.prefix,
        start_index=args.start_index,
        move_files=args.move,
        dry_run=args.dry_run,
    )

    print(f"Done: {exported} pairs processed.")


if __name__ == "__main__":
    main()
