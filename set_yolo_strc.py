#!/usr/bin/env python3

import os
import re
import shutil
import random
import json
import argparse
from pathlib import Path

VALID_IMG = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
SEED      = 42


def natural_sort_key(name: str):
    # Natural sort — same as annotate_yolo.py.
    parts = re.split(r'(\d+)', name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def build_class_map(source: Path) -> dict[str, int]:
    """
    Natural sort — same as annotate_yolo.py.
    Aa1 sets index 0; the rest in natural sort (A1, A2, ..., Z11);
    Aa15 sets the last index.
    """
    folders = [d.name for d in source.iterdir() if d.is_dir()]
    # Critical : it must be same order in v2 images dataset to match my best.pt from v1
    first = next((f for f in folders if f.lower() == "aa1"), None)
    rest  = sorted([f for f in folders if f.lower() != "aa1" and f.lower() != "aa15"], key=natural_sort_key)
    last  = next((f for f in folders if f.lower() == "aa15"), None)

    ordered = ([first] if first else []) + rest + ([last] if last else [])

    return {name: idx for idx, name in enumerate(ordered)}


def organize(source: Path,
             output: Path,
             val_split: float = 0.20,
             dry_run: bool = False) -> None:

    random.seed(SEED)
    class_map = build_class_map(source)

    # Output directories
    dirs = {
        "img_train": output / "images"  / "train",
        "img_val":   output / "images"  / "val",
        "lbl_train": output / "labels"  / "train",
        "lbl_val":   output / "labels"  / "val",
    }

    if not dry_run:
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val   = 0
    skipped     = 0
    no_label    = 0

    print(f"\n{'='*60}")
    print(f"  SphinxEyes Dataset Organizer — {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}")
    print(f"  Source  : {source}")
    print(f"  Output  : {output}")
    print(f"  Split   : {int((1-val_split)*100)}/{int(val_split*100)} train/val")
    print(f"  Clases  : {len(class_map)}")
    print(f"{'='*60}\n")

    for cls_name, cls_id in class_map.items():
        cls_dir = source / cls_name
        if not cls_dir.is_dir():
            continue

        images = sorted([
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMG
        ])

        pairs = []
        for img_path in images:
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                pairs.append((img_path, txt_path))
            else:
                no_label += 1

        if not pairs:
            print(f"  ⚠  {cls_name}: sin pares imagen+label — skip")
            skipped += 1
            continue

        # Stratified split
        random.shuffle(pairs)
        n_val   = max(1, int(len(pairs) * val_split))
        n_train = len(pairs) - n_val

        val_pairs   = pairs[:n_val]
        train_pairs = pairs[n_val:]

        # Copy files
        for (img_p, lbl_p), split in (
            [(p, "train") for p in train_pairs] +
            [(p, "val")   for p in val_pairs]
        ):
            # Unique name: class_original_name to avoid collisions
            stem     = f"{cls_name}_{img_p.stem}"
            img_dst  = dirs[f"img_{split}"] / (stem + img_p.suffix)
            lbl_dst  = dirs[f"lbl_{split}"] / (stem + ".txt")

            if not dry_run:
                shutil.copy2(img_p, img_dst)
                shutil.copy2(lbl_p, lbl_dst)

        total_train += n_train
        total_val   += n_val

        print(f"  {cls_name:15s} | {len(pairs):4d} imgs | "
              f"train={n_train:4d}  val={n_val:3d}")

    # Generate dataset.yaml
    yaml_path = output / "dataset.yaml"
    yaml_content = f"""# SphinxEyes v1 — YOLOv11 Dataset
# Generated automatically by organize_yolo.py
# Classes: {len(class_map)} | Train: {total_train} | Val: {total_val}

path: {output.resolve()}
train: images/train
val:   images/val

nc: {len(class_map)}
names: {json.dumps(list(class_map.keys()), ensure_ascii=False)}
"""

    if not dry_run:
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        # Save class_map.json for reference
        with open(output / "class_map.json", 'w') as f:
            json.dump(class_map, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    if dry_run:
        print(f"  [DRY RUN] No files copied.")
        print(f"  Estimated train : {total_train}")
        print(f"  Estimated val   : {total_val}")
    else:
        print(f"  ✓ Dataset organized successfully")
        print(f"  Train          : {total_train} images")
        print(f"  Val            : {total_val} images")
        print(f"  dataset.yaml   : {yaml_path}")
        print(f"  class_map.json : {output / 'class_map.json'}")

    if no_label > 0:
        print(f"  ⚠  {no_label} images without .txt — not included")
    if skipped > 0:
        print(f"  ⚠  {skipped} classes without valid pairs — ignored")

    print(f"{'='*60}\n")

    if not dry_run:
        print("  It is OK to train in Colab:")
        print(f"  from ultralytics import YOLO")
        print(f"  model = YOLO('yolo11l.pt')")
        print(f"  model.train(data='{yaml_path}', epochs=80, imgsz=224, batch=64)")
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SphinxEyes — Organiza dataset al formato YOLO"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Current root directory (SphinxEyes_Final)"
    )
    parser.add_argument(
        "--output", type=str, default="./SphinxEyes_v1",
        help="Output directory (default: ./SphinxEyes_v1)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.20,
        help="Validation fraction 0.0-1.0 (default: 0.20)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate without copying any files"
    )
    args   = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        
        print(f"ERROR: '{source}' does not exist.")
        exit(1)

    organize(source, output,
             val_split=args.val_split,
             dry_run=args.dry_run)