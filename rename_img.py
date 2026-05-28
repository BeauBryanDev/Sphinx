import sys
import argparse
from pathlib import Path

# this script is used only during the ETL pipeline
# We are dealing with many images, they came from different sources
# like Kaggle, UniData and Museums.
# They must be renamed according to their Gardiner signs code (class name)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def rename_split_dir(root: Path, prefix: str, start: int = 0, dry_run: bool = False) -> int:
    """
    Handle the case where images/ and labels/ are separate subdirectories
    inside the root (e.g. mosaics/images/ and mosaics/labels/).
    Images and labels are paired by sorted order.
    """
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.is_dir():
        print(f"  [SKIP] No 'images/' subdir found in {root}", file=sys.stderr)
        return 0

    images = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    labels = []
    if labels_dir.is_dir():
        labels = sorted(
            f for f in labels_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        )

    if not images:
        print(f"  [SKIP] No images found in {images_dir}")
        return 0

    if labels and len(labels) != len(images):
        print(
            f"  [WARN] Image count ({len(images)}) != label count ({len(labels)}) in {root}. "
            "Labels will be paired up to the shorter list."
        )

    renamed = 0

    # ── Step 1: move to temp names to avoid collisions ──────────────────────
    temp_pairs = []
    for i, img in enumerate(images):
        tmp_img = images_dir / f"__tmp_{img.name}"
        lbl = labels[i] if i < len(labels) else None
        tmp_lbl = labels_dir / f"__tmp_{lbl.name}" if lbl else None

        if not dry_run:
            img.rename(tmp_img)
            if lbl and tmp_lbl:
                lbl.rename(tmp_lbl)

        temp_pairs.append((tmp_img, img.suffix.lower(), tmp_lbl))

    # ── Step 2: rename to final names ────────────────────────────────────────
    for i, (tmp_img, ext, tmp_lbl) in enumerate(temp_pairs, start=start):
        final_img_name = f"{prefix}_{i}{ext}"
        final_lbl_name = f"{prefix}_{i}.txt"

        if dry_run:
            orig_img = tmp_img.name.replace("__tmp_", "", 1)
            orig_lbl = tmp_lbl.name.replace("__tmp_", "", 1) if tmp_lbl else "(no label)"
            print(f"  [DRY-RUN] images/{orig_img}  →  images/{final_img_name}"
                  f"   |   labels/{orig_lbl}  →  labels/{final_lbl_name}")
        else:
            tmp_img.rename(images_dir / final_img_name)
            if tmp_lbl:
                tmp_lbl.rename(labels_dir / final_lbl_name)
            renamed += 1

    return renamed


def rename_flat_dir(subdir: Path, start: int = 0, dry_run: bool = False) -> int:
    """
    Handle the classic flat case: image.jpg and image.txt live side by side
    in the same subdirectory.
    """
    dir_name = subdir.name

    images = sorted(
        f for f in subdir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        return 0

    renamed = 0

    # Step 1: temp names
    temp_paths = []
    for img in images:
        tmp_img = img.with_name(f"__tmp_{img.name}")
        label = img.with_suffix(".txt")
        tmp_label = label.with_name(f"__tmp_{label.name}") if label.exists() else None

        if not dry_run:
            img.rename(tmp_img)
            if tmp_label:
                label.rename(tmp_label)

        temp_paths.append((tmp_img, img.suffix.lower(), tmp_label))

    # Step 2: final names
    for i, (tmp_img, ext, tmp_label) in enumerate(temp_paths, start=start):
        final_name = f"{dir_name}_{i}{ext}"
        final_path = subdir / final_name

        if dry_run:
            original_name = tmp_img.name.replace("__tmp_", "", 1)
            original_label = tmp_label.name.replace("__tmp_", "", 1) if tmp_label else "(no label)"
            print(f"  [DRY-RUN] {original_name}  →  {final_name}   |  label: {original_label}  →  {dir_name}_{i}.txt")
        else:
            tmp_img.rename(final_path)
            if tmp_label:
                tmp_label.rename(subdir / f"{dir_name}_{i}.txt")
            renamed += 1

    return renamed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename all images (and their paired label .txt files) in a directory.\n"
            "Supports two layouts:\n"
            "  • Split:  root/images/*.jpg  +  root/labels/*.txt  (paired by sort order)\n"
            "  • Flat:   root/<class>/<img>.jpg  +  root/<class>/<img>.txt  (same folder)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root directory to process (e.g. ./mosaics/)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for split-layout output filenames (default: root folder name)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index of the counter (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the changes without modifying anything on disk",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    mode = "[DRY-RUN] " if args.dry_run else ""
    print(f"{mode}Processing: {root.resolve()}\n")

    # ── Detect layout ────────────────────────────────────────────────────────
    has_images_subdir = (root / "images").is_dir()

    if has_images_subdir:
        # Split layout: mosaics/images/ + mosaics/labels/
        prefix = args.prefix or root.name
        print(f"  Layout detected: SPLIT  (images/ + labels/)  →  prefix='{prefix}'\n")
        count = rename_split_dir(root, prefix=prefix, start=args.start, dry_run=args.dry_run)
        print(f"\n{mode}Done: {count} image(s) renamed.")
    else:
        # Flat layout: each subdir has images + labels side by side
        print("  Layout detected: FLAT  (per-class subdirs with paired .txt files)\n")
        total_renamed = 0
        total_dirs = 0
        subdirs = sorted(d for d in root.iterdir() if d.is_dir())
        for subdir in subdirs:
            count = rename_flat_dir(subdir, start=args.start, dry_run=args.dry_run)
            if count > 0 or args.dry_run:
                total_dirs += 1
                if not args.dry_run:
                    print(f"  ✓  {subdir.name}/  →  {count} image(s) renamed")
            total_renamed += count
        print(f"\n{mode}Done: {total_renamed} image(s) in {total_dirs} directories.")


if __name__ == "__main__":
    main()
