import os
import sys
import argparse
from pathlib import Path
 
 # this script is used only during the ETL pipeline 
 # We are dealing with many images, they came form difference soruces 
 # like Kaggle, UniData and Museums.
 # They must be rename accorindg to their Gardienr signs code (class name)
 

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
 
 
def rename_images_in_dir(subdir: Path, start: int = 0, dry_run: bool = False) -> int:
    """
    rename all images before training.
    """
    dir_name = subdir.name  # e.g. "d40"
 
    # Gather all images and sort them 
    images = sorted(
        f for f in subdir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
 
    if not images:
        return 0
 
    renamed = 0
 
    # First step: rename to temporary names to avoid collisions
    # (e.g. if d40_0.jpg already exists and I want to put d40_0.jpg in another file)
    temp_paths = []
    for img in images:
        tmp = img.with_name(f"__tmp_{img.name}")
        if not dry_run:
            img.rename(tmp)
        temp_paths.append((tmp, img.suffix.lower()))
 
    # Second step: rename to final name
    for i, (tmp_path, ext) in enumerate(temp_paths, start=start):
        final_name = f"{dir_name}_{i}{ext}"
        final_path = subdir / final_name
 
        if dry_run:
            original_name = tmp_path.name.replace("__tmp_", "", 1)
            print(f"  [DRY-RUN] {original_name}  →  {final_name}")
        else:
            tmp_path.rename(final_path)
            renamed += 1
 
    return renamed
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Rename all images in the subdirectories according to the name of the directory."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root directory containing the subdirectories with images (e.g. Sphinx/)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index of the counter (default: 0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the changes without modifying anything on disk"
    )
    args = parser.parse_args()
 
    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] Directory not found: {root}", file=sys.stderr)
        sys.exit(1)
 
    if args.dry_run:
        print(f"[DRY-RUN] Simulating: {root.resolve()}\n")
    else:
        print(f"[INFO] Processing: {root.resolve()}\n")
 
    total_renamed = 0
    total_dirs = 0
 

    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
 
    for subdir in subdirs:
        count = rename_images_in_dir(subdir, start=args.start, dry_run=args.dry_run)
        if count > 0 or args.dry_run:
            total_dirs += 1
            if not args.dry_run:
                print(f"  ✓  {subdir.name}/  →  {count} image(s) renamed")
        total_renamed += count
 
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Done: {total_renamed} image(s) in {total_dirs} directories.")
 
 
if __name__ == "__main__":
    main()
 
