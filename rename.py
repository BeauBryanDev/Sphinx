import os
import sys
import argparse
from pathlib import Path
 

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
 
 
def rename_images_in_dir(root: Path, start: int = 0, dry_run: bool = False) -> int:
    """
    rename image inside parent folder 
        <nombre_root>_{i}.<ext>
 
    Return file renames.
    """
    dir_name = root.name  # e.g. "glyphs"
 
    images = sorted(

        f for f in root.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
 
    if not images:
        print(f"[INFO] Images were not found: {root.name}/")
        return 0
 
    renamed = 0
 
    # First step: rename to temporary names to avoid collisions
    temp_paths = []
    for img in images:
        tmp = img.with_name(f"__tmp_{img.name}")
        if not dry_run:
            img.rename(tmp)
        temp_paths.append((tmp, img.suffix.lower()))
 
    # Rename with final name 
    for i, (tmp_path, ext) in enumerate(temp_paths, start=start):
        final_name = f"{dir_name}_{i}{ext}"
        final_path = root / final_name
 
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
        help="Root directory containing the subdirectories with images (e.g. glyphs/)"
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
        print(f"[ERROR] does nto exits : {root}", file=sys.stderr)
        sys.exit(1)
 
    if args.dry_run:
        print(f"[DRY-RUN] simualating : {root.resolve()}\n")
    else:
            print(f"[INFO] Procesando el directorio: {root.resolve()}\n")
    
    total_renamed = rename_images_in_dir(root, start=args.start, dry_run=args.dry_run)
 
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''} Done: {total_renamed} images renamed.")
 
 
if __name__ == "__main__":

    main()
