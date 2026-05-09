#!/usr/bin/env python3
"""Delete images smaller than MIN_W x MIN_H. Pass --dry-run to preview."""
import argparse, os, sys
from pathlib import Path
# pyrefly: ignore [missing-import]
from PIL import Image

MIN_W, MIN_H = 100, 100
EXTS = {".png", ".jpg", ".jpeg"}

def main():
        
    """ Remove small images from the dataset """

    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="glyphs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"not a dir: {root}")

    to_delete = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in EXTS or not p.is_file():
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            print(f"skip (unreadable): {p} ({e})")
            continue
        if w < MIN_W or h < MIN_H:
            to_delete.append((p, w, h))

    for p, w, h in to_delete:
        print(f"{w}x{h}  {p}")
        if not args.dry_run:
            os.remove(p)

    action = "would delete" if args.dry_run else "deleted"
    print(f"\n{action} {len(to_delete)} files")


if __name__ == "__main__":
    main()
