#!/usr/bin/env python3

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm

VALID_EXT  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
TARGET     = 224
MARGIN     = 6      # Margin px around sign 


def detect_background_color(img: np.ndarray) -> tuple:
    """
    It detect the background color by using the px color mean 
    """
    border = np.concatenate([
        img[0,  :].reshape(-1, 3),
        img[-1, :].reshape(-1, 3),
        img[:,  0].reshape(-1, 3),
        img[:, -1].reshape(-1, 3),
    ])
    return tuple(int(x) for x in np.median(border, axis=0))


def resize_to_square(img: np.ndarray,
                     target: int = TARGET,
                     margin: int = MARGIN) -> np.ndarray:
    """
    Re scale to largest possible size (target - 2*margin)
    keeping aspect ratio, centering inside canvas of background color.
    """
    h, w = img.shape[:2]

    # If it is already 224x224 — no need to resize
    if h == target and w == target:

        return img

    max_side = target - 2 * margin
    scale    = max_side / max(h, w)
    # new width and height
    new_w    = int(w * scale)
    new_h    = int(h * scale)

    # Inner area interpolation to reduce aliasing
    # INTER_LANCZOS4 to increase size
    interp   = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized  = cv2.resize(img, (new_w, new_h), interpolation=interp)

    bg_color = detect_background_color(img)
    canvas   = np.full((target, target, 3), bg_color, dtype=np.uint8)

    x_off    = (target - new_w) // 2
    y_off    = (target - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def process_directory(cls_dir: Path,
                       inplace: bool = False,
                       dry_run: bool = False) -> dict:
    """Process all image in one directory ."""
    images = [f for f in cls_dir.iterdir()
              if f.is_file() and f.suffix.lower() in VALID_EXT]

    stats = {"total": len(images), "resized": 0,
             "skipped": 0, "errors": 0}

    if not images:
        return stats

    for img_path in tqdm(images, desc=f"  {cls_dir.name}", leave=False):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                stats["errors"] += 1
                continue

            h, w = img.shape[:2]

            # It s already 224x224 — skip
            if h == TARGET and w == TARGET:
                stats["skipped"] += 1
                continue

            if dry_run:
                stats["resized"] += 1
                continue

            result = resize_to_square(img)

            if inplace:
                # Override origina file 
                # keep img as JPG to sabe space
                out_path = img_path.with_suffix('.jpg')
                cv2.imwrite(str(out_path), result,
                            [cv2.IMWRITE_JPEG_QUALITY, 93])
                # Delete  PNG original
                if img_path.suffix.lower() == '.png':
                    img_path.unlink()
            else:
                # Save with _224 sifix for better understanding 
                out_path = img_path.with_stem(
                    img_path.stem + '_224'
                ).with_suffix('.jpg')
                cv2.imwrite(str(out_path), result,
                            [cv2.IMWRITE_JPEG_QUALITY, 93])

            stats["resized"] += 1

        except Exception as e:
            tqdm.write(f"    ERROR {img_path.name}: {e}")
            stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="SphinxEyes — Batch resize a 224x224"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Root directory of the dataset or a class folder"
    )
    parser.add_argument(
        "--pause", type=float, default=0.5,
        help="Pause in seconds between folders (default: 0.5, laptop mode)"
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="Sobrescribe originales (default: guarda como _224.jpg)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just count images without writing anything"
    )
    args   = parser.parse_args()
    source = Path(args.source)

    if not source.exists():
        print(f"ERROR: '{source}' no existe.")
        return

    subdirs = [d for d in source.iterdir() if d.is_dir()]
    has_subdirs = any(
        any(f.suffix.lower() in VALID_EXT for f in d.iterdir() if f.is_file())
        for d in subdirs
    ) if subdirs else False

    if not has_subdirs:
        # source  is the parent directory / + class from Gardiner List 
        folders = [source]
    else:
      
        folders = sorted([d for d in source.iterdir() if d.is_dir()])

    print(f"\n{'='*55}")
    print(f"  SphinxEyes Batch Resize — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Target     : {TARGET}×{TARGET} px")
    print(f"  Modo       : {'inplace (sobrescribe)' if args.inplace else 'new file _224.jpg'}")
    print(f"  Pausa      : {args.pause}s among folders")
    print(f"  Folders    : {len(folders)}")
    print(f"{'='*55}\n")

    total_resized = 0
    total_skipped = 0
    total_errors  = 0

    for i, folder in enumerate(folders):
        stats = process_directory(
            folder,
            inplace  = args.inplace,
            dry_run  = args.dry_run
        )
        total_resized += stats["resized"]
        total_skipped += stats["skipped"]
        total_errors  += stats["errors"]

        if stats["resized"] > 0 or stats["errors"] > 0:
            tqdm.write(
                f"  {folder.name:12s} → "
                f"{stats['resized']} resized, "
                f"{stats['skipped']} skip, "
                f"{stats['errors']} err"
            )

        # Pause on betweeen folder to avoid overwhlem my CPU 
        if args.pause > 0 and i < len(folders) - 1:
            time.sleep(args.pause)

    print(f"\n{'='*55}")
    if args.dry_run:
        print(f"  [DRY RUN] would process {total_resized} images.")
    else:
        print(f"  COMPLETED ")
        print(f"  Resized  : {total_resized}")
        print(f"  Skipped  : {total_skipped} (already 224×224)")
        print(f"  Errors  : {total_errors}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()