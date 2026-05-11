from __future__ import annotations
 
import argparse
import random
import json
import secrets
from pathlib import Path
from collections import defaultdict
 
import cv2
import numpy as np
 
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
 
# Spare Classes less frequence id 
RARE_CLASSES = [
    "a30", "d1", "d28", "d35", "d45", "e1", "e7",
    "f26", "f32", "g7", "h1", "m20", "n29", "p1",
    "p3", "r4", "v4", "w24", "z11"
]
 
# Class Map should match the Gardiner_names.txt file    
GARDINER_NAMES = [
    "Aa1","a1","a2","a24","a30","a40","a42","a50","b1","cartouche",
    "d1","d2","d4","d10","d21","d28","d35","d36","d37","d39",
    "d40","d45","d46","d54","d55","d56","d58","d60","e1","e7",
    "e10","e16","e23","e34","f1","f4","f12","f13","f26","f31",
    "f32","f35","f39","f51","g1","g5","g7","g14","g17","g25",
    "g36","g39","g40","g41","g43","h1","h6","i1","i9","i10",
    "i12","i15","l1","l2","m2","m3","m12","m16","m17","m18",
    "m20","m22","m23","m24","m42","n1","n5","n8","n14","n17",
    "n23","n25","n26","n27","n29","n33","n35","n36","n37","o1",
    "o3","o4","o6","o28","o34","o49","p1","p3","q1","q3",
    "r4","r7","r8","r11","s3","s19","s28","s29","s34","s38",
    "s40","t22","t28","t30","u1","u6","u15","u23","u33","u36",
    "unknown","v4","v6","v7","v13","v20","v28","v29","v30","v31",
    "w10","w14","w17","w19","w22","w24","w25","x1","x8","y1",
    "y2","y3","y5","z1","z2","z3","z4","z7","z11","Aa15"
]
 
NAME_TO_ID = {name: idx for idx, name in enumerate(GARDINER_NAMES)}
 
 
def collect_rare_images(img_dir: Path,
                         lbl_dir: Path) -> dict[str, list[tuple[Path, Path]]]:
    """
    Collects pairs (image, label) for the rare classes.
    The filename must start with the class name.
    Example: a30_001.jpg → class a30
    """
    rare_pairs: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
 
    for img_path in img_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in VALID_EXT:
            continue
 
        #  It detects the class from the filename
        stem = img_path.stem
        matched_class = None
        for cls in RARE_CLASSES:
            if stem.startswith(cls + '_') or stem == cls:
                matched_class = cls
                break
 
        if matched_class is None:
            continue
 
        lbl_path = lbl_dir / (stem + '.txt')
        if lbl_path.exists():
            rare_pairs[matched_class].append((img_path, lbl_path))
 
    return dict(rare_pairs)
 
 
def load_yolo_labels(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Loads YOLO annotations: [(cls_id, xc, yc, w, h), ...]"""
    labels = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append((
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4])
                ))
    return labels
 
 
def remap_labels_to_tile(labels: list,
                          tile_row: int,
                          tile_col: int,
                          grid: int,
                          tile_size: int,
                          canvas_size: int) -> list:
    """
    Recalculates the YOLO coordinates of the original labels
    to the new coordinate system of the complete mosaic.
 
    The original labels are in normalized coordinates [0,1] of the tile.
    The new labels must be in normalized coordinates [0,1] of the canvas.
    """
    remapped = []
    tile_fraction = tile_size / canvas_size  
 
    for (cls_id, xc, yc, w, h) in labels:
        # Offset of the tile in the canvas (normalized)
        x_offset = tile_col * tile_fraction
        y_offset = tile_row * tile_fraction
 
        # New coordinates in the canvas
        new_xc = x_offset + xc * tile_fraction
        new_yc = y_offset + yc * tile_fraction
        new_w  = w  * tile_fraction
        new_h  = h  * tile_fraction
 
        # Clamp
        new_xc = max(0.0, min(1.0, new_xc))
        new_yc = max(0.0, min(1.0, new_yc))
        new_w  = max(0.001, min(1.0, new_w))
        new_h  = max(0.001, min(1.0, new_h))
 
        remapped.append((cls_id, new_xc, new_yc, new_w, new_h))
 
    return remapped
 
 
def build_mosaic_with_labels(
    all_pairs:   list[tuple[Path, Path]],
    grid:        int,
    tile_size:   int,
    canvas_size: int,
    rng:         random.Random
) -> tuple[np.ndarray, list]:
    """
    Builds a mosaic and its recalculated YOLO annotations.
    Returns (canvas_image, list_of_global_labels)
    """
    canvas      = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    all_labels  = []
 
    chosen = [rng.choice(all_pairs) for _ in range(grid * grid)]
 
    for idx, (img_path, lbl_path) in enumerate(chosen):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
 
        # Resize tile
        tile = cv2.resize(img, (tile_size, tile_size),
                          interpolation=cv2.INTER_AREA)
 
        row = idx // grid
        col = idx % grid
        y1  = row * tile_size
        x1  = col * tile_size
        canvas[y1:y1+tile_size, x1:x1+tile_size] = tile
 
        # Remap labels
        labels = load_yolo_labels(lbl_path)
        remapped = remap_labels_to_tile(
            labels, row, col, grid, tile_size, canvas_size
        )
        all_labels.extend(remapped)
 
    return canvas, all_labels
 
 
def main():
    parser = argparse.ArgumentParser(
        description="SphinxEyes — Mosaic Generator for rare classes"
    )
    parser.add_argument(
        "--source", type=str,
        default="./SphinxProjectV2/images/train",
        help="Training images directory"
    )
    parser.add_argument(
        "--labels", type=str,
        default="./SphinxProjectV2/labels/train",
        help="Training labels directory"
    )
    parser.add_argument(
        "--output", type=str,
        default="./SphinxProjectV2/mosaic",
        help="Output directory for mosaics"
    )
    parser.add_argument(
        "--count", type=int, default=80,
        help="Number of images generate ~ (default: 80)"
    )
    parser.add_argument(
        "--grid", type=int, default=6,
        help="Grid NxN (default: 6 → 6x6 = 36 tiles)"
    )
    parser.add_argument(
        "--canvas-size", type=int, default=640,
        help="Canvas size in px (default: 640)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: random each run)"
    )
    args = parser.parse_args()
 
    img_dir    = Path(args.source)
    lbl_dir    = Path(args.labels)
    output_dir = Path(args.output)
 
    if not img_dir.exists():
        raise SystemExit(f"ERROR: {img_dir} does not exist")
    if not lbl_dir.exists():
        raise SystemExit(f"ERROR: {lbl_dir} does not exist")
 
    if args.canvas_size % args.grid != 0:
        raise SystemExit("ERROR: canvas-size must be divisible by grid")
 
    tile_size = args.canvas_size // args.grid
 
    # Output directories
    out_img = output_dir / "images"
    out_lbl = output_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
 
    print(f"\nCollecting rare images...")
    rare_pairs = collect_rare_images(img_dir, lbl_dir)
 
    print(f"\nClasses found:")
    total_available = 0
    for cls in RARE_CLASSES:
        n = len(rare_pairs.get(cls, []))
        bar = '█' * min(n, 20)
        print(f"  {cls:12s}: {n:4d} imgs  {bar}")
        total_available += n
 
    if total_available == 0:
        raise SystemExit(
            "ERROR: No images of rare classes were found.\n"
            "Check that the files start with the class name.\n"
            "Ej: a30_001.jpg, g7_003.png"
        )
 
    # Combine all rare class pairs into a single list
    all_pairs = []
    for cls in RARE_CLASSES:
        all_pairs.extend(rare_pairs.get(cls, []))
 
    print(f"\nTotal pairs available : {len(all_pairs)}")
    print(f"Grid                    : {args.grid}x{args.grid}")
    print(f"Tile size               : {tile_size}x{tile_size}px")
    print(f"Canvas                  : {args.canvas_size}x{args.canvas_size}px")
    print(f"Mosaics to generate      : {args.count}")

    seed = args.seed
    if seed is None:
        seed = secrets.randbelow(2**32)

    print(f"Seed                    : {seed}")

    rng = random.Random(seed)
    generated = 0
 
    for idx in range(1, args.count + 1):
        canvas, labels = build_mosaic_with_labels(
            all_pairs, args.grid, tile_size, args.canvas_size, rng
        )
 
        # Guardar imagen
        img_path = out_img / f"mosaic_rare_{idx:04d}.jpg"
        cv2.imwrite(str(img_path), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
 
        # Guardar labels YOLO
        lbl_path = out_lbl / f"mosaic_rare_{idx:04d}.txt"
        with open(lbl_path, 'w') as f:
            for (cls_id, xc, yc, w, h) in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
 
        generated += 1
        if idx % 10 == 0:
            print(f"  {idx}/{args.count} mosaics generated...")
 
    print(f"\n{'='*55}")
    print(f"  ✓ {generated} mosaics generated")
    print(f"  Images : {out_img}")
    print(f"  Labels   : {out_lbl}")
    print(f"\n  Next step:")
    print(f"  Move the content of mosaic/images and mosaic/labels")
    print(f"  to your training directory before training V2.")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()
