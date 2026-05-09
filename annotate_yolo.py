#!/usr/bin/env python3


# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import argparse
# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import re 
# pyrefly: ignore [missing-import]
import json
# pyrefly: ignore [missing-import]
from pathlib import Path
# pyrefly: ignore [missing-import]
from tqdm import tqdm

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# ROI DETECTION  

def detect_roi(img: np.ndarray,
               threshold: int = 18,
               margin: float = 0.04) -> tuple[float, float, float, float]:
    """
    Detects the bounding box of the actual content ignoring black padding.
 
    Strategy:
      1. Explicit mask of black padding (pixels < 20 in the 3 channels)
      2. Crop to the non-black area
      3. Inside the crop: Otsu adapted to the real background (stone/papyrus/white)
      4. Bbox of the largest contour of the sign
      5. Conservative fallback only if everything fails
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
           if len(img.shape) == 3 else img.copy()
 
   
    BLACK_THRESHOLD  = 20
    MIN_CONTENT_FRAC = 0.05   

    if len(img.shape) == 3:
        not_black_mask = np.any(img > BLACK_THRESHOLD, axis=2)
    else:
        not_black_mask = gray > BLACK_THRESHOLD

    row_frac = not_black_mask.mean(axis=1)   
    col_frac = not_black_mask.mean(axis=0)   

    content_rows = np.where(row_frac >= MIN_CONTENT_FRAC)[0]
    content_cols = np.where(col_frac >= MIN_CONTENT_FRAC)[0]

    # PASO 2: Bbox of the non-black area
    if len(content_rows) < 5 or len(content_cols) < 5:
        # Image almost completely black — fallback
        return 0.5, 0.5, 0.90, 0.90

    ry = int(content_rows[0])
    rh = int(content_rows[-1]) + 1 - ry
    rx = int(content_cols[0])
    rw = int(content_cols[-1]) + 1 - rx

    # If the non-black area already covers >95% → no real black padding
    # Treat the entire image as content
    no_padding = (rw * rh) / (w * h) > 0.95

    if no_padding:
        # No black padding — use the entire image as base ROI
        rx, ry, rw, rh = 0, 0, w, h
 
    # Work inside the non-black crop
    crop_gray = gray[ry:ry+rh, rx:rx+rw]
    ch, cw    = crop_gray.shape[:2]
 
    if ch < 5 or cw < 5:
        return 0.5, 0.5, 0.90, 0.90
 
    # Detect background type inside the crop
    border_crop = np.concatenate([
        crop_gray[0, :], crop_gray[-1, :],
        crop_gray[:, 0], crop_gray[:, -1]
    ])
    bg_median = float(np.median(border_crop))
    dark_bg   = bg_median < 80
 
    # Otsu inside the crop
    blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    if dark_bg:
        _, binary = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
 
    #  Contour of the sign inside the crop
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
 
    min_area = 0.01 * ch * cw  # minimum 1% of the crop
    x, y, bw_px, bh_px = rx, ry, rw, rh  # default = full non-black area
 
    if contours:
        big_cnts = [c for c in contours if cv2.contourArea(c) > min_area]
        if big_cnts:
            all_pts      = np.vstack(big_cnts)
            cx, cy, cbw, cbh = cv2.boundingRect(all_pts)
            # Convert crop coords to original image coords
            x     = rx + cx
            y     = ry + cy
            bw_px = cbw
            bh_px = cbh
 
    # Verify that the sign bbox makes sense
    sign_ratio = (bw_px * bh_px) / (w * h)
    if sign_ratio < 0.01:
        # Bbox too small — use full non-black area
        x, y, bw_px, bh_px = rx, ry, rw, rh
 
    # Apply margin
    margin_px_x = int(w * margin)
    margin_px_y = int(h * margin)
 
    x1 = max(0, x - margin_px_x)
    y1 = max(0, y - margin_px_y)
    x2 = min(w, x + bw_px + margin_px_x)
    y2 = min(h, y + bh_px + margin_px_y)
 
    # Convert to YOLO normalized format
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width    = (x2 - x1) / w
    height   = (y2 - y1) / h
 
    # Clamp for safety
    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    width    = float(np.clip(width,    0.01, 1.0))
    height   = float(np.clip(height,   0.01, 1.0))
 
    return x_center, y_center, width, height
 

# Natural Sort Key in right order 

def natural_sort_key( name: str ) :
    
    parts = re.split( r'(\d+)', name.lower() ) 
    
    return [ int( p ) if p.isdigit()  else p for p in parts  ]

  
def build_class_map(source: Path) -> dict[str, int]:
    """
    Build the map {class_name: class_id} in natural order.
    Aa1 fixes the index 0; the rest in natural sort (A1, A2, ..., Z11);
    Aa15 fixes the last index.
    """
    folders = [d.name for d in source.iterdir() if d.is_dir()]
    # Critical this is my personal order 
    # this same order must be replciated exaclty in Roboflow 
    # otherwise the model will not be able to detect the hieroglyphs correctly
    first = next((f for f in folders if f.lower() == "aa1"), None)
    rest  = sorted([f for f in folders if f.lower() != "aa1" and f.lower() != "aa15"], key=natural_sort_key)
    last  = next((f for f in folders if f.lower() == "aa15"), None)
    # I must ensure this order is same in Roboflow 
    ordered = ([first] if first else []) + rest + ([last] if last else [])
    # Having iisues with this same sorting order in Roboflow for v2 image dataset 
    return {name: idx for idx, name in enumerate(ordered)}

 
# Core 
 
def annotate_dataset(source: Path,
                     margin: float = 0.04,
                     threshold: int = 18,
                     dry_run: bool = False,
                     only: str = None,
                     save_classes: bool = False) -> None:
 
    class_map = build_class_map(source)
 
    print(f"\n{'='*60}")
    print(f"  SphinxEyes YOLO Annotator — {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}")
    print(f"  Clases detectadas : {len(class_map)}")
    print(f"  Margen aplicado   : {margin*100:.1f}%")
    print(f"  Umbral fondo      : {threshold}")
    if only:
        print(f"  Folder mode       : only '{only}'")
    print(f"{'='*60}\n")
 
    # Save classes.txt
    if save_classes and not dry_run:
        classes_path = source / 'classes.txt'
        with open(classes_path, 'w') as f:
            for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
                f.write(f"{name}\n")
        print(f"  ✓ classes.txt saved in {classes_path}\n")
 
    # Save class_map.json for reference
    if save_classes and not dry_run:
        map_path = source / 'class_map.json'
        with open(map_path, 'w') as f:
            json.dump(dict(sorted(class_map.items(), key=lambda x: x[1])), f, indent=2)
        print(f"  ✓ class_map.json saved in {map_path}\n")
 
    total_ok      = 0
    total_skip    = 0
    total_fallback = 0
    problem_files = []
 
   # iterate directories
    folders = [source / name for name in class_map if (source / name).is_dir()]
    if only:
        folders = [f for f in folders if f.name.lower() == only.lower()]
        if not folders:
            print(f"ERROR: directory '{only}' not found in {source}")
            return
 
    for cls_dir in tqdm(folders, desc="Clases"):
        cls_name = cls_dir.name
        cls_id   = class_map[cls_name]
 
        images = [f for f in cls_dir.iterdir()
                  if f.suffix.lower() in VALID_EXT]
 
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                total_skip += 1
                problem_files.append(str(img_path))
                continue
 
            x_c, y_c, bw, bh = detect_roi(img,
                                            threshold=threshold,
                                            margin=margin)
 
           
            used_fallback = (bw > 0.87 and bh > 0.87)
            if used_fallback:
                total_fallback += 1
 
            yolo_line = f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n"
 
            if not dry_run:
                txt_path = img_path.with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    f.write(yolo_line)
 
            total_ok += 1
 
    # Final User report in CLI 
    print(f"\n{'='*60}")
    print(f"  Images   : {total_ok}")
    print(f"  Fallback (revisar) : {total_fallback}")
    print(f"  Errores (skip)     : {total_skip}")
 
    if total_fallback > 0:
        pct = total_fallback / max(total_ok, 1) * 100
        print(f"\n  ⚠  {pct:.1f}% usó fallback bbox.")
        print(f"     test --threshold 25 or --threshold 35 for those classes.")
 
    if problem_files:
        print(f"\n  Troublesome files :")
        # TODO: make a list of files with problems
        for f in problem_files[:10]:
            
            print(f"    {f}")
 
    if dry_run:
        print(f"\n  [DRY RUN] any file was overwritten")
        print(f"  run without --dry-run to apply changes.")
    else:
        print(f"\n  ✓ Annotations generated in YOLO format.")
        print(f"  Each image has its .txt in the same directory.")
 
    print(f"{'='*60}\n")
 
 
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SphinxEyes — YOLO Annotate Script "
    )
    parser.add_argument(
        "--source", type=str, default="./SphinxEyes_Final",
        help="Root directory of the dataset (default: ./SphinxEyes_Final)"
    )
    parser.add_argument(
        "--margin", type=float, default=0.04,
        help="Margin around the ROI as a fraction 0.0-1.0 (default: 0.04)"
    )
    parser.add_argument(
        "--threshold", type=int, default=18,
        help="Pixel threshold for dark background 0-255 (default: 18)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate without writing any .txt file"
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="EG Folder processing only  --only g17"
    )
    parser.add_argument(
        "--save-classes", action="store_true",
        help="Save classes.txt and class_map.json in the source directory"
    )
    args = parser.parse_args()
 
    source = Path(args.source)

    if not source.exists():
        print(f"ERROR: '{source}' no existe.")
        exit(1)
 
    annotate_dataset(
        source      = source,
        margin      = args.margin,
        threshold   = args.threshold,
        dry_run     = args.dry_run,
        only        = args.only,
        save_classes= args.save_classes
    )
 