#!/usr/bin/env python3
"""
SphinxEyes — Auto-Annotation YOLO Format
==========================================
Genera archivos .txt en formato YOLO para cada imagen en SphinxEyes_Final/.
Detecta automáticamente el contenido real de la imagen ignorando padding
negro o fondo de piedra/papiro.

Formato YOLO generado:
    <class_id> <x_center> <y_center> <width> <height>
    (todos los valores normalizados 0.0 - 1.0)

Uso:
    # Ver clases detectadas y preview sin escribir nada
    python annotate_yolo.py --source ./SphinxEyes_Final --dry-run

    # Generar anotaciones con margen del 5%
    python annotate_yolo.py --source ./SphinxEyes_Final --margin 0.05

    # Solo una carpeta específica
    python annotate_yolo.py --source ./SphinxEyes_Final --only g17

    # Con umbral manual para clases problemáticas (0-255)
    python annotate_yolo.py --source ./SphinxEyes_Final --only d21 --threshold 25

    # Generar classes.txt con el mapa class_name -> class_id
    python annotate_yolo.py --source ./SphinxEyes_Final --save-classes

Dependencias:
    pip install opencv-python numpy tqdm
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# ─── Detección de ROI ─────────────────────────────────────────────────────────

def detect_roi(img: np.ndarray,
               threshold: int = 18,
               margin: float = 0.04
               ) -> tuple[float, float, float, float]:
    """
    Detecta el bounding box del contenido real ignorando padding negro.

    Estrategia:
      1. Máscara explícita de padding negro (píxeles < 20 en los 3 canales)
      2. Crop al área no-negra
      3. Dentro del crop: Otsu adaptado al fondo real (piedra/papiro/blanco)
      4. Bbox del contorno más grande del signo
      5. Fallback conservador solo si todo falla
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
           if len(img.shape) == 3 else img.copy()

    # ── PASO 1: Eliminar padding negro explícitamente ──
    # Estrategia por perfil: una fila/columna es "contenido" solo si
    # al menos MIN_CONTENT_FRAC de sus píxeles son no-negros.
    # Robusto frente a ruido JPEG en bandas de padding negro
    # (1-2% píxeles ruidosos) — nunca alcanzan el 5% de contenido real.
    BLACK_THRESHOLD  = 20
    MIN_CONTENT_FRAC = 0.05   # 5% mínimo para considerar fila/col como contenido

    if len(img.shape) == 3:
        not_black_mask = np.any(img > BLACK_THRESHOLD, axis=2)
    else:
        not_black_mask = gray > BLACK_THRESHOLD

    row_frac = not_black_mask.mean(axis=1)   # fracción no-negra por fila
    col_frac = not_black_mask.mean(axis=0)   # fracción no-negra por columna

    content_rows = np.where(row_frac >= MIN_CONTENT_FRAC)[0]
    content_cols = np.where(col_frac >= MIN_CONTENT_FRAC)[0]

    # ── PASO 2: Bbox del área no-negra ──
    if len(content_rows) < 5 or len(content_cols) < 5:
        # Imagen casi completamente negra — fallback
        return 0.5, 0.5, 0.90, 0.90

    ry = int(content_rows[0])
    rh = int(content_rows[-1]) + 1 - ry
    rx = int(content_cols[0])
    rw = int(content_cols[-1]) + 1 - rx

    # Si el área no-negra ya cubre >95% → no hay padding negro real
    # Tratar toda la imagen como contenido
    no_padding = (rw * rh) / (w * h) > 0.95

    if no_padding:
        # Sin padding negro — usar toda la imagen como ROI base
        rx, ry, rw, rh = 0, 0, w, h

    # ── PASO 3: Trabajar dentro del crop no-negro ──
    crop_gray = gray[ry:ry+rh, rx:rx+rw]
    ch, cw    = crop_gray.shape[:2]

    if ch < 5 or cw < 5:
        return 0.5, 0.5, 0.90, 0.90

    # Detectar tipo de fondo dentro del crop
    border_crop = np.concatenate([
        crop_gray[0, :], crop_gray[-1, :],
        crop_gray[:, 0], crop_gray[:, -1]
    ])
    bg_median = float(np.median(border_crop))
    dark_bg   = bg_median < 80

    # Otsu dentro del crop
    blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    if dark_bg:
        _, binary = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── PASO 4: Contorno del signo dentro del crop ──
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    min_area = 0.01 * ch * cw  # mínimo 1% del crop
    x, y, bw_px, bh_px = rx, ry, rw, rh  # default = área no-negra completa

    if contours:
        
        big_cnts = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if big_cnts:
            
            all_pts      = np.vstack(big_cnts)
            cx, cy, cbw, cbh = cv2.boundingRect(all_pts)
            # Convertir coords del crop a coords de imagen original
            x     = rx + cx
            y     = ry + cy
            bw_px = cbw
            bh_px = cbh

    # Verificar que el bbox del signo tiene sentido
    sign_ratio = (bw_px * bh_px) / (w * h)
    if sign_ratio < 0.01:
        # Bbox demasiado pequeño — usar área no-negra completa
        x, y, bw_px, bh_px = rx, ry, rw, rh

    # ── Aplicar margen ──
    margin_px_x = int(w * margin)
    margin_px_y = int(h * margin)

    x1 = max(0, x - margin_px_x)
    y1 = max(0, y - margin_px_y)
    x2 = min(w, x + bw_px + margin_px_x)
    y2 = min(h, y + bh_px + margin_px_y)

    # Convertir a formato YOLO normalizado
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width    = (x2 - x1) / w
    height   = (y2 - y1) / h

    # Clamp para seguridad
    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    width    = float(np.clip(width,    0.01, 1.0))
    height   = float(np.clip(height,   0.01, 1.0))

    return x_center, y_center, width, height


# ─── Construcción del mapa de clases ─────────────────────────────────────────

def build_class_map(source: Path) -> dict[str, int]:
    """
    Construye el mapa {class_name: class_id} ordenado alfabéticamente.
    El orden es determinista — siempre el mismo class_id para cada carpeta.
    """
    folders = sorted([
        d.name for d in source.iterdir()
        if d.is_dir()
    ])
    return {name: idx for idx, name in enumerate(folders)}


# ─── Core ─────────────────────────────────────────────────────────────────────

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
        print(f"  Modo carpeta      : solo '{only}'")
    print(f"{'='*60}\n")

    # Guardar classes.txt
    if save_classes and not dry_run:
        classes_path = source / 'classes.txt'
        with open(classes_path, 'w') as f:
            for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
                f.write(f"{name}\n")
        print(f"  ✓ classes.txt guardado en {classes_path}\n")

    # Guardar class_map.json para referencia
    if save_classes and not dry_run:
        map_path = source / 'class_map.json'
        with open(map_path, 'w') as f:
            json.dump(class_map, f, indent=2, sort_keys=True)
        print(f" class_map.json guardado en {map_path}\n")

    total_ok      = 0
    total_skip    = 0
    total_fallback = 0
    problem_files = []

    # Iterar carpetas
    folders = [source / name for name in class_map if (source / name).is_dir()]
    if only:
        folders = [f for f in folders if f.name.lower() == only.lower()]
        if not folders:
            print(f"ERROR: carpeta '{only}' no encontrada en {source}")
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

            # Detectar si usó fallback (bbox muy grande = posible problema)
            used_fallback = (bw > 0.87 and bh > 0.87)
            if used_fallback:
                total_fallback += 1

            yolo_line = f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n"

            if not dry_run:
                txt_path = img_path.with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    f.write(yolo_line)

            total_ok += 1

    # Reporte final
    print(f"\n{'='*60}")
    print(f"  Imágenes anotadas  : {total_ok}")
    print(f"  Fallback (revisar) : {total_fallback}")
    print(f"  Errores (skip)     : {total_skip}")

    if total_fallback > 0:
        pct = total_fallback / max(total_ok, 1) * 100
        print(f"\n  ⚠  {pct:.1f}% usó fallback bbox.")
        print(f"     Prueba --threshold 25 o --threshold 35 para esas clases.")

    if problem_files:
        print(f"\n  Archivos con error:")
        for f in problem_files[:10]:
            print(f"    {f}")

    if dry_run:
        print(f"\n  [DRY RUN] No se escribió ningún archivo.")
        print(f"  Ejecuta sin --dry-run para aplicar.")
    else:
        print(f"\n  ✓ Anotaciones generadas en formato YOLO.")
        print(f"  Cada imagen tiene su .txt en el mismo directorio.")

    print(f"{'='*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SphinxEyes — Generador automático de anotaciones YOLO"
    )
    parser.add_argument(
        "--source", type=str, default="./SphinxEyes_Final",
        help="Directorio raíz del dataset (default: ./SphinxEyes_Final)"
    )
    parser.add_argument(
        "--margin", type=float, default=0.04,
        help="Margen alrededor del ROI como fracción 0.0-1.0 (default: 0.04)"
    )
    parser.add_argument(
        "--threshold", type=int, default=18,
        help="Umbral de píxel para fondo oscuro 0-255 (default: 18)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula sin escribir ningún archivo .txt"
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Procesa solo una carpeta específica, ej: --only g17"
    )
    parser.add_argument(
        "--save-classes", action="store_true",
        help="Guarda classes.txt y class_map.json en el directorio fuente"
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