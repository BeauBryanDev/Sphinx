#!/usr/bin/env python3
"""
SphinxEyes — YOLO Bbox Visualizer
===================================
Visualiza las bounding boxes anotadas en formato YOLO sobre las imágenes.
Útil para inspección visual antes del training V1.

Uso:
    # Ver todas las imágenes de una clase (tecla por tecla)
    python bbox_viewer.py --source ./SphinxEyes_Final/d21

    # Ver dataset completo carpeta por carpeta
    python bbox_viewer.py --source ./SphinxEyes_Final

    # Guardar todas las visualizaciones como imágenes (sin ventana)
    python bbox_viewer.py --source ./SphinxEyes_Final/g17 --save

    # Solo mostrar imágenes con bbox sospechoso (muy grande o muy pequeño)
    python bbox_viewer.py --source ./SphinxEyes_Final --suspicious-only

Controles de teclado:
    SPACE / → : siguiente imagen
    ←         : imagen anterior
    D         : marcar imagen como "delete" (escribe en delete_list.txt)
    Q / ESC   : salir

Dependencias:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

VALID_EXT  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
COLOR_BOX  = (0, 200, 80)     # verde
COLOR_SUSP = (0, 80, 220)     # rojo — bbox sospechoso
COLOR_TEXT = (255, 255, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def load_yolo_annotation(txt_path: Path, img_w: int, img_h: int):
    """
    Lee el .txt YOLO y devuelve lista de bboxes en píxeles.
    Retorna: [(class_id, x1, y1, x2, y2), ...]
    """
    if not txt_path.exists():
        return []

    bboxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2) * img_w)
            y1 = int((yc - bh / 2) * img_h)
            x2 = int((xc + bw / 2) * img_w)
            y2 = int((yc + bh / 2) * img_h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            bboxes.append((cls_id, x1, y1, x2, y2, bw, bh))

    return bboxes


def is_suspicious(bw: float, bh: float) -> bool:
    """
    Bbox sospechoso si:
    - Es demasiado grande (>92% de la imagen) → probablemente fallback
    - Es demasiado pequeño (<20% de la imagen) → posible error
    """
    area = bw * bh
    return area > 0.85 or area < 0.04


def draw_bbox(img: np.ndarray, bboxes: list, cls_name: str) -> np.ndarray:
    """Dibuja bboxes sobre la imagen con info de clase y coordenadas."""
    vis = img.copy()
    h, w = vis.shape[:2]

    # Escalar a tamaño visible si es 224x224
    scale = 3 if w <= 224 else 1
    if scale > 1:
        vis = cv2.resize(vis, (w * scale, h * scale),
                         interpolation=cv2.INTER_NEAREST)
    vis_h, vis_w = vis.shape[:2]

    for (cls_id, x1, y1, x2, y2, bw, bh) in bboxes:
        susp  = is_suspicious(bw, bh)
        color = COLOR_SUSP if susp else COLOR_BOX

        # Escalar coords
        sx1, sy1 = x1 * scale, y1 * scale
        sx2, sy2 = x2 * scale, y2 * scale

        # Bbox
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), color, 2)

        # Label
        label = f"{cls_name} [{bw:.2f}x{bh:.2f}]"
        if susp:
            label += " ⚠"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(vis, (sx1, sy1 - th - 6),
                      (sx1 + tw + 4, sy1), color, -1)
        cv2.putText(vis, label, (sx1 + 2, sy1 - 4),
                    FONT, 0.5, COLOR_TEXT, 1)

    # Info en esquina
    info = f"{cls_name} | {vis_w//scale}x{vis_h//scale}px"
    cv2.putText(vis, info, (8, vis_h - 10),
                FONT, 0.45, (200, 200, 200), 1)

    if not bboxes:
        cv2.putText(vis, "NO ANNOTATION", (8, 30),
                    FONT, 0.7, (0, 80, 220), 2)

    return vis


def visualize_directory(cls_dir: Path,
                         save: bool = False,
                         suspicious_only: bool = False) -> list:
    """
    Muestra imágenes de un directorio con sus bboxes.
    Retorna lista de imágenes marcadas para borrar.
    """
    images = sorted([f for f in cls_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in VALID_EXT])

    if not images:
        return []

    cls_name   = cls_dir.name
    delete_list = []

    if save:
        out_dir = cls_dir.parent / f"_bbox_preview" / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    while 0 <= idx < len(images):
        img_path = images[idx]
        txt_path = img_path.with_suffix('.txt')

        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        h, w    = img.shape[:2]
        bboxes  = load_yolo_annotation(txt_path, w, h)

        # Filtro modo suspicious_only
        if suspicious_only:
            has_susp = any(is_suspicious(bw, bh)
                           for (_, _, _, _, _, bw, bh) in bboxes)
            if not has_susp and bboxes:
                idx += 1
                continue

        vis = draw_bbox(img, bboxes, cls_name)

        if save:
            out_path = out_dir / f"{img_path.stem}_bbox.jpg"
            cv2.imwrite(str(out_path), vis)
            idx += 1
            continue

        # Título de ventana
        title = (f"[{idx+1}/{len(images)}] {cls_name} — "
                 f"{img_path.name} | SPACE:next  ←:prev  D:delete  Q:quit")
        cv2.imshow(title, vis)
        cv2.setWindowTitle(title, title)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):       # Q o ESC
            cv2.destroyAllWindows()
            return delete_list
        elif key in (ord(' '), 83, 3):  # SPACE o →
            idx += 1
        elif key in (81, 2):            # ←
            idx = max(0, idx - 1)
        elif key == ord('d'):           # marcar para borrar
            delete_list.append(str(img_path))
            print(f"  🗑  Marcado: {img_path.name}")
            idx += 1

        cv2.destroyAllWindows()

    return delete_list


def main():
    parser = argparse.ArgumentParser(
        description="SphinxEyes — Visualizador de bounding boxes YOLO"
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Directorio de clase o raíz del dataset")
    parser.add_argument("--save", action="store_true",
                        help="Guarda visualizaciones sin abrir ventana")
    parser.add_argument("--suspicious-only", action="store_true",
                        help="Solo muestra imágenes con bbox sospechoso")
    args   = parser.parse_args()
    source = Path(args.source)

    if not source.exists():
        print(f"ERROR: '{source}' no existe.")
        return

    # Detectar si es carpeta de clase o dataset raíz
    subdirs = sorted([d for d in source.iterdir() if d.is_dir()])
    has_images_direct = any(
        f.suffix.lower() in VALID_EXT
        for f in source.iterdir() if f.is_file()
    )

    if has_images_direct:
        folders = [source]
    else:
        folders = subdirs

    all_deletes = []

    for folder in folders:
        print(f"\n  Clase: {folder.name}")
        deletes = visualize_directory(
            folder,
            save=args.save,
            suspicious_only=args.suspicious_only
        )
        all_deletes.extend(deletes)

    # Guardar lista de archivos marcados para borrar
    if all_deletes:
        out = source / "delete_list.txt"
        with open(out, 'w') as f:
            for p in all_deletes:
                f.write(p + '\n')
        print(f"\n  {len(all_deletes)} imágenes marcadas → {out}")
        print(f"  Revisa el archivo y borra manualmente si confirmas.")

    if args.save:
        print(f"\n  Previews guardadas en {source}/_bbox_preview/")


if __name__ == "__main__":
    main()