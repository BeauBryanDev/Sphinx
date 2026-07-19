import cv2
import numpy as np
import argparse
from pathlib import Path
 
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
 
 
def detect_column_separators(img: np.ndarray,
                               min_gap_width: int = 8) -> list[int]:
    """
    Detecta las posiciones x de los separadores verticales entre columnas.
    Los separadores son bandas verticales con pocos píxeles oscuros.
    Retorna lista de coordenadas x donde están los separadores.
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
             if len(img.shape) == 3 else img.copy()
    h, w   = gray.shape
 
    # Binarizar — signos son oscuros sobre fondo claro
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_img = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    # Perfil vertical: fracción de píxeles oscuros por columna
    col_profile = bin_img.mean(axis=0) / 255.0
 
    # Suavizar el perfil para evitar ruido
    kernel      = np.ones(5) / 5
    col_smooth  = np.convolve(col_profile, kernel, mode='same')
 
    # Separadores = columnas con pocos píxeles oscuros (<8% del alto)
    SEPARATOR_THRESHOLD = 0.08
    is_separator = col_smooth < SEPARATOR_THRESHOLD
 
    # Agrupar píxeles consecutivos de separador
    separators = []
    in_sep     = False
    sep_start  = 0
 
    for x in range(w):
        if is_separator[x] and not in_sep:
            in_sep    = True
            sep_start = x
        elif not is_separator[x] and in_sep:
            in_sep   = False
            sep_width = x - sep_start
            if sep_width >= min_gap_width:
                # Centro del separador
                separators.append((sep_start + x) // 2)
 
    # Añadir bordes de la imagen
    separators = [0] + separators + [w]
    return separators
 
 
def detect_row_separators(col_img: np.ndarray,
                           min_gap_height: int = 6) -> list[int]:
    """
    Detecta separadores horizontales dentro de una columna.
    Retorna posiciones y de los separadores.
    """
    gray = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY) \
           if len(col_img.shape) == 3 else col_img.copy()
    h, w = gray.shape
 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_img = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    # Perfil horizontal: fracción de píxeles oscuros por fila
    row_profile = bin_img.mean(axis=1) / 255.0
    kernel      = np.ones(3) / 3
    row_smooth  = np.convolve(row_profile, kernel, mode='same')
 
    SEPARATOR_THRESHOLD = 0.04
    is_separator = row_smooth < SEPARATOR_THRESHOLD
 
    separators = []
    in_sep     = False
    sep_start  = 0
 
    for y in range(h):
        if is_separator[y] and not in_sep:
            in_sep    = True
            sep_start = y
        elif not is_separator[y] and in_sep:
            in_sep    = False
            sep_height = y - sep_start
            if sep_height >= min_gap_height:
                separators.append((sep_start + y) // 2)
 
    separators = [0] + separators + [h]
    return separators
 
 
def segment_image(img: np.ndarray,
                  signs_per_segment: int = 4,
                  min_segment_size: int = 60,
                  margin: int = 4) -> list[np.ndarray]:
    """
    Segmenta la imagen en crops de signs_per_segment signos cada uno.
    Retorna lista de imágenes recortadas.
    """
    h, w    = img.shape[:2]
    segments = []
 
    # Detectar columnas
    col_seps = detect_column_separators(img)
 
    for i in range(len(col_seps) - 1):
        x1 = max(0, col_seps[i] + margin)
        x2 = min(w, col_seps[i + 1] - margin)
 
        if (x2 - x1) < min_segment_size:
            continue
 
        col_img = img[:, x1:x2]
 
        # Detectar filas dentro de la columna
        row_seps = detect_row_separators(col_img)
 
        # Agrupar en segmentos de signs_per_segment signos
        for j in range(0, len(row_seps) - 1, signs_per_segment):
            y1 = max(0, row_seps[j] + margin)
            y2_idx = min(j + signs_per_segment, len(row_seps) - 1)
            y2 = min(h, row_seps[y2_idx] - margin)
 
            if (y2 - y1) < min_segment_size:
                continue
 
            segment = img[y1:y2, x1:x2]
            if segment.size > 0:
                segments.append(segment)
 
    # Fallback: si no se detectaron separadores, cortar en grid uniforme
    if len(segments) < 3:
        n_cols = 8
        n_rows = 6
        col_w  = w // n_cols
        row_h  = h // n_rows
 
        for col in range(n_cols):
            for row in range(0, n_rows - signs_per_segment + 1, 2):
                x1 = col * col_w
                x2 = min((col + 1) * col_w, w)
                y1 = row * row_h
                y2 = min((row + signs_per_segment) * row_h, h)
                seg = img[y1:y2, x1:x2]
                if seg.shape[0] > min_segment_size and \
                   seg.shape[1] > min_segment_size:
                    segments.append(seg)
 
    return segments
 
 
def preview_segmentation(img: np.ndarray,
                          col_seps: list[int]) -> np.ndarray:
    """Genera imagen con las columnas detectadas marcadas en rojo."""
    preview = img.copy()
    for x in col_seps[1:-1]:
        cv2.line(preview, (x, 0), (x, img.shape[0]),
                 (0, 0, 255), 2)
    return preview
 
 
def process_directory(source: Path,
                       output: Path,
                       signs_per_segment: int = 4,
                       preview: bool = False) -> None:
    images = [f for f in source.iterdir()
              if f.is_file() and f.suffix.lower() in VALID_EXT]
 
    if not images:
        print(f"No images found in {source}")
        return
 
    output.mkdir(parents=True, exist_ok=True)
    if preview:
        (output / '_preview').mkdir(exist_ok=True)
 
    total_segments = 0
 
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
 
        h, w = img.shape[:2]
        print(f"\n  {img_path.name} ({w}x{h}px)")
 
        # Preview de columnas detectadas
        if preview:
            col_seps = detect_column_separators(img)
            prev     = preview_segmentation(img, col_seps)
            cv2.imwrite(str(output / '_preview' /
                            f"{img_path.stem}_cols.jpg"), prev)
            print(f"    Columnas detectadas: {len(col_seps)-2}")
 
        # Segmentar
        segments = segment_image(img,
                                  signs_per_segment=signs_per_segment)
 
        for idx, seg in enumerate(segments):
            out_name = f"{img_path.stem}_seg_{idx:03d}.jpg"
            out_path = output / out_name
            cv2.imwrite(str(out_path), seg,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
 
        total_segments += len(segments)
        print(f"    Segmentos generados: {len(segments)}")
 
    print(f"\n{'='*50}")
    print(f"  Total segmentos: {total_segments}")
    print(f"  Guardados en   : {output}")
    print(f"  Próximo paso   : subir a Roboflow y etiquetar")
    print(f"{'='*50}")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="SphinxEyes — Segmentador de paredes de pirámides"
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Directorio con imágenes de paredes")
    parser.add_argument("--output", type=str, default="./unas_strips",
                        help="Directorio de salida (default: ./unas_strips)")
    parser.add_argument("--signs", type=int, default=4,
                        help="Signos por segmento (default: 4)")
    parser.add_argument("--preview", action="store_true",
                        help="Genera preview con columnas marcadas en rojo")
    args = parser.parse_args()
 
    process_directory(
        source           = Path(args.source),
        output           = Path(args.output),
        signs_per_segment= args.signs,
        preview          = args.preview
    )
 
 
if __name__ == "__main__":
    main()
 