import cv2
import numpy as np
from pathlib import Path

img = cv2.imread('./Pyramid_of_Unas/pyramid_of_Unas1.jpg')
h, w = img.shape[:2]

# La imagen tiene ~10 columnas visibles
# Cortar en strips verticales
n_cols = 10
col_width = w // n_cols

output_dir = Path('./unas_strips/')
output_dir.mkdir(exist_ok=True)

strips = []
for i in range(n_cols):
    x1 = i * col_width
    x2 = min((i + 1) * col_width, w)
    strip = img[:, x1:x2]
    
    # Cada strip cortarlo en segmentos de ~5 signos
    # Asumiendo ~30 signos por columna, cada signo ~h/30 px
    signs_per_col = 25
    seg_height = h // signs_per_col * 4  # grupos de 4 signos
    
    for j in range(0, h, seg_height):
        segment = strip[j:min(j+seg_height, h), :]
        if segment.shape[0] > 50:  # descartar segmentos muy pequeños
            fname = output_dir / f'unas_col{i:02d}_seg{j:04d}.jpg'
            cv2.imwrite(str(fname), segment)
            strips.append(fname)

print(f"Segmentos generados: {len(strips)}")