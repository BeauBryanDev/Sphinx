import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

ONNX_PATH = "best.onnx"  
TEST_IMG   = "test_image.jpg"  

gardiner_names = [
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
sess    = ort.InferenceSession(ONNX_PATH,
          providers=['CUDAExecutionProvider','CPUExecutionProvider'])

inp     = sess.get_inputs()[0]
out     = sess.get_outputs()[0]

print(f"Input  name  : {inp.name}")
print(f"Input  shape : {inp.shape}")   # [1, 3, 1024, 1024] esperado
print(f"Output name  : {out.name}")
print(f"Output shape : {out.shape}")   # [1, 154, ~21504] esperado

assert inp.shape[2] == 1024 and inp.shape[3] == 1024, \
    "ERROR: imgsz no es 1024"
assert out.shape[1] == 4 + len(gardiner_names), \
    f"ERROR: output channels {out.shape[1]} != {4 + len(gardiner_names)}"
print(f"\n✓ Shape correcto: [1, {out.shape[1]}, {out.shape[2]}]")
print(f"  = 4 bbox coords + {len(gardiner_names)} clases")

# ── 2. Verificar orden de clases con una inferencia real ──
img = cv2.imread(TEST_IMG)
img = cv2.resize(img, (1024, 1024))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x   = img.astype(np.float32) / 255.0
x   = np.transpose(x, (2, 0, 1))[None]  # [1, 3, 1024, 1024]

raw = sess.run([out.name], {inp.name: x})[0]  # [1, 154, N]
raw = raw[0]  # [154, N]

# Extraer scores de clase (columnas 4-153)
class_scores = raw[4:, :]          # [150, N]
max_scores   = class_scores.max(axis=1)   # max score por clase

print(f"\nTop 10 clases con mayor activación máxima:")
top10 = np.argsort(max_scores)[::-1][:10]
for idx in top10:
    print(f"  [{idx:3d}] {gardiner_names[idx]:12s} : {max_scores[idx]:.4f}")

# ── 3. Verificar que 'cartouche' está en índice 9 ──
cartouche_idx = gardiner_names.index('cartouche')
print(f"\nCartouche index : {cartouche_idx}  "
      f"(esperado: 9)")
print(f"Unknown index   : {gardiner_names.index('unknown')}  "
      f"(esperado: 120)")
print(f"Aa15 index      : {gardiner_names.index('Aa15')}  "
      f"(esperado: 149)")

# ── 4. Verificar distribución de scores ──
all_class_scores = class_scores.flatten()
print(f"\nDistribución de scores (sigmoid, no suman 1):")
print(f"  Min  : {all_class_scores.min():.4f}")
print(f"  Max  : {all_class_scores.max():.4f}")
print(f"  Mean : {all_class_scores.mean():.4f}")
print(f"  >0.25: {(all_class_scores > 0.25).sum()} anchors")
print(f"  >0.50: {(all_class_scores > 0.50).sum()} anchors")
print(f"  >0.80: {(all_class_scores > 0.80).sum()} anchors")

print(f"\n✓ Phase 1 verificación completa")
print(f"  Si todos los asserts pasaron, el ONNX está listo para Phase 2")
