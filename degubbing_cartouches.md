# Debugging: why the V4 model overlooks cartouches

Diagnostic script: `cartouche_det.py` — runs the ONNX model over every
`cartouche_*.{png,jpeg,jpg,webp}` in the repo root and inspects the RAW
cartouche-class score (class index **9**) for every anchor, BEFORE any conf
threshold or NMS. Preprocessing = letterbox @1024, identical to the production
global pass.

## Results

| Image | Size | Max cartouche score | Verdict |
|---|---|---|---|
| `cartouche_1.webp` | 992×662 | **0.068** | MISSED |
| `cartouche_2.jpg` | 408×612 | **0.056** | MISSED |
| `cartouche_3.jpg` | 866×1390 | 0.801 | ✅ DETECTED (9 anchors) |
| `cartouche_4.jpeg` | 447×447 | 0.113 | SUB-THRESHOLD |
| `cartouche_5.jpeg` | 387×516 | 0.870 | ✅ DETECTED (58 anchors) |
| `cartouche_6.jpeg` | 377×530 | 0.153 | SUB-THRESHOLD |

## What this proves about the "overlooking"

**It is not a threshold-tuning problem, and not a class-confusion problem.**
Two diagnostics rule those out:

1. **`winner @ best anchor` is `'cartouche'` on every image** — even on the
   misses. So the model is *not* mistaking cartouches for some other Gardiner
   class. When it fires at all, cartouche wins its anchor cleanly. Lowering
   `CARTOUCHE_CONF` below 0.25 wouldn't rescue images 1 & 2 — their score
   (0.06) is floor-level noise, indistinguishable from background.

2. **The split is bimodal and correlates with input scale.** The model either
   sees the oval strongly (0.80–0.87) or essentially not at all (0.06–0.15).
   There is no middle. The two clean DETECTs are the two tallest images
   (1390 px and 516 px of a narrow crop); the hard MISSes are the wide/low-res
   ones (992×662, 408×612).

## Why — root cause

The oval is dying in **letterbox downscaling to 1024**. A cartouche is a large,
thin, low-frequency outline. On a wide image like `cartouche_1` (992×662),
letterbox barely rescales, but the oval's stroke is a handful of pixels wide and
the enclosing shape spans most of the frame — **larger than anything the model
saw in training**. V4 was fine-tuned on cartouches that occupy a
*sign-cluster-sized* region inside a wall, not ones that fill the whole crop. A
cartouche that is basically the entire image has no surrounding context and an
aspect/scale the anchors weren't trained for → the objectness for that oval
never accumulates → 0.06.

The DETECTED cases (3, 5) are tall crops where the cartouche sits as a *column*
with clear margin around it — much closer to the training distribution.

**In one line:** the model isn't confusing cartouches or thresholding them out —
it's failing to fire because these standalone, frame-filling cartouche crops are
out-of-distribution in scale/context versus the in-wall cartouches V4 was
trained on.

## Cheap things to confirm it

- Re-run with **`mode='stretch'`** instead of letterbox — if scores on 1 & 2
  jump, it's purely the aspect/scale squash.
- **Pad each crop** (e.g. paste the cartouche onto a 2–3× larger neutral canvas)
  before inference — if that lifts the misses, it confirms "oval too big / no
  context," and points to a training-data fix (add zoomed-in standalone
  cartouches to V5) rather than any pipeline change.
