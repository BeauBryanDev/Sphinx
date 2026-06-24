# SphinxEyes

End-to-end computer-vision and NLP pipeline for the detection, reading-order
assembly, semantic correction, and transliteration of Middle Egyptian
hieroglyphs from raw photographic input.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Detection Model — V4](#3-detection-model--v4)
4. [Spatial Layer](#4-spatial-layer)
5. [NLP Correction Layer](#5-nlp-correction-layer)
6. [Cartouche Matching](#6-cartouche-matching)
7. [FastAPI Backend](#7-fastapi-backend)
8. [Frontend](#8-frontend)
9. [Repository Structure](#9-repository-structure)
10. [Artifacts](#10-artifacts)
11. [Running the Pipeline](#11-running-the-pipeline)
12. [Docker Deployment](#12-docker-deployment)
13. [Training — V4 Summary](#13-training--v4-summary)
14. [Roadmap](#14-roadmap)

---

## 1. Project Overview

SphinxEyes addresses the automated reading of Middle Egyptian hieroglyphic
inscriptions. The problem is non-trivial: glyphs appear in dense, multi-row
or multi-column layouts carved or painted on stone; individual signs are
visually similar across dozens of Gardiner classes; reading direction
(left-to-right vs right-to-left, horizontal vs vertical) is determined by
the orientation of figurative signs rather than by a fixed convention; and a
significant proportion of signs occur inside cartouches — oval enclosures used
to frame royal names — whose interior layout must be treated as an independent
reading unit.

The pipeline covers the full chain from raw image bytes to a structured,
corrected, transliterated output:

```
raw image
  -> image enhancement (CLAHE + denoising + unsharp mask)
  -> YOLOv11-Large ONNX detection (150 Gardiner classes)
  -> layout classification (rows vs columns, geometry-driven)
  -> reading-order assembly (quadrat clustering)
  -> semantic correction (Viterbi over lexicon trie + bigram LM)
  -> cartouche interior reading (Needleman-Wunsch against royal-name database)
  -> structured JSON output (FastAPI)
```

The corpus used for the NLP layer is the BBAW Middle Egyptian corpus
(`bbaw_clean.parquet`), which provides attested Gardiner code sequences for
bigram probability estimation.

<img src="horus_eyes.webp" alt="SphinxEyes — Horus Eyes" width="320" />

---

## 2. System Architecture

```
+------------------------------------------------------------------+
|  Client  (React 18 / Vite / Tailwind 3.4)                        |
|    POST multipart/form-data  ->  /predict                        |
+----------------------------+-------------------------------------+
                             | HTTP
+----------------------------v-------------------------------------+
|  FastAPI  (uvicorn, single worker)                                |
|  app/                                                             |
|  +-- core/lifespan.py   -- SphinxPipeline loaded once at startup  |
|  +-- routers/predict.py -- POST /predict                          |
|  +-- routers/health.py  -- GET /health                            |
|  +-- services/          -- InferenceService (pipeline -> schema)  |
|  +-- schemas/           -- Pydantic I/O contracts                 |
|  +-- utils/             -- image decoding, validation             |
+----------------------------+-------------------------------------+
                             | Python import
+----------------------------v-------------------------------------+
|  pipeline.py  (SphinxPipeline)                                    |
|  +-- enhance_img.py        -- image preprocessing                 |
|  +-- spatial_logic.py      -- ONNX post-processing, NMS, RO      |
|  +-- layout_detector.py    -- geometry-driven layout vote         |
|  +-- sphinx_corrector.py   -- Viterbi + bigram LM                 |
|  +-- cartouche_matcher.py  -- Needleman-Wunsch royal-name match  |
+----------------------------+-------------------------------------+
                             |
+----------------------------v-------------------------------------+
|  artifacts/                                                       |
|  +-- best_model_v4.onnx              (97.5 MB, opset 18)          |
|  +-- class_map50_v4.json             (150 classes)                |
|  +-- sphinx_trie_v4.pkl              (38,073 entries)             |
|  +-- bbaw_clean.parquet              (BBAW corpus)                |
|  +-- confusion_matrix_v4_normalized.csv                           |
+------------------------------------------------------------------+
```

All heavy artifacts are loaded once at process startup via the FastAPI
lifespan context manager and stored on `app.state.pipeline`. Request handlers
receive the pipeline through FastAPI dependency injection (`routers/deps.py`).

---

## 3. Detection Model — V4

**Architecture:** YOLOv11-Large, fine-tuned from V3 best checkpoint.  
**Export format:** ONNX opset 18, static shape `[1, 154, 21504]`,
`nms=False` (NMS is handled in `spatial_logic.py` for full top-3 access).

### Class definitions

- 148 Gardiner sign classes (most-frequent signs in Middle Egyptian)
- 1 `cartouche` class (the oval bracket enclosure)
- 1 `unknown` class (signs outside the 148-class vocabulary)

Class-index order is fixed: `Aa1=0, A1=1, ... Z11=148, Aa15=149`. This order
is authoritative and must not change across model versions.

### V4 training configuration

| Parameter | Value |
|---|---|
| Base checkpoint | V3 `best.pt` (fine-tune, not from scratch) |
| Training images | 1,253 |
| Training instances | 53,706 |
| Validation images | 151 |
| Validation instances | 5,093 |
| Best epoch | 44 / 50 |
| Image size | 1024 px |
| Optimizer | SGD, lr0=0.0015, momentum=0.92 |
| Augmentation | mosaic=0.3, degrees=3.0, no flips |
| Frozen layers | 5 |
| Early stopping patience | 12 epochs |

### V4 validation metrics

| Metric | Value |
|---|---|
| mAP50 | 0.884 |
| mAP50-95 | 0.654 |
| Precision | 0.896 |
| Recall | 0.827 |

Key improvement over V3: cartouche interiors were explicitly labeled in the
V4 training set (+45 images with hand-labeled interior signs). Sign E34
(hare, found in the royal name *Unas*) is now detected natively at mAP50
0.468, surfacing as top-1 at confidence 0.84–0.92 on Unas pyramid-text
fixtures. Cartouche re-entry inference (a V3 fallback) is eliminated.

Known low-recall classes: `d56` (mAP50 0.00, 1 val instance), `n23` (0.19),
`i1` (0.27). Predictions on these classes should be treated with reduced
confidence.

### Inference parameters (defaults)

| Parameter | Value | Description |
|---|---|---|
| `imgsz` | 1024 | Input resolution (stretch, not letterbox) |
| `conf_threshold` | 0.25 | Minimum confidence to retain a detection |
| `iou_threshold` | 0.45 | NMS IoU threshold |
| `max_detections` | 300 | Cap per image after NMS |

Resize mode is **stretch** (not letterbox) throughout — this matches the
training distribution, where all source images were squashed to 224x224 via
`ETL/batch_resize.py`.

---

## 4. Spatial Layer

**File:** `spatial_logic.py`

The spatial layer converts raw ONNX tensor output into an ordered reading
sequence.

### Post-processing (`postprocess_onnx`)

The model output tensor `[1, 154, N]` contains N anchors, each with 4
coordinate values and 150 class logits. Processing steps:

1. Apply sigmoid to class logits.
2. Filter anchors with `max_class_score < conf_threshold`.
3. Run class-agnostic NMS on surviving anchors (IoU threshold 0.45).
   Class-agnostic NMS is used deliberately: a single physical glyph
   predicted under two confusable classes must collapse to one Detection;
   the alternative classes are preserved in the top-3 slot.
4. Extract top-3 `(class_name, score)` pairs per surviving anchor.

Top-3 extraction runs only on NMS survivors (typically dozens), never on
the full anchor set of ~21,504.

### Cartouche membership tagging (`tag_cartouche_members`)

Detections are tested for centroid containment within each cartouche bbox.
Detections whose centroid falls inside a cartouche bbox are tagged
`inside_cartouche=True` and assigned a `cartouche_id`.

Edge-clipped cartouches are removed before the layout vote
(`_filter_edge_cartouches`, margin = 1% of image dimension). A cartouche
with any bbox edge within this margin is a photo-crop artifact with an
incomplete interior and is discarded. This filter must run before the layout
vote to prevent partial cartouche bboxes from corrupting the cartouche-aspect
signal.

### Layout detection (`layout_detector.py`)

Layout (rows vs columns) is determined geometrically from the detections
themselves, not from pixel heuristics. The voter considers cartouche aspect
ratio, vertical band count, and sign aspect ratio distribution. The legacy
pixel-level function `detect_layout(img)` is retained for callers that do
not have detections available.

### Reading-order assembly

Signs are grouped into quadrats (rectangular spatial cells) via density-based
clustering, then ordered according to detected layout and direction. Cartouche
interiors are assembled with `single_line=True`. Without this flag, the
line-grouping logic splits the interior sequence and introduces spurious
Unknown tokens.

---

## 5. NLP Correction Layer

**File:** `sphinx_corrector.py`

The correction layer operates on the top-3 YOLO candidates per slot and
recovers the most linguistically plausible Gardiner code sequence.

### Algorithms

**Bigram language model** — Markov chain, order 1, built from BBAW
`gardiner_seq`. Transition probabilities stored as log-probabilities.
Laplace smoothing applied for zero-count bigrams.

**Viterbi segmenter** — Dynamic programming over the flat YOLO sequence.
At each position, all candidate segments of length 1 to `MAX_WORD_LEN` are
scored as:

```
score = log(freq + 1)                          # trie record frequency
      + log P(segment[0] | prev_last_code)     # bigram transition
      + edit_penalty                           # 0.0 exact / -1.0 dist=1 / -2.5 dist=2
      + linguistic_prior                       # determinative / particle / layout bonus
      + substitution_cost                      # from V4 confusion matrix
```

Backtracking from `dp[N]` recovers the optimal segmentation. If `dp[N]`
remains at `-inf`, the segmenter degrades gracefully to single-sign segments.

**Beam search over top-3 candidates** — The beam maintains the B
highest-scoring partial sequences and expands each with all 3 YOLO candidates
per slot. The highest-scoring complete path is selected.

**Confidence gate (Naive Bayes)** — Prevents the corrector from overriding
a high-confidence YOLO prediction. If the posterior probability of the
correction being correct falls below threshold 0.30, the original YOLO code
is kept.

**Unknown-slot resolution** — When YOLO outputs `Unknown` at a slot, the
trie enumerates child edges at that depth in surviving beam nodes and proposes
the most frequent real Gardiner code as a replacement.

**Substitution cost matrix** — Loaded from
`artifacts/confusion_matrix_v4_normalized.csv`. Orientation: rows=predicted,
cols=truth, cell = P(truth | predicted). Cost = `-log(P + epsilon)`,
epsilon=1e-4, capped at ~9.21.

### Linguistic priors

| Prior | Bonus | Trigger |
|---|---|---|
| `DETERMINATIVE_BONUS` | 0.5 | Sign appears in `determinatives.json` |
| `PARTICLE_BONUS` | 1.0 | Sign appears in `initial_particles.json` |
| `LAYOUT_BONUS` | 1.5 | Sign coincides with a reading-order boundary hint |

### Entry point

```python
from sphinx_corrector import correct, load_corrector

trie, log_prob, unigrams, sub_cost = load_corrector(
    trie_pkl     = 'artifacts/sphinx_trie_v4.pkl',
    bbaw_parquet = 'artifacts/bbaw_clean.parquet',
    confusion_csv= 'artifacts/confusion_matrix_v4_normalized.csv',
)

result = correct(yolo_topk, trie, log_prob, unigrams, sub_cost_matrix=sub_cost)
print(result['flat_translit'])
```

`load_corrector()` returns a 4-tuple `(trie, log_prob, unigrams, sub_cost)`.

---

## 6. Cartouche Matching

**File:** `cartouche_matcher.py`

Each cartouche interior sequence is aligned against a database of attested
royal names (`royal_names.json`) using the Needleman-Wunsch global alignment
algorithm. The alignment score incorporates the V4 substitution cost matrix
so that visually similar confusions are penalized less than semantically
distant ones.

A panel-consensus step (`apply_panel_consensus`) reconciles multiple
cartouches detected in the same image. If one cartouche produces a
high-confidence direct match and another a weak or refused match, the
consensus promotes the stronger identity where geometrically consistent.

Royal name normalization (`normalize_code`) maps YOLO-lowercase class names
(e.g., `g17`) to canonical Gardiner codes (e.g., `G17`). The mapping is
defined in `gardiner_Map.py` and is the single normalization source of truth
throughout the pipeline.

### V4 results on canonical fixtures

| Image | Layout | Detections | Outer score | Cartouche result |
|---|---|---|---|---|
| `Unas1c.jpg` | columns | 100 | 1.39 | 2/2 direct *wnjs* (E34 native) |
| `image_2_test.jpg` | columns | 77 | 2.08 | 3/3 direct *wnjs* |
| `glyph_wall.jpeg` | columns | 60 | 3.47 | 0 cartouches in scene |

---

## 7. FastAPI Backend

**Directory:** `app/`  
**Entry point:** `uvicorn app.main:app`

### Structure

```
app/
+-- main.py                     # FastAPI factory, CORS middleware, lifespan
+-- core/
|   +-- config.py               # pydantic-settings: all paths and thresholds
|   +-- lifespan.py             # artifact validation + SphinxPipeline construction
+-- models/                     # Internal domain type wrappers
+-- schemas/                    # Pydantic I/O contracts (API surface only)
+-- services/
|   +-- sphinx_inference.py     # InferenceService.run() — pipeline call + schema mapping
+-- utils/
|   +-- image_utils.py          # decode_upload -> BGR ndarray, validation
|   +-- post_process_utils.py   # low-level ONNX helpers (thin at Phase 5)
+-- routers/
    +-- deps.py                 # get_pipeline(request) -> SphinxPipeline
    +-- predict.py              # POST /predict
    +-- health.py               # GET /health
```

`pipeline.py` remains at the repository root. It is imported by
`services/sphinx_inference.py` and is independently runnable as a CLI smoke
test. All domain scripts (`spatial_logic`, `sphinx_corrector`,
`cartouche_matcher`, etc.) also remain at root.

### API

**`POST /predict`** — `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | `UploadFile` | required | Raw image (JPEG, PNG, WebP) |
| `direction` | `str` | `rtl` | Reading direction: `rtl` or `ltr` |
| `layout` | `str or null` | null | `rows`, `columns`, or auto-detect |
| `preset` | `str` | `default` | Enhancement preset: `default`, `aggressive`, `gentle` |

Returns `PredictResponse` (JSON):

```json
{
  "layout": "columns",
  "direction": "rtl",
  "image_shape": [1024, 768],
  "n_detections": 77,
  "n_cartouches": 3,
  "outer": {
    "flat_corrected_seq": ["G17", "N35", "D21"],
    "flat_translit": "m n r",
    "score": 2.08,
    "had_fallback": false
  },
  "cartouches": [
    {
      "translit": "wnjs",
      "english": "Unas",
      "score": 0.91,
      "inferred": false,
      "verified": true,
      "n_members": 6
    }
  ]
}
```

**`GET /health`** — Returns pipeline status, class count, and ONNX output
shape. Used by container orchestrators as a liveness probe.

### Configuration

All settings are read from environment variables or `.env` at startup.

| Variable | Default | Description |
|---|---|---|
| `ONNX_PATH` | `artifacts/best_model_v4.onnx` | ONNX weights path |
| `CLASS_MAP` | `artifacts/class_map50_v4.json` | Class index map |
| `TRIE_PKL` | `artifacts/sphinx_trie_v4.pkl` | Serialized lexicon trie |
| `BBAW_PARQUET` | `artifacts/bbaw_clean.parquet` | Bigram LM corpus |
| `CONFUSION_CSV` | `artifacts/confusion_matrix_v4_normalized.csv` | Sub-cost matrix |
| `IMGSZ` | `1024` | ONNX inference resolution |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `MAX_UPLOAD_MB` | `20` | Hard cap on uploaded image size |

---

## 8. Frontend

**Directory:** `frontend/`  
**Target stack:** TypeScript, React 18, Vite 4/5, Tailwind CSS 3.4,
Recharts, Axios.

The UI follows an Egyptian gold and desert amber visual language — warm ochre
backgrounds, gold-toned accent colours, and hieroglyph-inspired typographic
detail — referencing the visual palette of Middle Kingdom tomb and pyramid
inscriptions. Design reference: `Sphinx_Eyes_Mockup.png` in the repo root.

Current `package.json` references React 19 / Vite 7 / Tailwind 4. A full
dependency downgrade and configuration rewrite is scheduled after the FastAPI
backend is wired. The component structure and routing logic are complete and
do not require changes.

### Stack

| Layer | Library | Version target |
|---|---|---|
| Language | TypeScript | 5.x |
| UI framework | React | 18.x |
| Build tool | Vite | 4 or 5 |
| Styling | Tailwind CSS | 3.4 |
| Charts | Recharts | 2.x |
| HTTP client | Axios | 1.x |

### Structure

```
frontend/src/
+-- api/           -- glyphApi.ts, chatApi.ts, analyticsApi.ts
+-- components/    -- ChatPanel, GlyphDecoder, ActivityChart, layout/
+-- hooks/         -- useGlyphDecoder, useChat, useAnalytics
+-- pages/         -- HomePage, GlyphsPage, TransliterationPage, LearnPage
+-- services/      -- http.ts (Axios instance, VITE_API_BASE_URL)
+-- types/         -- GlyphDecodingResult, ChatMessage, UserProfile
+-- assets/        -- UI imagery (Horus eyes, Sphinx, pyramid background)
```

### Backend adapter gap

`glyphApi.ts` currently calls `POST /v1/glyphs/decode` expecting
`GlyphDecodingResult`. The backend serves `POST /predict` returning
`PredictResponse`. The URL and response shape mapping must be implemented in
`glyphApi.ts` at wiring time — not in any component.

---

## 9. Repository Structure

```
.
+-- pipeline.py                  # SphinxPipeline -- CLI + FastAPI import target
+-- spatial_logic.py             # Detection, NMS, reading-order assembly
+-- sphinx_corrector.py          # Viterbi correction + bigram LM
+-- sphinx_trie.py               # SphinxTrie class (module-scope, pickle-safe)
+-- cartouche_matcher.py         # Needleman-Wunsch royal-name alignment
+-- layout_detector.py           # Geometry-driven layout classification
+-- enhance_img.py               # CLAHE + denoising + unsharp + gamma
+-- wall_segmenter.py            # Multi-panel wall segmentation
+-- build_master_trie.py         # Rebuild sphinx_trie_v4.pkl from sources
+-- mine_determinatives.py       # Rebuild determinatives.json from dickson.csv
+-- validate_onnx.py             # ONNX sanity-check (shape, sigmoid, class order)
+-- gardiner_Map.py              # GARDINER_MAP -- normalization source of truth
+-- royal_names.json             # Attested royal-name database
+-- determinatives.json          # Determinative sign set
+-- initial_particles.json       # Initial particle sign set
|
+-- app/                         # FastAPI backend (Phase 5)
+-- artifacts/                   # Model weights and NLP artifacts (not committed)
+-- ETL/                         # Dataset preparation scripts
+-- frontend/                    # React frontend
+-- TrainingV2/                  # YOLO-format training export
+-- Glyph2025/                   # Raw scraped glyph corpus (~349 classes)
+-- SphinxEyes_Final/            # Curated per-class source-of-truth
+-- Backups/                     # V3 artifacts (reference only)
|
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- .env                         # Local environment overrides (not committed)
+-- CLAUDE.md                    # AI assistant project context
```

---

## 10. Artifacts

All model and NLP artifacts are stored under `artifacts/` and are not
committed to version control.

| File | Size | Description |
|---|---|---|
| `best_model_v4.onnx` | 97.5 MB | YOLOv11-Large V4 weights, opset 18 |
| `class_map50_v4.json` | — | 150-class index map with per-class mAP50 |
| `sphinx_trie_v4.pkl` | 23.1 MB | Serialized SphinxTrie (38,073 entries) |
| `bbaw_clean.parquet` | — | BBAW Middle Egyptian corpus (bigram source) |
| `confusion_matrix_v4_normalized.csv` | — | Row-normalized confusion, P(truth\|pred) |

The normalized confusion matrix is derived from `confusion_matrix_v4_raw.csv`
by dropping the background row/column and row-normalizing so that each row
sums to 1.0.

---

## 11. Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

Python 3.11 or 3.12 recommended. ONNX Runtime CPU is the default provider.

### CLI smoke test

```bash
python pipeline.py Unas1c.jpg
```

Runs the full pipeline on a single image and prints layout, detection count,
corrected outer-text sequence, transliteration, and cartouche matches.

### Component self-tests

```bash
python sphinx_corrector.py    # 5 correction tests
python spatial_logic.py       # 14 spatial layer tests
python cartouche_matcher.py   # royal-name alignment self-tests
python validate_onnx.py       # ONNX shape and output validation
```

### FastAPI development server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server validates all artifact paths at startup and raises
`FileNotFoundError` listing every missing file if any artifact is absent.

---

## 12. Docker Deployment

### Infrastructure overview

```
docker compose
+-- sphinx_pyramid    FastAPI backend          :8000  (public)
+-- sphinx_memory     PostgreSQL 16            :5437  (localhost only)

Network : sphinx_net  (bridge, internal service discovery by name)
Volume  : sphinx_db_data  (named, persists across restarts)
```

### Build

The Dockerfile uses a two-stage build:

| Stage | Base | Purpose |
|---|---|---|
| `builder` | `python:3.12-slim` | Compile wheels (numpy, asyncpg, cryptography) into `/opt/venv` |
| `runtime` | `python:3.12-slim` | Copy venv only; no build toolchain in the final image |

The runtime image runs as a non-root `sphinx` user (uid 1000). Build
toolchain (`gcc`, `build-essential`) is present only in the builder stage
and does not appear in the final image.

Runtime OS dependencies installed in the final image:

| Package | Reason |
|---|---|
| `libpq5` | asyncpg runtime |
| `libgomp1` | OpenMP runtime for ONNX Runtime CPU provider |
| `libgl1`, `libglib2.0-0` | OpenCV headless runtime |
| `curl` | Container health check |

### Starting services

```bash
# First run: build image and start all services
docker compose up --build

# Subsequent runs
docker compose up

# Background
docker compose up -d
```

### Artifact mount

Model artifacts are mounted read-only from the host so that retraining does
not require an image rebuild:

```yaml
volumes:
  - ./artifacts:/app/artifacts:ro
```

All six artifact files must exist under `./artifacts/` before starting:

```
artifacts/
+-- best_model_v4.onnx
+-- class_map50_v4.json
+-- sphinx_trie_v4.pkl
+-- bbaw_clean.parquet
+-- confusion_matrix_v4_normalized.csv
```

### Health check

The container health check polls `GET /health/live` every 30 seconds.
Start period is 60 seconds to allow ONNX session creation and trie
deserialization (~15–20 seconds on CPU-only hardware) before the
orchestrator begins counting failures.

### Database

PostgreSQL 16 (`sphinx_memory`) is configured for Phase 6 persistence.
It is healthy-checked via `pg_isready` before `sphinx_pyramid` receives
traffic (`depends_on: condition: service_healthy`). The database port
(5432 internal) is exposed only on `127.0.0.1:5437` on the host — not
accessible from outside the machine in development.

Credentials are read from environment variables with development defaults:

```
POSTGRES_USER     = sphinx
POSTGRES_PASSWORD = sphinx_dev_password
POSTGRES_DB       = sphinxeyes
```

Override all three in `.env` before any deployment outside localhost.

---

## 13. Training — V4 Summary

Training was executed on a Google Colab A100 instance using Ultralytics
YOLOv11. The V4 dataset added 45 manually labeled images with explicit
interior cartouche annotations, enabling native detection of signs previously
invisible to V3 at global inference resolution.

Key design decisions:

- Fine-tuned from V3 `best.pt` rather than training from scratch, preserving
  learned feature representations for the 148-class Gardiner vocabulary.
- Mosaic augmentation reduced to 0.3 (from default 1.0) to avoid generating
  unrealistic sign compositions that do not occur in authentic inscriptions.
- Horizontal and vertical flips disabled — hieroglyph orientation is
  semantically significant and must not be randomized.
- Rotation limited to 3 degrees to simulate minor camera tilt without
  introducing implausible reading angles.
- Five backbone layers frozen to stabilize low-level feature extraction during
  fine-tuning.

ONNX export uses `nms=False`. All post-processing (confidence filtering,
class-agnostic NMS, top-3 extraction) is handled in `spatial_logic.py` for
full programmatic control and to allow the NLP layer access to the full
top-3 candidate distribution per anchor.

---

## 14. Roadmap

| Phase | Item | Status |
|---|---|---|
| 1–4 | CV + spatial + NLP + cartouche layers | Complete |
| 5 | FastAPI backend (`app/`) | In progress |
| 5 | Frontend wiring (`glyphApi.ts` adapter) | Pending — after backend |
| 5 | Frontend stack downgrade (React 18, Vite 4/5, Tailwind 3.4) | Pending |
| 6 | PostgreSQL schema + async persistence | Pending |
| 6 | GPT transliteration integration | Pending |
| 7 | Real-image integration test (labeled fixture, assert transliteration) | Final gate |
| — | V5 training (address `d56`, `n23`, `i1` low-recall classes) | Deferred |
