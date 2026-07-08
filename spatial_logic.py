#!/usr/bin/env python3
"""
spatial_logic.py — Phase 2 spatial layer for SphinxEyes (steps 1 + 2).

Role in the pipeline
--------------------
    ONNXRuntime raw output [1, 154, N]
        --> postprocess_onnx()       (conf filter -> NMS on max score -> top-3)
        --> List[Detection]          (bbox, centroid, top3, area)
        --> tag_cartouche_members()  (PRIMARY: centroid containment)
        --> cartouche_reentry()      (FALLBACK: inset crop + re-inference,
                                      only when containment found < 2 members)
        --> [next: quadrat clustering -> reading order -> sphinx_corrector]

Design decisions
----------------
    1. NMS is CLASS-AGNOSTIC. One physical glyph predicted as two confusable
       classes must collapse to ONE Detection; the alternatives survive in
       top3. Per-class NMS would emit duplicate boxes for the corrector.

    2. Top-3 extraction runs only on NMS survivors (~dozens), never on the
       full ~21k anchors (fragile breakpoint #7 in CLAUDE.md).

    3. Cartouche interiors come from RE-ENTRY, always. V3 was intentionally
       trained with no labels inside cartouches (curriculum decision), so
       the model is systematically blind there at global resolution. Every
       cartouche gets an inset crop + second inference; inside the crop the
       bracket context is gone and the model sees an ordinary sign column —
       its training regime. Containment tagging still runs first: it
       catches stray interior detections the model emits anyway, and the
       re-entry dedupe reconciles them.

    4. Re-entry inference is injected as `infer_fn(crop) -> list[Detection]`
       (bboxes in crop coordinates). Tests mock it; production passes the
       ONNX wrapper. Inner-pass `cartouche` detections are dropped to break
       recursion (fragile breakpoint #3); inner detections duplicating an
       existing global one (IoU > DEDUPE_IOU) update it in place when they
       score higher, instead of appending a twin.

Usage
-----
    from spatial_logic import Detection, postprocess_onnx, \
        tag_cartouche_members, cartouche_reentry

    raw  = sess.run([out.name], {inp.name: x})[0]      # [1, 154, N]
    dets = postprocess_onnx(raw, class_names)
    cartouche_idxs = tag_cartouche_members(dets)
    dets = cartouche_reentry(img, dets, cartouche_idxs, infer_fn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CARTOUCHE_CLASS         = 'cartouche'
CONF_THRESHOLD          = 0.25   # min max-class score to keep an anchor
NMS_IOU                 = 0.50   # class-agnostic NMS threshold
TOP_K                   = 3      # alternatives kept per detection
CARTOUCHE_CONF          = 0.25   # min score to treat a det as a cartouche
                                 # (was 0.50; real cartouches on weathered
                                 # stone surface at ~0.28 — sandstone_wall)
CARTOUCHE_EXPAND_FRAC   = 0.10   # expand cartouche bbox before containment
# Re-entry crop insets as (short_axis_frac, long_axis_frac) per side.
# Two complementary passes whose results are unioned via the dedupe step:
#   shallow (6%/6%)  — keeps signs hugging the bracket ends
#   deep    (6%/10%) — cuts bracket curve + tie-knot, zooms interior more
# Empirically (test_image_vn5, 6 cartouches): each pass alone misses signs
# the other finds; the union covers 6/6 with the best score per sign.
REENTRY_INSETS          = ((0.06, 0.06), (0.06, 0.10))
REENTRY_CONF            = 0.20   # conf threshold for the re-entry pass —
                                 # lower than global: interior signs were
                                 # never labeled in training, scores run low
MIN_CARTOUCHE_MEMBERS   = 2      # legacy heuristic (always=False only)
DEDUPE_IOU              = 0.50   # inner det vs global det dedupe threshold

# Step 3 — quadrat clustering
DUP_OVERLAP             = 0.55   # intersection/min-area above which two
                                 # boxes are the SAME physical sign
QUADRAT_ALIGN           = 0.50   # min projection-overlap ratio along the
                                 # reading axis to share a quadrat
QUADRAT_GAP_FRAC        = 0.60   # max perpendicular gap (x median size)
QUADRAT_SIZE_RATIO      = (0.5, 2.0)   # sqrt-area ratio guard (plan rule)
QUADRAT_MAX_SIGNS       = 4      # split components larger than this
QUADRAT_MAX_EXTENT      = 2.5    # max merged cross-extent (x median sign
                                 # size). A merged pair spanning more has
                                 # crossed into the neighbouring row/column
                                 # (sandstone_wall: g5+l2 spanned 2.9x).
                                 # NOT lower: two equal stacked signs are
                                 # ~2.2x, must stay mergeable.

# Step 4 — line assembly + gap insertion
LINE_GAP_FRAC           = 0.60   # cross-axis jump (x median) = new line
MISSING_GAP_FRAC        = 1.50   # reading-axis gap (x median step) above
                                 # which synthetic Unknown slots go in
MAX_GAP_INSERTS         = 2      # max synthetic slots per gap
UNKNOWN_CLASS           = 'unknown'   # YOLO-space name; glue maps to the
                                      # trie's 'Unknown' token


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One detected sign. bbox is (x1, y1, x2, y2) in global image pixels."""
    bbox             : tuple[float, float, float, float]
    top3             : list[tuple[str, float]]   # [(class_name, score)] desc
    inside_cartouche : bool = False
    cartouche_id     : Optional[int] = None      # index of parent cartouche
    from_reentry     : bool = False              # came from the fallback pass

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cls(self) -> str:
        return self.top3[0][0]

    @property
    def max_score(self) -> float:
        return self.top3[0][1]

    def is_cartouche(self, conf: float = CARTOUCHE_CONF) -> bool:
        return self.cls == CARTOUCHE_CLASS and self.max_score > conf


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def iou(a: tuple, b: tuple) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def expand_bbox(
    bbox: tuple, frac: float, img_w: Optional[float] = None,
    img_h: Optional[float] = None,
) -> tuple:
    """Grow a bbox by `frac` of its own size on each side; clamp to image."""
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    if img_w is not None:
        x1, x2 = max(0.0, x1), min(float(img_w), x2)
    if img_h is not None:
        y1, y2 = max(0.0, y1), min(float(img_h), y2)
    return (x1, y1, x2, y2)


def inset_bbox(
    bbox: tuple, frac: float, frac_long: Optional[float] = None,
) -> tuple:
    """
    Shrink a bbox by `frac` of its own size on each side (re-entry crop).

    If `frac_long` is given, the LONG axis of the box is inset by that
    fraction instead. For a cartouche the bracket curve and the tie-knot
    sit at the long-axis ends, so cutting deeper there removes them while
    keeping the signs (which span the short axis nearly edge to edge).
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    fl = frac if frac_long is None else frac_long
    if h >= w:                      # vertical cartouche: long axis = y
        dx, dy = w * frac, h * fl
    else:                           # horizontal cartouche: long axis = x
        dx, dy = w * fl, h * frac
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def contains_point(bbox: tuple, pt: tuple) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


# ---------------------------------------------------------------------------
# Step 1 — ONNX postprocess: raw tensor -> List[Detection]
# ---------------------------------------------------------------------------

def nms_class_agnostic(
    boxes_xyxy: np.ndarray,    # [M, 4]
    scores    : np.ndarray,    # [M]
    iou_thresh: float = NMS_IOU,
) -> list[int]:
    """Greedy class-agnostic NMS. Returns kept indices, score-descending."""
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    suppressed = np.zeros(len(order), dtype=bool)
    for rank, i in enumerate(order):
        if suppressed[rank]:
            continue
        keep.append(int(i))
        bi = boxes_xyxy[i]
        for rank2 in range(rank + 1, len(order)):
            if suppressed[rank2]:
                continue
            if iou(tuple(bi), tuple(boxes_xyxy[order[rank2]])) > iou_thresh:
                suppressed[rank2] = True
    return keep


def postprocess_onnx(
    raw         : np.ndarray,         # [1, 4+C, N] or [4+C, N]
    class_names : list[str],
    conf_thresh : float = CONF_THRESHOLD,
    iou_thresh  : float = NMS_IOU,
    top_k       : int   = TOP_K,
) -> list[Detection]:
    """
    Decode the V3 ONNX output (nms=False export) into Detection objects.

    Layout per anchor column: rows 0-3 = (cx, cy, w, h) in input-image
    pixels; rows 4..4+C-1 = independent sigmoid class scores.

    Order of operations (fragile breakpoint #7): conf-filter on max class
    score -> class-agnostic NMS -> top-k extraction on survivors only.
    """
    if raw.ndim == 3:
        raw = raw[0]
    C = len(class_names)
    assert raw.shape[0] == 4 + C, \
        f"channel mismatch: tensor has {raw.shape[0]}, expected {4 + C}"

    boxes  = raw[:4, :]              # [4, N]  cx, cy, w, h
    scores = raw[4:, :]              # [C, N]

    max_scores = scores.max(axis=0)  # [N]
    mask = max_scores >= conf_thresh
    if not mask.any():
        return []

    boxes_f  = boxes[:, mask]
    scores_f = scores[:, mask]
    max_f    = max_scores[mask]

    cx, cy, w, h = boxes_f
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    keep = nms_class_agnostic(xyxy, max_f, iou_thresh)

    detections: list[Detection] = []
    k = min(top_k, C)
    for i in keep:
        col = scores_f[:, i]
        top_idx = np.argpartition(col, -k)[-k:]
        top_idx = top_idx[np.argsort(col[top_idx])[::-1]]
        top3 = [(class_names[int(j)], float(col[j])) for j in top_idx]
        detections.append(Detection(bbox=tuple(float(v) for v in xyxy[i]),
                                    top3=top3))
    return detections


# ---------------------------------------------------------------------------
# Step 2a — Cartouche containment tagging (PRIMARY path)
# ---------------------------------------------------------------------------

def tag_cartouche_members(
    detections    : list[Detection],
    cartouche_conf: float = CARTOUCHE_CONF,
    expand_frac   : float = CARTOUCHE_EXPAND_FRAC,
    img_w         : Optional[float] = None,
    img_h         : Optional[float] = None,
) -> list[int]:
    """
    Tag every detection whose centroid sits inside an (expanded) cartouche
    bbox with inside_cartouche=True and the cartouche's index.

    The expansion compensates for the model's imprecise cartouche boxes;
    centroid containment (vs bbox-IoU) keeps signs touching the bracket
    correctly tagged. A detection inside two overlapping cartouches is
    assigned to the smaller one (tighter fit wins).

    Returns the indices of the cartouche detections themselves.
    """
    cartouche_idxs = [
        i for i, d in enumerate(detections) if d.is_cartouche(cartouche_conf)
    ]
    # Smaller cartouches assign last -> tighter fit wins on overlap
    for ci in sorted(cartouche_idxs, key=lambda i: -detections[i].area):
        zone = expand_bbox(detections[ci].bbox, expand_frac, img_w, img_h)
        for j, det in enumerate(detections):
            if j == ci or det.is_cartouche(cartouche_conf):
                continue
            if contains_point(zone, det.centroid):
                det.inside_cartouche = True
                det.cartouche_id = ci
    return cartouche_idxs


# ---------------------------------------------------------------------------
# Step 2b — Cartouche re-entry (FALLBACK path)
# ---------------------------------------------------------------------------

def cartouche_reentry(
    image          : np.ndarray,                          # HxWx3 global image
    detections     : list[Detection],
    cartouche_idxs : list[int],
    infer_fn       : Callable[[np.ndarray], list[Detection]],
    always         : bool  = True,
    min_members    : int   = MIN_CARTOUCHE_MEMBERS,
    insets         : tuple = REENTRY_INSETS,
    dedupe_iou     : float = DEDUPE_IOU,
) -> list[Detection]:
    """
    Inset-crop each cartouche, re-run inference on the crop(s), and map the
    results back to global coordinates.

    V3 was INTENTIONALLY trained with no labels inside cartouches
    (curriculum-learning decision), so the model is systematically blind
    there at global resolution. Re-entry is therefore the PRIMARY mechanism
    for cartouche interiors — `always=True` re-enters every cartouche.
    Set always=False to fall back to the legacy heuristic (re-enter only
    when containment tagged fewer than `min_members` signs).

    Each cartouche is cropped once per (short_frac, long_frac) pair in
    `insets` and the passes are UNIONED: results of earlier passes are
    appended before later passes run, so the dedupe step reconciles them,
    keeping the higher-scoring reading of each sign. Shallow insets keep
    bracket-hugging signs; deep insets remove the bracket curve / tie-knot
    and zoom the interior. Inside the crop the bracket context is gone, so
    the model sees an ordinary sign column — its training regime.

    infer_fn contract: takes an HxWx3 crop, returns list[Detection] with
    bboxes in CROP pixel coordinates (any internal resize is its business —
    see make_onnx_infer_fn for the reference implementation).

    Inner-pass rules:
      - inner `cartouche` detections are dropped (breaks recursion,
        fragile breakpoint #3 — bracket leakage shows up as this class)
      - an inner det overlapping an existing det (IoU > dedupe_iou)
        updates that det in place if it scores higher; never appended twice
      - surviving inner dets are tagged inside_cartouche / from_reentry

    Returns the (extended) detection list. Re-entry results never trigger
    another re-entry.
    """
    img_h, img_w = image.shape[:2]

    member_count: dict[int, int] = {ci: 0 for ci in cartouche_idxs}
    for det in detections:
        if det.inside_cartouche and det.cartouche_id in member_count:
            member_count[det.cartouche_id] += 1

    for ci in cartouche_idxs:
        if not always and member_count[ci] >= min_members:
            continue

        for frac_short, frac_long in insets:
            cx1, cy1, cx2, cy2 = inset_bbox(
                detections[ci].bbox, frac_short, frac_long
            )
            x1 = max(0, int(round(cx1)));  y1 = max(0, int(round(cy1)))
            x2 = min(img_w, int(round(cx2)));  y2 = min(img_h, int(round(cy2)))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue                  # degenerate crop, skip

            crop = image[y1:y2, x1:x2]
            inner = infer_fn(crop)

            for det in inner:
                if det.cls == CARTOUCHE_CLASS:
                    continue              # recursion / bracket leakage
                gx1, gy1, gx2, gy2 = det.bbox
                gbox = (gx1 + x1, gy1 + y1, gx2 + x1, gy2 + y1)

                # Dedupe against every det so far (including earlier
                # passes' output): update in place if better
                dup = None
                for existing in detections:
                    if iou(gbox, existing.bbox) > dedupe_iou:
                        dup = existing
                        break
                if dup is not None:
                    if det.max_score > dup.max_score:
                        dup.bbox = gbox
                        dup.top3 = det.top3
                    dup.inside_cartouche = True
                    dup.cartouche_id = ci
                    continue

                detections.append(Detection(
                    bbox             = gbox,
                    top3             = det.top3,
                    inside_cartouche = True,
                    cartouche_id     = ci,
                    from_reentry     = True,
                ))

    return detections


# ---------------------------------------------------------------------------
# Step 3 — DSU quadrat clustering
# ---------------------------------------------------------------------------

class DSU:
    """Union-Find with path compression + union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def overlap_ratio(a: tuple, b: tuple) -> float:
    """Intersection / min(area). Better than IoU for the double-box case:
    a thin box fully inside a taller box scores ~1.0 here but low IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(min(area_a, area_b), 1e-9)


def merge_duplicate_boxes(
    detections     : list[Detection],
    overlap_thresh : float = DUP_OVERLAP,
) -> list[Detection]:
    """
    Collapse multiple detections of the SAME physical sign into one slot
    (fragile breakpoint #5). Survives NMS because the duplicate boxes have
    IoU < 0.5 (e.g. a thin f31 box inside a taller s29 box on one stroke).

    Two detections merge iff intersection/min-area > overlap_thresh, both
    are non-cartouche, and they share the same cartouche context. DSU
    handles transitivity. The highest-scoring member keeps its bbox and
    identity; top-3s of the group are unioned (max score per code, top 3).

    `cartouche_id` indices are remapped to the returned list's positions
    (cartouche detections never merge, so they always survive).
    """
    n = len(detections)
    dsu = DSU(n)
    for i in range(n):
        di = detections[i]
        if di.cls == CARTOUCHE_CLASS:
            continue
        for j in range(i + 1, n):
            dj = detections[j]
            if dj.cls == CARTOUCHE_CLASS:
                continue
            if (di.inside_cartouche, di.cartouche_id) != \
               (dj.inside_cartouche, dj.cartouche_id):
                continue
            if overlap_ratio(di.bbox, dj.bbox) > overlap_thresh:
                dsu.union(i, j)

    merged: list[Detection] = []
    seen_root: set[int] = set()
    for i in range(n):                      # preserve original order
        root = dsu.find(i)
        if root in seen_root:
            continue
        seen_root.add(root)
        group = [detections[k] for k in range(n) if dsu.find(k) == root]
        base = max(group, key=lambda d: d.max_score)
        if len(group) > 1:
            scores: dict[str, float] = {}
            for d in group:
                for c, s in d.top3:
                    scores[c] = max(scores.get(c, 0.0), s)
            base.top3 = sorted(scores.items(),
                               key=lambda kv: kv[1], reverse=True)[:3]
        merged.append(base)

    # Remap cartouche_id (old index -> new index of the same object)
    new_idx = {id(obj): k for k, obj in enumerate(merged)}
    for d in merged:
        if d.cartouche_id is not None:
            d.cartouche_id = new_idx[id(detections[d.cartouche_id])]
    return merged


@dataclass
class Quadrat:
    """One visual block of 1-4 signs sharing a slot in the reading order."""
    members: list[Detection]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (min(d.bbox[0] for d in self.members),
                min(d.bbox[1] for d in self.members),
                max(d.bbox[2] for d in self.members),
                max(d.bbox[3] for d in self.members))

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def ordered(self, direction: str = 'rtl') -> list[Detection]:
        """
        Within-quadrat reading order: top-to-bottom bands, then ltr/rtl
        inside each band. Band break = y-centroid jump > 0.5 x median
        member height.
        """
        if len(self.members) <= 1:
            return list(self.members)
        med_h = float(np.median([d.height for d in self.members]))
        by_y = sorted(self.members, key=lambda d: d.centroid[1])
        bands: list[list[Detection]] = [[by_y[0]]]
        for d in by_y[1:]:
            band_y = np.mean([m.centroid[1] for m in bands[-1]])
            if d.centroid[1] - band_y > 0.5 * med_h:
                bands.append([d])
            else:
                bands[-1].append(d)
        out: list[Detection] = []
        for band in bands:
            band.sort(key=lambda d: d.centroid[0], reverse=(direction == 'rtl'))
            out.extend(band)
        return out


def _axis_overlap(a: tuple, b: tuple, axis: int) -> float:
    """Projection-overlap ratio of two bboxes on x (axis=0) or y (axis=1),
    normalized by the smaller extent."""
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    inter = min(a[hi], b[hi]) - max(a[lo], b[lo])
    if inter <= 0:
        return 0.0
    return inter / max(min(a[hi] - a[lo], b[hi] - b[lo]), 1e-9)


def _split_component(
    members   : list[Detection],
    stack_axis: int,                  # 0 = x (columns layout), 1 = y (rows)
    max_signs : int,
) -> list[list[Detection]]:
    """Recursively split an oversized component at its largest gap along
    the stack axis."""
    if len(members) <= max_signs:
        return [members]
    members = sorted(members, key=lambda d: d.centroid[stack_axis])
    gaps = [members[k + 1].centroid[stack_axis] - members[k].centroid[stack_axis]
            for k in range(len(members) - 1)]
    cut = int(np.argmax(gaps)) + 1
    return (_split_component(members[:cut], stack_axis, max_signs)
            + _split_component(members[cut:], stack_axis, max_signs))


def cluster_quadrats(
    detections    : list[Detection],
    layout        : str   = 'columns',          # 'rows' | 'columns'
    align_overlap : float = QUADRAT_ALIGN,
    gap_frac      : float = QUADRAT_GAP_FRAC,
    size_ratio    : tuple = QUADRAT_SIZE_RATIO,
    max_signs     : int   = QUADRAT_MAX_SIGNS,
) -> list[Quadrat]:
    """
    Group detections into quadrats via DSU connected components.

    ANTI-CHAINING RULE: two signs share a quadrat only if they stack
    PERPENDICULAR to the reading axis —
      layout='rows'    (horizontal reading): vertically stacked signs
                       (x-projection overlap > align_overlap, y-gap small)
      layout='columns' (vertical reading):   side-by-side signs
                       (y-projection overlap > align_overlap, x-gap small)
    Signs adjacent ALONG the reading axis never merge, so a crowded row
    can't chain into one giant component (the failure mode of the naive
    centroid-distance rule).

    Additional guards: sqrt-area ratio within `size_ratio`; components
    larger than `max_signs` split at their largest perpendicular gap.
    Cartouche-class detections never merge (each is its own quadrat).
    Caller chooses the subset: outer text = not inside_cartouche;
    cartouche interiors = per-cartouche member lists.

    Returns quadrats sorted by centroid (y, then x) for determinism;
    line-level reading order is step 4's job.
    """
    if not detections:
        return []
    # axis along which quadrat-mates align = reading axis
    read_axis  = 0 if layout == 'rows' else 1     # x for rows, y for columns
    stack_axis = 1 - read_axis

    sign_dets = [d for d in detections if d.cls != CARTOUCHE_CLASS]
    med_stack = (float(np.median([(d.width if stack_axis == 0 else d.height)
                                  for d in sign_dets]))
                 if sign_dets else 1.0)

    n = len(detections)
    dsu = DSU(n)
    for i in range(n):
        di = detections[i]
        if di.cls == CARTOUCHE_CLASS:
            continue
        for j in range(i + 1, n):
            dj = detections[j]
            if dj.cls == CARTOUCHE_CLASS:
                continue
            if _axis_overlap(di.bbox, dj.bbox, read_axis) < align_overlap:
                continue
            lo, hi = (0, 2) if stack_axis == 0 else (1, 3)
            gap = max(di.bbox[lo], dj.bbox[lo]) - min(di.bbox[hi], dj.bbox[hi])
            if gap > gap_frac * med_stack:
                continue
            # Cross-line guard: a merged pair spanning more than
            # QUADRAT_MAX_EXTENT sign-sizes on the stack axis has leaked
            # into the neighbouring row/column, even if the gap is tiny
            # (adjacent rows can sit closer than intra-quadrat stacks).
            extent = max(di.bbox[hi], dj.bbox[hi]) - min(di.bbox[lo], dj.bbox[lo])
            if extent > QUADRAT_MAX_EXTENT * med_stack:
                continue
            r = (di.area / max(dj.area, 1e-9)) ** 0.5
            if not (size_ratio[0] <= r <= size_ratio[1]):
                continue
            dsu.union(i, j)

    quadrats: list[Quadrat] = []
    for idxs in dsu.groups().values():
        members = [detections[k] for k in idxs]
        for part in _split_component(members, stack_axis, max_signs):
            quadrats.append(Quadrat(members=part))
    quadrats.sort(key=lambda q: (q.centroid[1], q.centroid[0]))
    return quadrats


# ---------------------------------------------------------------------------
# Step 4 — line assembly: quadrats -> reading order + boundary hints
# ---------------------------------------------------------------------------

@dataclass
class ReadingOrder:
    """Final spatial-layer output, ready for sphinx_corrector.correct()."""
    slots           : list[list[tuple[str, float]]]   # top-3 per slot
    boundary_hints  : list[int]          # slot indices where a line ends
                                         # (exclusive end — matches the j
                                         # convention in viterbi_segment)
    slot_detections : list[Optional[Detection]]   # None = synthetic Unknown
    lines           : list[list[Quadrat]]

    @property
    def n_synthetic(self) -> int:
        return sum(1 for d in self.slot_detections if d is None)


def _group_lines(
    quadrats : list[Quadrat],
    cross    : int,                      # cross axis: 0=x (columns), 1=y (rows)
    gap_frac : float,
) -> list[list[Quadrat]]:
    """1-D cluster quadrats on the cross axis using a RUNNING-MEAN center
    (comparing to the last element drifts on slanted photos — the bug in
    the old order_signs.py)."""
    med = float(np.median([(q.bbox[2] - q.bbox[0]) if cross == 0
                           else (q.bbox[3] - q.bbox[1]) for q in quadrats]))
    qs = sorted(quadrats, key=lambda q: q.centroid[cross])
    lines: list[list[Quadrat]] = [[qs[0]]]
    for q in qs[1:]:
        mean_c = float(np.mean([m.centroid[cross] for m in lines[-1]]))
        if abs(q.centroid[cross] - mean_c) > gap_frac * med:
            lines.append([q])
        else:
            lines[-1].append(q)
    return lines


def assemble_reading_order(
    quadrats         : list[Quadrat],
    layout           : str   = 'columns',     # 'rows' | 'columns'
    direction        : str   = 'rtl',         # 'ltr' | 'rtl'
    line_gap_frac    : float = LINE_GAP_FRAC,
    missing_gap_frac : float = MISSING_GAP_FRAC,
    max_inserts      : int   = MAX_GAP_INSERTS,
    extent           : Optional[tuple] = None,
    single_line      : bool  = False,
) -> ReadingOrder:
    """
    Assemble quadrats into final reading order.

    1. Group quadrats into lines on the cross axis (columns: x; rows: y).
    2. Order lines: columns follow `direction` (rtl = rightmost column
       first); rows always read top-down.
    3. Walk each line along the reading axis (columns: top-down; rows:
       per `direction`), emitting each quadrat's members via
       Quadrat.ordered(direction) — one slot (top-3) per sign.
    4. SYNTHETIC UNKNOWN INSERTION (fragile breakpoint #1): when the
       edge-gap between consecutive quadrats in a line exceeds
       `missing_gap_frac` x median quadrat step, YOLO probably dropped
       sign(s) there — insert round(gap/step) Unknown slots (capped at
       `max_inserts`) so the Unknown-resolver / royal-name matcher can
       fill them. With `extent` (e.g. the inset cartouche bbox), leading
       and trailing gaps are checked too — a missing FIRST sign (the
       Unas e34 case) is only detectable against a known extent.
    5. After each line, append len(slots) to boundary_hints (exclusive
       end index — the corrector's LAYOUT_BONUS convention).

    `single_line=True` skips line grouping and treats every quadrat as
    one line. REQUIRED for cartouche interiors: narrow signs are not
    x-aligned, so line grouping splits the interior into fake columns and
    the per-line extent checks then flood it with spurious Unknowns.

    Returns a ReadingOrder. Slots use YOLO-space class names; the glue
    layer normalizes via cartouche_matcher.normalize_code.
    """
    if not quadrats:
        return ReadingOrder([], [], [], [])

    read_axis = 0 if layout == 'rows' else 1
    cross     = 1 - read_axis
    lo, hi    = (0, 2) if read_axis == 0 else (1, 3)

    med_step = float(np.median(
        [q.bbox[hi] - q.bbox[lo] for q in quadrats]))

    lines = ([list(quadrats)] if single_line
             else _group_lines(quadrats, cross, line_gap_frac))
    # Line order: columns follow direction; rows always top-down
    if layout == 'columns' and direction == 'rtl':
        lines.sort(key=lambda ln: -float(np.mean([q.centroid[0] for q in ln])))
    elif layout == 'columns':
        lines.sort(key=lambda ln: float(np.mean([q.centroid[0] for q in ln])))
    else:
        lines.sort(key=lambda ln: float(np.mean([q.centroid[1] for q in ln])))

    descending = (layout == 'rows' and direction == 'rtl')

    slots           : list[list[tuple[str, float]]] = []
    slot_detections : list[Optional[Detection]]     = []
    boundary_hints  : list[int]                     = []

    def emit_unknowns(gap: float) -> None:
        if gap <= missing_gap_frac * med_step:
            return
        # one missing sign ≈ one med_step of empty space
        n = min(max_inserts, max(1, int(gap // med_step)))
        for _ in range(n):
            slots.append([(UNKNOWN_CLASS, 0.0)])
            slot_detections.append(None)

    for line in lines:
        line.sort(key=lambda q: q.centroid[read_axis], reverse=descending)

        for k, q in enumerate(line):
            if k == 0:
                if extent is not None:
                    lead = (extent[hi] - q.bbox[hi] if descending
                            else q.bbox[lo] - extent[lo])
                    emit_unknowns(lead)
            else:
                prev = line[k - 1]
                gap = (prev.bbox[lo] - q.bbox[hi] if descending
                       else q.bbox[lo] - prev.bbox[hi])
                emit_unknowns(gap)
            for d in q.ordered(direction):
                slots.append(list(d.top3))
                slot_detections.append(d)

        if extent is not None and line:
            last = line[-1]
            trail = (last.bbox[lo] - extent[lo] if descending
                     else extent[hi] - last.bbox[hi])
            emit_unknowns(trail)

        boundary_hints.append(len(slots))

    return ReadingOrder(
        slots           = slots,
        boundary_hints  = boundary_hints,
        slot_detections = slot_detections,
        lines           = lines,
    )


# ---------------------------------------------------------------------------
# Reference infer_fn for production / re-entry (ONNXRuntime + letterbox)
# ---------------------------------------------------------------------------

def letterbox(
    bgr: np.ndarray, imgsz: int = 1024, pad_value: int = 114,
) -> tuple[np.ndarray, float, int, int]:
    """
    Aspect-preserving resize onto a square canvas (Ultralytics-style).
    Returns (canvas, scale, dx, dy) where original = (model - d) / scale.

    NOTE: for THIS model, plain stretch outperforms letterbox on re-entry
    crops (6/6 vs 3-5/6 cartouches covered on test_image_vn5). The training
    data went through ETL/batch_resize.py's in-place 224x224 squash, so the
    model learned aspect-distorted signs — stretch matches that
    distribution. Letterbox is kept for experiments and future models
    trained on aspect-preserved data.
    """
    import cv2
    h, w = bgr.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(bgr, (nw, nh))
    canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    dx, dy = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def make_onnx_infer_fn(
    session,                          # onnxruntime.InferenceSession
    class_names : list[str],
    conf_thresh : float = REENTRY_CONF,
    iou_thresh  : float = NMS_IOU,
    imgsz       : int   = 1024,
    mode        : str   = 'letterbox',  # 'letterbox' | 'stretch'
) -> Callable[[np.ndarray], list[Detection]]:
    """
    Build an infer_fn satisfying the cartouche_reentry contract: BGR crop
    in -> list[Detection] with bboxes in crop pixel coordinates out.
    Also usable for the global pass (pass conf_thresh=CONF_THRESHOLD).

    mode='letterbox' matches the V4 training distribution (Ultralytics
    trained at 1024 with letterbox) and reproduces Colab raw-YOLO output
    exactly (A/B on grand_glyphs.jpeg: 0 missing / 0 extra vs 5 extra
    for stretch). Use it for the global pass. mode='stretch' kept for
    legacy callers; the old 224x224-squash rationale is obsolete.
    """
    import cv2
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    def _run(canvas: np.ndarray) -> list[Detection]:
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        raw = session.run([out_name], {inp_name: x})[0]
        return postprocess_onnx(raw, class_names, conf_thresh, iou_thresh)

    def infer_stretch(bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        dets = _run(cv2.resize(bgr, (imgsz, imgsz)))
        sx, sy = w / imgsz, h / imgsz
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            d.bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        return dets

    def infer_letterbox(bgr: np.ndarray) -> list[Detection]:
        canvas, scale, dx, dy = letterbox(bgr, imgsz)
        dets = _run(canvas)
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            d.bbox = ((x1 - dx) / scale, (y1 - dy) / scale,
                      (x2 - dx) / scale, (y2 - dy) / scale)
        return dets

    if mode == 'stretch':
        return infer_stretch
    if mode == 'letterbox':
        return infer_letterbox
    raise ValueError(f"mode must be 'stretch' or 'letterbox', got {mode!r}")


# ---------------------------------------------------------------------------
# CLI smoke test (synthetic — no ONNX model or image files needed)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("spatial_logic.py — smoke test")
    print("=" * 60)

    names = ['Aa1', 'cartouche', 'g17', 'n35', 'd21', 'unknown']

    # ------------------------------------------------------------------
    print("Test 1 — Detection dataclass properties")
    d = Detection(bbox=(10.0, 20.0, 50.0, 100.0),
                  top3=[('g17', 0.9), ('n35', 0.05), ('unknown', 0.01)])
    assert d.centroid == (30.0, 60.0)
    assert d.width == 40.0 and d.height == 80.0 and d.area == 3200.0
    assert d.cls == 'g17' and d.max_score == 0.9
    assert not d.is_cartouche()
    print("  centroid/area/cls/max_score                    OK")

    # ------------------------------------------------------------------
    print("Test 2 — postprocess_onnx: conf filter, NMS, top-3")
    C, N = len(names), 6
    raw = np.zeros((4 + C, N), dtype=np.float32)
    raw[4:, :] = 0.01                                  # background noise

    # anchor 0: g17 at (100,100) 40x40, score 0.90
    raw[:4, 0] = [100, 100, 40, 40];  raw[4 + 2, 0] = 0.90
    raw[4 + 3, 0] = 0.30; raw[4 + 5, 0] = 0.10         # alternates
    # anchor 1: duplicate of anchor 0, slightly shifted, lower score
    raw[:4, 1] = [104, 102, 40, 40];  raw[4 + 2, 1] = 0.70
    # anchor 2: n35 far away, score 0.60
    raw[:4, 2] = [300, 100, 30, 30];  raw[4 + 3, 2] = 0.60
    # anchor 3: below conf threshold everywhere
    raw[:4, 3] = [500, 100, 30, 30];  raw[4 + 0, 3] = 0.10
    # anchors 4-5: zeros (dead)

    dets = postprocess_onnx(raw[None], names, conf_thresh=0.25)
    assert len(dets) == 2, f"expected 2 detections, got {len(dets)}"
    assert dets[0].cls == 'g17' and abs(dets[0].max_score - 0.90) < 1e-6
    assert dets[1].cls == 'n35'
    # top-3 ordering on the survivor
    assert [c for c, _ in dets[0].top3] == ['g17', 'n35', 'unknown']
    # bbox decoded cxcywh -> xyxy
    assert dets[0].bbox == (80.0, 80.0, 120.0, 120.0)
    print(f"  kept {len(dets)} dets (dup suppressed, low-conf dropped)  OK")
    print(f"  top3 of first: {[(c, round(s, 2)) for c, s in dets[0].top3]}")

    # ------------------------------------------------------------------
    print("Test 3 — cartouche containment tagging")
    cart = Detection(bbox=(100.0, 50.0, 300.0, 120.0),
                     top3=[('cartouche', 0.80), ('Aa1', 0.05), ('g17', 0.02)])
    inside    = Detection(bbox=(120.0, 60.0, 150.0, 110.0),
                          top3=[('g17', 0.85), ('n35', 0.1), ('d21', 0.02)])
    # centroid x=302 — just past the raw edge, caught by the 10% expansion
    bracket   = Detection(bbox=(294.0, 60.0, 310.0, 110.0),
                          top3=[('n35', 0.70), ('g17', 0.1), ('d21', 0.05)])
    outside   = Detection(bbox=(400.0, 60.0, 430.0, 110.0),
                          top3=[('d21', 0.90), ('g17', 0.05), ('n35', 0.02)])
    dets3 = [cart, inside, bracket, outside]
    cart_idxs = tag_cartouche_members(dets3)
    assert cart_idxs == [0]
    assert inside.inside_cartouche  and inside.cartouche_id  == 0
    assert bracket.inside_cartouche and bracket.cartouche_id == 0
    assert not outside.inside_cartouche
    assert not cart.inside_cartouche       # cartouches never tag themselves
    print("  inside tagged, bracket-edge caught by expansion, "
          "outside untouched  OK")

    # ------------------------------------------------------------------
    print("Test 4 — re-entry fallback fires on empty cartouche")
    img = np.zeros((200, 500, 3), dtype=np.uint8)
    cart4 = Detection(bbox=(100.0, 50.0, 300.0, 150.0),
                      top3=[('cartouche', 0.9), ('Aa1', 0.02), ('g17', 0.01)])
    lone  = Detection(bbox=(130.0, 70.0, 160.0, 130.0),
                      top3=[('g17', 0.55), ('n35', 0.2), ('d21', 0.1)])
    dets4 = [cart4, lone]
    idxs4 = tag_cartouche_members(dets4)
    assert lone.inside_cartouche          # 1 member < MIN_CARTOUCHE_MEMBERS

    calls = []
    def mock_infer(crop):
        calls.append(crop.shape)
        return [
            # bracket leakage: must be filtered
            Detection(bbox=(0.0, 0.0, 160.0, 88.0),
                      top3=[('cartouche', 0.6), ('Aa1', 0.1), ('g17', 0.05)]),
            # duplicate of `lone`, higher score: updates it in place
            Detection(bbox=(9.0, 13.0, 41.0, 75.0),
                      top3=[('g17', 0.80), ('n35', 0.1), ('d21', 0.05)]),
            # genuinely new inner sign
            Detection(bbox=(82.0, 15.0, 112.0, 75.0),
                      top3=[('n35', 0.75), ('g17', 0.1), ('d21', 0.05)]),
        ]

    out4 = cartouche_reentry(img, dets4, idxs4, mock_infer)
    # two inset passes per cartouche: (6%,6%) then (6%,10%)
    # horizontal cartouche (100,50,300,150): long axis = x
    #   pass 1 origin (112,56), 176x88;  pass 2 origin (120,56), 160x88
    assert len(calls) == 2, f"expected 2 passes, got {len(calls)}"
    assert calls[0] == (88, 176, 3), f"pass-1 crop shape {calls[0]}"
    assert calls[1] == (88, 160, 3), f"pass-2 crop shape {calls[1]}"
    # cross-pass union dedupes: still exactly 3 dets, no twins
    assert len(out4) == 3, f"expected 3 dets after re-entry, got {len(out4)}"
    # `lone` updated in place by the higher-scoring pass-1 duplicate;
    # pass-2's shifted twin (IoU 0.6, equal score) must NOT update again
    assert abs(lone.max_score - 0.80) < 1e-6
    assert lone.bbox == (121.0, 69.0, 153.0, 131.0)
    # new det mapped to global coords via pass-1 origin and tagged
    new = out4[2]
    assert new.from_reentry and new.inside_cartouche and new.cartouche_id == 0
    assert new.bbox == (194.0, 71.0, 224.0, 131.0)
    assert all(d.cls != CARTOUCHE_CLASS or d is cart4 for d in out4), \
        "inner cartouche must be filtered"
    print("  2-pass crops, bracket filter, cross-pass dedupe, "
          "global mapping  OK")

    # ------------------------------------------------------------------
    print("Test 5 — legacy mode (always=False) skips populated cartouches")
    cart5 = Detection(bbox=(100.0, 50.0, 300.0, 150.0),
                      top3=[('cartouche', 0.9), ('Aa1', 0.02), ('g17', 0.01)])
    m1 = Detection(bbox=(120.0, 70.0, 150.0, 130.0),
                   top3=[('g17', 0.8), ('n35', 0.1), ('d21', 0.05)])
    m2 = Detection(bbox=(180.0, 70.0, 210.0, 130.0),
                   top3=[('n35', 0.7), ('g17', 0.1), ('d21', 0.05)])
    dets5 = [cart5, m1, m2]
    idxs5 = tag_cartouche_members(dets5)

    def must_not_run(crop):
        raise AssertionError("legacy mode must not fire: 2 members tagged")

    out5 = cartouche_reentry(img, dets5, idxs5, must_not_run, always=False)
    assert len(out5) == 3
    # default always=True DOES fire on the same input
    fired = []
    dets5b = [cart5, m1, m2]
    cartouche_reentry(img, dets5b, idxs5, lambda c: fired.append(1) or [])
    assert fired, "always=True must re-enter even populated cartouches"
    print("  legacy skip + always-mode fire                 OK")

    # ------------------------------------------------------------------
    print("Test 6 — asymmetric inset on a vertical cartouche")
    # vertical box 60x200: long axis = y -> x inset 6%, y inset 10%
    box = inset_bbox((100.0, 50.0, 160.0, 250.0), 0.06, 0.10)
    assert box == (103.6, 70.0, 156.4, 230.0), f"got {box}"
    # letterbox round-trip: tall crop maps back exactly
    crop6 = np.zeros((300, 100, 3), dtype=np.uint8)
    canvas, scale, dx, dy = letterbox(crop6, imgsz=1024)
    assert canvas.shape == (1024, 1024, 3)
    # a point at crop (50, 150) -> model (50*s+dx, 150*s+dy) -> back
    mx, my = 50 * scale + dx, 150 * scale + dy
    assert abs((mx - dx) / scale - 50) < 1e-9
    assert abs((my - dy) / scale - 150) < 1e-9
    print("  long-axis inset + letterbox round-trip         OK")

    # ------------------------------------------------------------------
    print("Test 7 — duplicate-box merge (DSU, top-3 union, id remap)")
    cart7 = Detection(bbox=(80.0, 80.0, 180.0, 420.0),
                      top3=[('cartouche', 0.9), ('Aa1', 0.02), ('g17', 0.01)])
    s29 = Detection(bbox=(100.0, 100.0, 130.0, 200.0),
                    top3=[('s29', 0.79), ('o34', 0.05), ('f31', 0.04)],
                    inside_cartouche=True, cartouche_id=0, from_reentry=True)
    f31 = Detection(bbox=(105.0, 102.0, 122.0, 198.0),     # thin twin inside
                    top3=[('f31', 0.50), ('m17', 0.10), ('z4', 0.03)],
                    inside_cartouche=True, cartouche_id=0, from_reentry=True)
    far = Detection(bbox=(100.0, 300.0, 130.0, 400.0),
                    top3=[('n35', 0.60), ('n37', 0.05), ('z7', 0.02)],
                    inside_cartouche=True, cartouche_id=0, from_reentry=True)
    m7 = merge_duplicate_boxes([cart7, s29, f31, far])
    assert len(m7) == 3, f"expected 3 after merge, got {len(m7)}"
    assert m7[1] is s29                       # higher score keeps identity
    assert [c for c, _ in s29.top3] == ['s29', 'f31', 'm17']   # union'd
    assert abs(dict(s29.top3)['f31'] - 0.50) < 1e-9
    # cartouche survived unmerged despite full overlap with its members
    assert m7[0] is cart7
    # cartouche_id remapped to new positions (cart7 still index 0 here)
    assert s29.cartouche_id == 0 and far.cartouche_id == 0
    print("  same-stroke twin merged, cartouche immune, ids remapped  OK")

    # ------------------------------------------------------------------
    print("Test 8 — quadrat clustering, columns layout (anti-chaining)")
    # column of: flat n35, then m17+s29 side-by-side pair, then flat x1
    n35q = Detection(bbox=(260.0, 60.0, 330.0, 85.0),
                     top3=[('n35', 0.6), ('n37', 0.05), ('z7', 0.02)])
    m17q = Detection(bbox=(300.0, 100.0, 315.0, 180.0),
                     top3=[('m17', 0.65), ('f31', 0.1), ('z4', 0.03)])
    s29q = Detection(bbox=(265.0, 100.0, 290.0, 180.0),
                     top3=[('s29', 0.79), ('o34', 0.05), ('f31', 0.04)])
    x1q  = Detection(bbox=(265.0, 200.0, 325.0, 225.0),
                     top3=[('x1', 0.7), ('z1', 0.05), ('x8', 0.02)])
    qs = cluster_quadrats([n35q, m17q, s29q, x1q], layout='columns')
    sizes = sorted(len(q.members) for q in qs)
    assert sizes == [1, 1, 2], f"expected [1,1,2], got {sizes}"
    pair = next(q for q in qs if len(q.members) == 2)
    assert {d.cls for d in pair.members} == {'m17', 's29'}
    # within-quadrat rtl: m17 (right) reads before s29 (left)
    assert [d.cls for d in pair.ordered('rtl')] == ['m17', 's29']
    assert [d.cls for d in pair.ordered('ltr')] == ['s29', 'm17']
    # vertical neighbors in the column never merged (no y-overlap)
    print("  side-by-side pair grouped, column flow not chained, "
          "rtl order  OK")

    # ------------------------------------------------------------------
    print("Test 9 — quadrat clustering, rows layout (anti-chaining)")
    # a row of 3 adjacent signs + one stacked pair
    a = Detection(bbox=(100.0, 100.0, 140.0, 140.0),
                  top3=[('g17', 0.8), ('g18', 0.05), ('Aa1', 0.02)])
    b = Detection(bbox=(150.0, 100.0, 190.0, 140.0),
                  top3=[('d21', 0.8), ('d22', 0.05), ('Aa1', 0.02)])
    c = Detection(bbox=(200.0, 100.0, 240.0, 140.0),
                  top3=[('x1', 0.8), ('z1', 0.05), ('Aa1', 0.02)])
    top = Detection(bbox=(250.0, 95.0, 290.0, 115.0),
                    top3=[('n35', 0.7), ('n37', 0.05), ('z7', 0.02)])
    bot = Detection(bbox=(250.0, 122.0, 290.0, 142.0),
                    top3=[('z1', 0.6), ('x1', 0.05), ('z4', 0.02)])
    qs = cluster_quadrats([a, b, c, top, bot], layout='rows')
    sizes = sorted(len(q.members) for q in qs)
    assert sizes == [1, 1, 1, 2], f"expected [1,1,1,2], got {sizes}"
    pair = next(q for q in qs if len(q.members) == 2)
    assert [d.cls for d in pair.ordered('rtl')] == ['n35', 'z1']  # top first
    print("  stacked pair grouped, crowded row not chained          OK")

    # ------------------------------------------------------------------
    print("Test 10 — oversized component splits at largest gap")
    run = []
    for k, x in enumerate((0, 20, 40, 75, 95)):     # largest gap 40->75
        run.append(Detection(bbox=(float(x), 100.0, float(x + 14), 180.0),
                             top3=[('m17', 0.6), ('z4', 0.05), ('f31', 0.02)]))
    qs = cluster_quadrats(run, layout='columns', max_signs=4)
    sizes = sorted(len(q.members) for q in qs)
    assert sizes == [2, 3], f"expected split [2,3], got {sizes}"
    print("  5-sign chain split into 3+2 at the widest gap          OK")

    # ------------------------------------------------------------------
    print("Test 11 — line assembly: columns rtl + boundary hints")
    def det(x1, y1, x2, y2, cls_, score=0.8):
        return Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                         top3=[(cls_, score), ('z1', 0.05), ('Aa1', 0.02)])
    # right column: g17, n35 (top-down). left column: d21, x1.
    rg, rn = det(300, 100, 340, 140, 'g17'), det(300, 160, 340, 200, 'n35')
    ld, lx = det(100, 100, 140, 140, 'd21'), det(100, 160, 140, 200, 'x1')
    quads = cluster_quadrats([rg, rn, ld, lx], layout='columns')
    ro = assemble_reading_order(quads, layout='columns', direction='rtl')
    seq = [s[0][0] for s in ro.slots]
    assert seq == ['g17', 'n35', 'd21', 'x1'], seq      # right column first
    assert ro.boundary_hints == [2, 4], ro.boundary_hints
    assert ro.n_synthetic == 0
    # ltr flips column order
    ro2 = assemble_reading_order(quads, layout='columns', direction='ltr')
    assert [s[0][0] for s in ro2.slots] == ['d21', 'x1', 'g17', 'n35']
    print("  column order, boundary hints, ltr flip                 OK")

    # ------------------------------------------------------------------
    print("Test 12 — synthetic Unknown insertion (mid-gap + leading edge)")
    # one column; ~1.7-step gap between n35 and s29 (one sign missing)
    t = det(100, 100, 140, 130, 'n35')             # h=30 -> med_step=30
    btm = det(100, 180, 140, 210, 's29')           # gap=50 > 1.5*30
    quads = cluster_quadrats([t, btm], layout='columns')
    ro = assemble_reading_order(quads, layout='columns', direction='rtl')
    seq = [s[0][0] for s in ro.slots]
    assert seq == ['n35', 'unknown', 's29'], seq
    assert ro.slot_detections[1] is None
    assert ro.slots[1] == [('unknown', 0.0)]
    # with extent: leading gap above the first sign also flagged
    # (the Unas e34 case — missing FIRST sign, no quadrat above it)
    ro = assemble_reading_order(quads, layout='columns', direction='rtl',
                                extent=(95.0, 50.0, 145.0, 240.0))
    seq = [s[0][0] for s in ro.slots]
    assert seq == ['unknown', 'n35', 'unknown', 's29'], seq
    # huge gap (~6 steps) inserts only MAX_GAP_INSERTS synthetic slots
    big = det(100, 330, 140, 360, 'x1')
    quads = cluster_quadrats([t, big], layout='columns')
    ro = assemble_reading_order(quads, layout='columns', direction='rtl')
    seq = [s[0][0] for s in ro.slots]
    assert seq == ['n35', 'unknown', 'unknown', 'x1'], seq
    print("  mid-gap, leading-edge (extent), multi-insert cap       OK")

    # ------------------------------------------------------------------
    print("Test 13 — rows layout, rtl reads right-to-left")
    r1a, r1b = det(300, 100, 340, 140, 'g17'), det(250, 100, 290, 140, 'd21')
    r2a = det(300, 200, 340, 240, 'x1')
    quads = cluster_quadrats([r1a, r1b, r2a], layout='rows')
    ro = assemble_reading_order(quads, layout='rows', direction='rtl')
    seq = [s[0][0] for s in ro.slots]
    assert seq == ['g17', 'd21', 'x1'], seq     # row 1 right->left, then row 2
    assert ro.boundary_hints == [2, 3]
    # rtl rows: ~1.75-step gap to the LEFT of the first sign
    quads = cluster_quadrats([r1a, det(190, 100, 230, 140, 'd21')],
                             layout='rows')
    ro = assemble_reading_order(quads, layout='rows', direction='rtl')
    seq = [s[0][0] for s in ro.slots]
    assert seq == ['g17', 'unknown', 'd21'], seq
    print("  rtl row order, rtl mid-gap insertion                   OK")

    # ------------------------------------------------------------------
    print("Test 14 — single_line: cartouche interiors are ONE column")
    # x-misaligned signs that line-grouping wrongly splits into 2 columns
    n_  = det(140, 100, 160, 120, 'n35')            # narrow, sits right
    m_  = det(100, 140, 112, 170, 'm17')            # tall pair, sits left
    s_  = det(118, 140, 143, 170, 's29')
    quads = cluster_quadrats([n_, m_, s_], layout='columns')
    ext = (95.0, 60.0, 160.0, 200.0)
    multi = assemble_reading_order(quads, layout='columns', direction='rtl',
                                   extent=ext)
    single = assemble_reading_order(quads, layout='columns', direction='rtl',
                                    extent=ext, single_line=True)
    assert len(multi.lines) == 2          # the failure mode: fake columns
    assert len(single.lines) == 1
    seq = [s[0][0] for s in single.slots]
    # one leading Unknown (gap above n35), then top-down, rtl in the band
    assert seq == ['unknown', 'n35', 's29', 'm17'], seq
    print("  one line forced (multi-mode split into 2), rtl bands   OK")

    print()
    print("All tests passed.")
