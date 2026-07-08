#!/usr/bin/env python3
"""
Phase 1 — verify V3 best.onnx.

Three checks:
  1.1  output shape is [1, 154, ~21504]
  1.2  per-anchor scores are independent sigmoids (multi-label, do NOT sum to 1)
  1.3  150 class channels line up with the master class_map50_v2.json order
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ONNX_PATH  = Path('best.onnx')
IMG_PATH   = Path('test_image_vn5.png')
CLASS_JSON = Path('class_map50_v2.json')
IMGSZ      = 1024


def letterbox(im, new_shape=IMGSZ, color=(114, 114, 114)):
    h0, w0 = im.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape - new_w, new_shape - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def preprocess(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        sys.exit(f'cannot read {path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im, r, pad = letterbox(rgb, IMGSZ)
    x = im.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]   # 1,3,H,W
    return np.ascontiguousarray(x), r, pad, bgr.shape[:2]


def main():
    print(f'Loading {ONNX_PATH} ...')
    sess = ort.InferenceSession(str(ONNX_PATH),
                                providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f'  input : {inp.name} {inp.shape} {inp.type}')
    print(f'  output: {out.name} {out.shape} {out.type}')

    classes = list(json.load(open(CLASS_JSON)).keys())
    print(f'\nclass list from {CLASS_JSON}: {len(classes)} entries')
    print(f'  first 3 : {classes[:3]}')
    print(f'  last 3  : {classes[-3:]}')
    assert len(classes) == 150, f'expected 150 classes, got {len(classes)}'

    print(f'\nRunning {IMG_PATH} ...')
    x, r, pad, orig_hw = preprocess(IMG_PATH)
    print(f'  preprocessed: {x.shape}  ratio={r:.4f}  pad={pad}  orig={orig_hw}')
    y = sess.run(None, {inp.name: x})[0]
    print(f'  raw output : {y.shape}  dtype={y.dtype}')

    # === 1.1 shape check ===
    assert y.ndim == 3 and y.shape[0] == 1 and y.shape[1] == 154, \
        f'unexpected shape {y.shape}; expected [1, 154, ~21504]'
    n_anchors = y.shape[2]
    print(f'\n[1.1] OK  shape=[1, 154, {n_anchors}]  '
          f'(boxes=4, classes=150, anchors={n_anchors})')

    # === 1.2 score distribution ===
    cls_scores = y[0, 4:, :]                       # (150, n_anchors)
    print(f'\n[1.2] score distribution (all 150 × {n_anchors} = '
          f'{cls_scores.size:,} values):')
    print(f'  min  = {cls_scores.min():.6f}')
    print(f'  max  = {cls_scores.max():.6f}')
    print(f'  mean = {cls_scores.mean():.6f}')

    per_anchor_sum = cls_scores.sum(axis=0)
    print(f'\n  per-anchor sum (should NOT cluster around 1.0 if sigmoid):')
    print(f'    min = {per_anchor_sum.min():.4f}')
    print(f'    max = {per_anchor_sum.max():.4f}')
    print(f'    mean= {per_anchor_sum.mean():.4f}')

    max_per_anchor = cls_scores.max(axis=0)
    for thr in (0.25, 0.50, 0.80):
        n = int((max_per_anchor > thr).sum())
        print(f'  anchors with max_class_score > {thr:.2f}: {n}')

    # === 1.3 top-K detections, class names mapped via class_map50_v2.json ===
    print(f'\n[1.3] top-10 anchors by max_class_score (class names from '
          f'{CLASS_JSON} order):')
    top_idx = np.argsort(-max_per_anchor)[:10]
    boxes = y[0, :4, :]   # cx, cy, w, h
    for rank, ai in enumerate(top_idx):
        scores = cls_scores[:, ai]
        top3 = np.argsort(-scores)[:3]
        cx, cy, w, h = boxes[:, ai]
        names = [(classes[i], float(scores[i])) for i in top3]
        print(f'  #{rank+1:2d}  anchor={ai:>5d}  '
              f'box=({cx:6.1f},{cy:6.1f},{w:5.1f},{h:5.1f})  '
              f'top3={names}')

    print(f'\nAll Phase-1 checks done.')


if __name__ == '__main__':
    main()
