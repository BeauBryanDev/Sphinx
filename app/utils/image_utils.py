from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException


MAX_DIM = 8000
MIN_DIM = 32
SEGMENTATION_ASPECT_RATIO = 3.0   # width/height or height/width above this -> likely multi-panel wall


def decode_upload(data: bytes) -> np.ndarray:
    """Raw bytes -> BGR ndarray. Raises HTTP 400 on decode failure."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. "
                            "Upload a valid JPEG, PNG, or WebP file.")
    return bgr


def validate_image(bgr: np.ndarray, max_mb: int = 20) -> None:
    """
    Raises HTTP 400 if the image is outside acceptable bounds.
    Called after decode_upload so we check pixel dimensions, not file size.
    """
    h, w = bgr.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        raise HTTPException(status_code=400,
                            detail=f"Image too small ({w}x{h}). Minimum {MIN_DIM}px on each side.")
    if h > MAX_DIM or w > MAX_DIM:
        raise HTTPException(status_code=400,
                            detail=f"Image too large ({w}x{h}). Maximum {MAX_DIM}px on each side.")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise HTTPException(status_code=400,
                            detail="Image must be a 3-channel colour image.")
    size_mb = bgr.nbytes / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(status_code=400,
                            detail=f"Decoded image is {size_mb:.1f} MB, exceeds limit of {max_mb} MB.")


def needs_segmentation(bgr: np.ndarray) -> bool:
    """
    Heuristic: a panoramic wall image with aspect ratio > 3:1 likely contains
    multiple panels and should be passed through wall_segmenter before inference.
    Returns True when wall_segmenter.segment() should be called first.
    """
    h, w = bgr.shape[:2]
    ratio = max(w, h) / max(min(w, h), 1)
    return ratio >= SEGMENTATION_ASPECT_RATIO
