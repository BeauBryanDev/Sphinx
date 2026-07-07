from __future__ import annotations
from typing import Optional
from pydantic import BaseModel
from app.schemas.corrections import CorrectionOut
from app.schemas.cartouches import CartoucheOut
from app.schemas.transliterations import TransliterationOut


class OuterOut(BaseModel):
    slots          : list[list[tuple[str, float]]]
    boundary_hints : list
    n_synthetic    : int
    correction     : CorrectionOut


class PredictResponse(BaseModel):
    layout       : str
    direction    : str
    image_shape  : tuple[int, int]
    n_detections : int
    n_cartouches : int
    outer        : OuterOut
    cartouches   : list[CartoucheOut]
    # LLM stage (Phase 6). None when translate=false was requested;
    # populated with error field set when the LLM stage failed.
    transliteration : Optional[TransliterationOut] = None
