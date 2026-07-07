from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class CartoucheOut(BaseModel):
    bbox          : tuple[float, float, float, float]
    n_members     : int
    inferred      : bool
    translit      : Optional[str]   = None
    english       : Optional[str]   = None
    spelling      : Optional[list[str]] = None
    score         : Optional[float] = None
    aligned_codes : Optional[list[str]] = None
    verified      : Optional[bool]  = None
