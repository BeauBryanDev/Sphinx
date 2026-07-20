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
    # Raw interior top-1 Gardiner codes in reading order. Present even when
    # the royal-name match REFUSED (translit=None), so the client can still
    # forward the signs to the LLM instead of dropping them silently.
    interior_codes : Optional[list[str]] = None
