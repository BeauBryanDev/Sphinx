from __future__ import annotations
from pydantic import BaseModel


class SegmentedWordOut(BaseModel):
    codes       : list[str]
    translit    : str
    translation : str
    freq        : int
    edit_dist   : int
    confidence  : float
    source      : str


class ResolvedUnknownOut(BaseModel):
    slot     : int
    proposed : str
    reason   : str
    freq     : int


class CorrectionOut(BaseModel):
    segmented_words    : list[SegmentedWordOut]
    unknowns_resolved  : list[ResolvedUnknownOut]
    flat_corrected_seq : list[str]
    flat_translit      : str
    flat_translation   : str
    score              : float
    had_fallback       : bool
