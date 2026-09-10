
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Register = Literal[
    'unknown',        # let the model pick a neutral monumental register
    'monumental',     # formal temple/stela style
    'literary',       # Middle-Kingdom tale style
    'letter',         # epistolary
    'religious',      # hymn / offering formula style
]

# Schemas for the retro-translation feature (POST /reverse/):
# modern English -> Middle Egyptian (transliteration + Gardiner codes).

# The whole linguistic job is done by the LLM  :: the CV pipeline plays no
# part here. The response mirrors the LLM's staged output: grammar
# rendering, phonetic (Leiden) transliteration, then sign coding.

class ReverseTranslateRequest(BaseModel):
    text     : str = Field(..., min_length=1, max_length=1000,
                           description='Modern English text to render '
                                       'into Middle Egyptian.')
    register : Register = 'unknown'


class ReverseWord(BaseModel):
    """One Egyptian word of the composed text."""
    english         : str          # the English word/phrase it renders
    transliteration : str          # Leiden conventions
    gardiner_codes  : list[str]    # canonical codes, reading order
    literal         : str          # literal meaning of the Egyptian word
    note            : str = ''     # determinative / grammar remark


class ReverseTranslationOut(BaseModel):
    source_text          : str
    normalized_english   : str            # the plain rephrasing actually translated
    transliteration      : str            # full Leiden transliteration
    gardiner_codes       : list[str]      # flat sequence, reading order
    words                : list[ReverseWord] = []
    grammar_notes        : str = ''
    confidence           : Literal['HIGH', 'MEDIUM', 'LOW'] = 'LOW'
    model                : str = ''
    error                : Optional[str] = None
