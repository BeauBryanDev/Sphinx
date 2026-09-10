
from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from app.core.config import settings
from app.schemas.reverse_translation import (
    ReverseTranslationOut,
    ReverseWord,
)
# Retro-translation service: modern English -> Middle Egyptian.

# It is a LLM feature — the vision pipeline plays no part. The model works
# in the three staged steps of a composition exercise:
#   1. normalize the English into translatable Middle Egyptian semantics
#      (Egyptian has no word for 'internet'; it does for 'scribe');
#   2. compose real Middle Egyptian: VSO grammar, suffix pronouns,
#      honorific transposition, then the Leiden transliteration;
#   3. encode each word as Gardiner signs: phonograms + phonetic
#      complements + determinatives, as a scribe would spell it.

# Degrades gracefully like the other LLM services: no API key or any
# OpenAI failure returns a ReverseTranslationOut with `error` set.

logger = logging.getLogger('sphinxeyes.reverse')

# Canonical Gardiner code shape: category letter(s) + number + optional
# variant letter, e.g. M17, Aa1, N35, W25a. Used to sanitize LLM output.
_CODE_RE = re.compile(r'^(?:[A-Z]|Aa|NL|NU)\d{1,3}[a-z]?$')

SYSTEM_PROMPT = """You are an expert Egyptologist and philologist specializing in Middle Egyptian \
(the classical language of Dynasties XI-XVIII). Your task is REVERSE translation — composition: \
render modern English into genuine Middle Egyptian, exactly as a trained scribe would write it, \
and return machine-readable Gardiner sign codes.

METHOD — work in three explicit stages:

STAGE 1 — SEMANTIC NORMALIZATION (English -> translatable English):
- Rephrase the input into concepts that exist in the Middle Egyptian lexicon. Egyptian has no
  'computer' or 'coffee'; it has 'scribe', 'beer', 'boat', 'praise', 'eternity'.
- For untranslatable modern concepts choose the closest attested Egyptian notion and say so in
  the grammar notes (e.g. 'engineer' -> 'overseer of works', jmy-r kAt).
- Keep proper names: transcribe them phonetically using uniliteral signs, and if the name belongs
  to a king or is asked to be royal, enclose it in a cartouche.

STAGE 2 — COMPOSITION + TRANSLITERATION (translatable English -> Middle Egyptian):
- Compose real Middle Egyptian, not word-for-word English: VSO order where verbal, adverbial and
  nominal sentence patterns where appropriate (A B / A pw / jw + subject + stative), suffix
  pronouns (=j, =k, =f ...), dependent pronouns, genitival n(y)-t, nfr-Hr constructions.
- Prefer ATTESTED words and formulae over invented compounds: htp-di-nsw, anx wDA snb, mAa-xrw,
  nb tAwy, Dt nHH, jmy-r, sS (scribe), mr(y) (beloved).
- Apply honorific transposition: nTr, nsw, ra are WRITTEN first inside their phrase even though
  read later.
- Produce the transliteration in Unified Leiden conventions: aA, i/j, w, b, p, f, m, n, r, h, H,
  x, X, s, S, q, k, g, t, T, d, D; '=' for suffix pronouns, '.' for endings, ( ) for restored
  elements.

STAGE 3 — SIGN CODING (transliteration -> Gardiner codes):
- Spell each word as a scribe would: phonogram(s) + phonetic complement(s) + determinative.
  Do NOT skip determinatives — they are part of correct spelling (A1 man, B1 woman, A2 man with
  hand to mouth for speech/thought, D54 legs for motion, N5 sun for time, O49 for towns,
  Y1 papyrus roll for abstracts, Z1 stroke for logograms, Z2 plural strokes).
- Use CANONICAL Gardiner codes: capital category letter + number (M17, N35, Aa1, D21, X1, G43).
  Never invent codes. Cartouche contents are wrapped as: "<" codes... ">" is NOT valid — instead
  list the codes and set the word's note to 'cartouche'.
- Codes must be in READING order (the order they would be encountered reading the text),
  including honorific transposition already applied.
- Prefer common monumental signs when several spellings are attested.

OUTPUT — JSON only, no preamble, exactly these keys:
{
  "normalized_english": "the plain English actually translated (after Stage 1)",
  "transliteration": "full Leiden transliteration of the whole composition",
  "gardiner_codes": ["M17", "G43", ...]  (flat, whole text, reading order),
  "words": [
    {"english": "...", "transliteration": "...", "gardiner_codes": ["..."],
     "literal": "literal meaning", "note": "determinative/grammar/cartouche remark or ''"}
  ],
  "grammar_notes": "2-4 sentences: construction chosen, any Stage-1 substitutions, transposition applied",
  "confidence": "HIGH|MEDIUM|LOW"  (LOW when heavy modern-concept substitution was needed)
}
Answer in ENGLISH only."""


def _build_user_prompt(text: str, register: str) -> str:
    reg = ('a neutral monumental register'
           if register == 'unknown' else f'the {register} register')
    return (
        f'Render the following English into Middle Egyptian, using {reg}.\n'
        f'ENGLISH TEXT:\n{text}\n\n'
        f'Apply the three stages and answer with the JSON object only.'
    )


def _clean_codes(codes: object) -> list[str]:
    """Keep only syntactically valid canonical Gardiner codes."""
    if not isinstance(codes, list):
        return []
    out = []
    for c in codes:
        c = str(c).strip()
        if _CODE_RE.match(c):
            out.append(c)
            
        else:
            logger.warning(f'reverse: dropped malformed Gardiner code {c!r}')
            
    return out


class ReverseTranslationService:
    """One-shot English -> Middle Egyptian composition via GPT-4o."""

    def __init__(self) -> None:
        self._client = (OpenAI(api_key=settings.openai_api_key)
                        if settings.openai_api_key else None)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def translate(self, text: str, 
                  register: str = 'unknown'
                  ) -> ReverseTranslationOut:
        
        if not self.enabled:
            
            return ReverseTranslationOut(
                source_text=text, normalized_english='', transliteration='',
                gardiner_codes=[], model=settings.openai_model,
                error='OPENAI_API_KEY not configured — reverse translation disabled.',
            )

        try:
            response = self._client.chat.completions.create(
                model           = settings.openai_model,
                messages        = [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': _build_user_prompt(text, register)},
                ],
                temperature     = settings.translit_temperature,
                response_format = {'type': 'json_object'},
            )
            
            parsed = json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.exception('reverse translation LLM call failed')
            
            return ReverseTranslationOut(
                source_text=text, normalized_english='', transliteration='',
                gardiner_codes=[], model=settings.openai_model,
                error=f'LLM call failed: {e}',
            )

        words = []
        
        for w in parsed.get('words', []) or []:
            if not isinstance(w, dict):
                continue
            
            words.append(ReverseWord(
                english         = str(w.get('english', '')),
                transliteration = str(w.get('transliteration', '')),
                gardiner_codes  = _clean_codes(w.get('gardiner_codes')),
                literal         = str(w.get('literal', '')),
                note            = str(w.get('note', '')),
            ))

        flat = _clean_codes(parsed.get('gardiner_codes'))
        
        if not flat and words:                      # fall back to per-word codes
            flat = [c for w in words for c in w.gardiner_codes]

        confidence = parsed.get('confidence')
        if confidence not in ('HIGH', 'MEDIUM', 'LOW'):
            confidence = 'LOW'

        return ReverseTranslationOut(
            source_text        = text,
            normalized_english = str(parsed.get('normalized_english', '')),
            transliteration    = str(parsed.get('transliteration', '')),
            gardiner_codes     = flat,
            words              = words,
            grammar_notes      = str(parsed.get('grammar_notes', '')),
            confidence         = confidence,
            model              = settings.openai_model,
        )
