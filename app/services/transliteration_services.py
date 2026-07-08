"""
LLM transliteration + translation stage (Phase 6).

English port of the prototype scripts `prompt2GPT.py` and
`segement_into_chunck.py` (repo root), adapted to the real pipeline
output shape:

  - chunks are cut at the spatial layer's line boundaries
    (`boundary_hints`) first, then at `max_signs`, so a chunk never
    straddles a physical line of text;
  - matched cartouches are injected as authoritative context (the
    Needleman-Wunsch match against royal_names.json outranks anything
    the LLM would guess from raw codes);
  - per-sign confidence comes from the detector's top-1 slot scores.

The service degrades gracefully: no API key, or any OpenAI failure,
returns a TransliterationOut with `error` set — never breaks /predict/.
"""
from __future__ import annotations

import json
import logging

from openai import OpenAI

from app.core.config import settings
from app.schemas.transliterations import (
    ChunkTransliteration,
    TextContext,
    TransliterationOut,
)

logger = logging.getLogger('sphinxeyes.transliteration')


# Chunking (port of segement_into_chunck.segment_into_chunks)

def segment_into_chunks(
    codes           : list[str],
    confidences     : list[float],
    boundary_hints  : list[int],
    max_signs       : int = 20,
) -> list[list[tuple[str, float]]]:
    """
    Split the ordered (code, conf) sequence into chunks of at most
    `max_signs`, cutting preferentially at line boundaries so a chunk
    never mixes signs from different physical lines.
    """
    pairs = list(zip(codes, confidences))
    bounds = sorted(b for b in boundary_hints if 0 < b <= len(pairs))
    if not bounds or bounds[-1] != len(pairs):
        bounds.append(len(pairs))

    chunks: list[list[tuple[str, float]]] = []
    start = 0
    for b in bounds:
        line = pairs[start:b]
        # split an over-long line at max_signs
        for i in range(0, len(line), max_signs):
            piece = line[i:i + max_signs]
            if piece:
                chunks.append(piece)
        start = b
    return chunks


# Prompt (port of prompt2GPT.build_egyptologist_prompt)

# The role/methodology half of the prompt. Static -> sent as the system
# message (also lets OpenAI cache it across the per-chunk calls).
SYSTEM_PROMPT = """You are an expert Egyptologist and philologist specializing in Middle Egyptian \
(the classical language of Dynasties XI-XVIII, also used ceremonially long after). You read \
hieroglyphic inscriptions from Gardiner sign codes and produce scholarly transliterations and \
English glosses.

HOW THE INPUT WAS PRODUCED (important for judging its reliability):
The Gardiner sequence comes from a computer-vision pipeline, not a human copyist:
1. A YOLO detector recognizes individual signs on the photograph (each has a confidence score).
2. A spatial algorithm reconstructs reading order (quadrat stacking, line breaks).
3. A lexicon-based corrector (Viterbi over a dictionary trie + bigram model) may have already
   substituted some low-confidence detections.
4. Cartouches are matched against a verified royal-name lexicon (Beckerath) — when a royal name
   is given, it is MORE reliable than the raw sign codes around it.
Consequences you must handle:
- Signs may be MISCLASSIFIED as visually similar signs. Frequent confusions of this detector:
  G1 (vulture) <-> G5 (falcon) <-> G39 (duck); N5 (sun disc) <-> N33 (pellet) <-> Aa1 (placenta)
  <-> O49 (town); X1 (bread loaf) <-> X8 (conical loaf); S29 (folded cloth) <-> O34 (door bolt);
  W19 (milk jug) <-> W14 (water jar); M23 (sedge) <-> M22 (rush); D21 (mouth) <-> D4 (eye);
  Y1 <-> Y2 (papyrus rolls); Z1 <-> Z4 (strokes). When a LOW or MED confidence sign yields
  nonsense but one of its confusion partners yields coherent Middle Egyptian, prefer the partner
  and say so in the notes.
- A sign may be MISSING (detector miss) — small phonetic complements and determinatives are the
  usual casualties. You may posit an omitted complement when the reading obviously requires it.
- 'Unknown' tokens are undetected signs: treat them as lacunae, transliterate as [...].
- Reading order is usually right but not guaranteed within a quadrat; minor transpositions
  (e.g. honorific transposition of nTr / nsw / ra) should be restored silently.

METHOD — work like a philologist, not a code mapper:
1. Segment the sign string into words: identify uniliterals, biliterals, triliterals, phonetic
   complements (do not transliterate a complement twice), determinatives (classify, never
   pronounce), and logograms.
2. Look for the high-frequency formulae of monumental texts and let them anchor the reading:
   htp-di-nsw (offering formula), sA ra (son of Ra), nb tAwy (lord of the Two Lands),
   nTr nfr (the good god), di anx (given life), mAa-xrw (true of voice), anx wDA snb,
   nswt-bity (dual king), Dt / nHH (forever), epithets of deities and royal titulary.
3. If a royal cartouche is identified, use it as the chronological and thematic anchor: titles
   and epithets adjacent to a cartouche almost always belong to the standard titulary sequence.
4. Use the archaeological context: a temple wall favours royal/divine formulae; a stela favours
   the offering formula and filiation (X sA Y, mAat-xrw); pyramid texts favour Old Kingdom
   spellings; a papyrus may be literary or administrative.
5. Commit to ONE most-probable reading. Note real alternatives briefly instead of hedging.

OUTPUT — JSON only, no preamble, exactly these keys:
{
  "transliteration": "Unified Leiden conventions (aA not aleph-glyph fallback; use . for suffixes, = for clitics, [...] for lacunae, ( ) for restored signs)",
  "english_gloss": "~ one plain-English sentence, functional gloss for non-specialists",
  "linguistic_notes": "max 2 sentences: key ambiguity, any confusion-pair substitution you made, notable grammar",
  "confidence": "HIGH|MEDIUM|LOW",
  "period_note": "one short remark tying the reading to the stated period/reign, or '' if context was unknown"
}
Answer in ENGLISH only."""


def build_egyptologist_prompt(
    codes            : list[str],
    confidences      : list[float],
    ctx              : TextContext,
    direction        : str,
    layout           : str,
    cartouche_names  : list[str],
    previous_context : str | None,
    chunk_info       : str,
) -> str:
    """Build the per-chunk USER message (the system message is static)."""
    signs = ' — '.join(
        f'{c}({"HIGH" if s >= 0.80 else "MED" if s >= 0.50 else "LOW"}:{s:.2f})'
        for c, s in zip(codes, confidences)
    )

    def h(v: str) -> str:                     # human-readable context value
        return v.replace('_', ' ')

    cartouche_block = (
        'Royal cartouches identified in this scene (lexicon-verified — '
        'treat as authoritative, more reliable than raw sign codes):\n  '
        + '\n  '.join(cartouche_names)
        if cartouche_names else 'Contains cartouche: no'
    )
    previous_block = (
        f'\nPREVIOUS SEGMENTS of the same inscription (already transliterated '
        f'— keep names, epithets and topic consistent with them):\n'
        f'{previous_context}\n' if previous_context else ''
    )

    return f"""DETECTED SIGN SEQUENCE ({chunk_info}; each segment is one physical line of text; Gardiner codes with detector confidence):
{signs}

ARCHAEOLOGICAL CONTEXT (fields marked 'unknown' were not supplied by the user — do not invent them, but exploit every field that IS given):
- Period: {h(ctx.period)}
- Dynasty: {ctx.dynasty}
- King's reign: {ctx.kings_reign}
- Text type: {h(ctx.text_type)}
- Physical support: {h(ctx.support)}
- Location type: {h(ctx.location_type)}
- Site: {ctx.site}
- Reading direction: {direction}
- Layout: {layout}
- {cartouche_block}
{previous_block}
Transliterate and gloss this segment. JSON only."""



# Service (port of segement_into_chunck.transliterate_wall)
class TransliterationService:
    """Chunk the corrected sequence, call the LLM per chunk, assemble."""

    def __init__(self) -> None:
        self._client = (OpenAI(api_key=settings.openai_api_key)
                        if settings.openai_api_key else None)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def transliterate(
        self,
        raw       : dict,          # SphinxPipeline.run() output
        ctx       : TextContext,
    ) -> TransliterationOut:
        outer = raw['outer']
        codes = outer['correction']['flat_corrected_seq']
        confs = [slot[0][1] if slot else 0.0 for slot in outer['slots']]
        # corrected seq and slots are index-aligned; guard anyway
        if len(confs) != len(codes):
            confs = (confs + [0.0] * len(codes))[:len(codes)]

        cartouche_names = [
            f"{c['translit']} — {c['english']} (interior: {' '.join(c['spelling'] or [])})"
            for c in raw['cartouches'] if c.get('translit')
        ]
        return self.transliterate_sequence(
            codes, confs, outer['boundary_hints'], cartouche_names,
            direction=raw['direction'], layout=raw['layout'], ctx=ctx,
        )

    def transliterate_sequence(
        self,
        codes           : list[str],
        confidences     : list[float],
        boundary_hints  : list[int],
        cartouche_names : list[str],
        *,
        direction       : str,
        layout          : str,
        ctx             : TextContext,
    ) -> TransliterationOut:
        if not self.enabled:
            return TransliterationOut(
                chunks=[], full_transliteration='', full_translation='',
                model=settings.openai_model, n_chunks=0,
                error='OPENAI_API_KEY not configured — transliteration disabled.',
            )

        chunks = segment_into_chunks(
            codes, confidences, boundary_hints,
            max_signs=settings.translit_max_signs,
        )

        results: list[ChunkTransliteration] = []
        for i, chunk in enumerate(chunks):
            c_codes = [c for c, _ in chunk]
            c_confs = [s for _, s in chunk]
            previous = ' | '.join(r.transliteration for r in results) or None

            prompt = build_egyptologist_prompt(
                c_codes, c_confs, ctx,
                direction=direction, layout=layout,
                cartouche_names=cartouche_names,
                previous_context=previous,
                chunk_info=f'Segment {i + 1} of {len(chunks)}',
            )
            try:
                response = self._client.chat.completions.create(
                    model           = settings.openai_model,
                    messages        = [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user',   'content': prompt},
                    ],
                    temperature     = settings.translit_temperature,
                    response_format = {'type': 'json_object'},
                )
                parsed = json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.exception(f'LLM transliteration failed on chunk {i}')
                return TransliterationOut(
                    chunks=results,
                    full_transliteration=' | '.join(r.transliteration for r in results),
                    full_translation=' '.join(r.english_gloss for r in results),
                    model=settings.openai_model, n_chunks=len(chunks),
                    error=f'LLM call failed on segment {i + 1}/{len(chunks)}: {e}',
                )

            results.append(ChunkTransliteration(
                chunk_index      = i,
                gardiner_codes   = c_codes,
                transliteration  = str(parsed.get('transliteration', '')),
                english_gloss    = str(parsed.get('english_gloss', '')),
                linguistic_notes = str(parsed.get('linguistic_notes', '')),
                confidence       = parsed.get('confidence', 'LOW')
                                   if parsed.get('confidence') in ('HIGH', 'MEDIUM', 'LOW')
                                   else 'LOW',
                period_note      = str(parsed.get('period_note', '')),
            ))

        return TransliterationOut(
            chunks               = results,
            full_transliteration = ' | '.join(r.transliteration for r in results),
            full_translation     = ' '.join(r.english_gloss for r in results),
            model                = settings.openai_model,
            n_chunks             = len(results),
        )
