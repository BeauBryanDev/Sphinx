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
import os
import time
from pathlib import Path

from openai import OpenAI

from app.core.config import settings
from app.schemas.transliterations import (
    ChunkTransliteration,
    TextContext,
    TransliterationOut,
)

logger = logging.getLogger('sphinxeyes.transliteration')

# Debug: dump the EXACT messages sent to OpenAI (system + per-chunk user
# prompts + raw LLM replies) to a JSON file under debug_prompts/ AND echo a
# summary to the server console. Enable with SPHINX_DEBUG_PROMPTS=1 in .env
# (read via settings, per-request — no restart-with-exported-var needed).
# Payload only — never the API key.
DEBUG_DIR = Path(__file__).resolve().parents[2] / 'debug_prompts'


def _debug_enabled() -> bool:
    # settings.debug_prompts loads SPHINX_DEBUG_PROMPTS from .env; the env
    # var still wins if it's exported in the shell.
    return settings.debug_prompts or os.getenv('SPHINX_DEBUG_PROMPTS') == '1'


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
        # blank/whitespace (client sent '' after the user erased a field)
        # must read as 'unknown', not an empty line the LLM could misread
        return (v or '').strip().replace('_', ' ') or 'unknown'

    # Matched names are authoritative; UNRESOLVED entries (match refused)
    # carry raw interior signs only — never present those as verified.
    matched    = [n for n in cartouche_names if not n.startswith('[UNRESOLVED')]
    unresolved = [n for n in cartouche_names if n.startswith('[UNRESOLVED')]
    parts = []
    if matched:
        parts.append(
            'Royal cartouches identified in this scene (lexicon-verified — '
            'treat as authoritative, more reliable than raw sign codes):\n  '
            + '\n  '.join(matched))
    if unresolved:
        parts.append(
            'Cartouches detected but NOT resolved to a known royal name — '
            'read their raw interior signs yourself (a royal name or epithet '
            'is likely; do not invent a specific king):\n  '
            + '\n  '.join(unresolved))
    cartouche_block = '\n- '.join(parts) if parts else 'Contains cartouche: no'
    previous_block = (
        f'\nPREVIOUS SEGMENTS of the same inscription (already transliterated '
        f'— keep names, epithets and topic consistent with them):\n'
        f'{previous_context}\n' if previous_context else ''
    )

    return f"""DETECTED SIGN SEQUENCE ({chunk_info}; each segment is one physical line of text; Gardiner codes with detector confidence):
{signs}

ARCHAEOLOGICAL CONTEXT (fields marked 'unknown' were not supplied by the user — do not invent them, but exploit every field that IS given):
- Period: {h(ctx.period)}
- Dynasty: {h(ctx.dynasty)}
- King's reign: {h(ctx.kings_reign)}
- Text type: {h(ctx.text_type)}
- Physical support: {h(ctx.support)}
- Location type: {h(ctx.location_type)}
- Site: {h(ctx.site)}
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

    @staticmethod
    def _write_debug(dump: dict) -> None:
        """Persist the exact LLM payload for inspection (overwrites the
        'latest' file each request; keeps one timestamped copy too)."""
        try:
            DEBUG_DIR.mkdir(exist_ok=True)
            latest = DEBUG_DIR / 'latest.json'
            latest.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
            stamped = DEBUG_DIR / f"payload_{dump['timestamp'].replace(':', '-').replace(' ', '_')}.json"
            stamped.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
            logger.info(f'LLM payload dumped to {latest}')
        except Exception:
            logger.exception('failed to write debug prompt dump')

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

        cartouche_names = []
        for c in raw['cartouches']:
            if c.get('translit'):
                cartouche_names.append(
                    f"{c['translit']} — {c['english']} (interior: {' '.join(c['spelling'] or [])})"
                )
            elif c.get('n_members', 0) > 0:
                raw_codes = ' '.join(
                    slot[0][0] for slot in c.get('slots', []) if slot
                )
                if raw_codes:
                    cartouche_names.append(
                        f"[UNRESOLVED cartouche — raw signs detected but no confident "
                        f"royal-name match: {raw_codes}]"
                    )
                    
            else :
                cartouche_names.append(f"[UNRESOLVED cartouche — no signs detected]")
        
        
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

        debug_dump: dict = {
            'timestamp'       : time.strftime('%Y-%m-%d %H:%M:%S'),
            'model'           : settings.openai_model,
            'temperature'     : settings.translit_temperature,
            'input'           : {
                'codes'           : codes,
                'confidences'     : [round(c, 4) for c in confidences],
                'boundary_hints'  : boundary_hints,
                'cartouche_names' : cartouche_names,
                'direction'       : direction,
                'layout'          : layout,
                'context'         : ctx.model_dump(),
            },
            'system_prompt'   : SYSTEM_PROMPT,
            'chunks'          : [],
        } if _debug_enabled() else None

        if debug_dump is not None:
            # Echo the key facts to the server console so they're visible in
            # the uvicorn CLI without opening the JSON dump. print(flush) —
            # the app configures no logging handler, so sphinxeyes.* INFO
            # logs would be swallowed by the root logger's WARNING default.
            lines = [
                '=' * 70,
                'SPHINX_DEBUG_PROMPTS — LLM request',
                f'  model={settings.openai_model} '
                f'temp={settings.translit_temperature} chunks={len(chunks)}',
                f'  codes ({len(codes)}): {" ".join(codes)}',
                f'  cartouche_names -> LLM ({len(cartouche_names)}):',
                *(f'    - {cn}' for cn in cartouche_names),
            ]
            if not cartouche_names:
                lines.append('    (none — LLM will see no royal names!)')
            lines.append('=' * 70)
            print('\n'.join(lines), flush=True)

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
            if debug_dump is not None:
                debug_dump['chunks'].append({
                    'chunk_index'  : i,
                    'user_prompt'  : prompt,
                    'llm_raw_reply': None,          # filled after the call
                })
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
                raw_reply = response.choices[0].message.content
                if debug_dump is not None:
                    debug_dump['chunks'][-1]['llm_raw_reply'] = raw_reply
                    self._write_debug(debug_dump)
                parsed = json.loads(raw_reply)
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
