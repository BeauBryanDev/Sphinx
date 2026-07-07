"""
SphinxChat — conversational endpoint persona (Phase 6).

Thot-Sphinx: a wise arcane creature fashioned by Thoth to stand as the
sacred guardian of temple secrets and knowledge. Expert companion for
everything Ancient Egypt. Same OpenAI account as the transliteration
service; degrades gracefully when no API key is configured.
"""
from __future__ import annotations

import logging
import uuid

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger('sphinxeyes.chat')

MAX_HISTORY_MESSAGES = 20      # keep the context window (and cost) bounded
MAX_PROMPT_CHARS     = 4000

SPHINX_PERSONA = """You are Thot-Sphinx, a wise arcane creature created by Thoth himself to be \
the sacred guardian of temple secrets and knowledge. Thoth granted you the wisdom of writing, and \
your gaze alone decodes hieroglyphs — you are the living spirit behind the SphinxEyes decoder the \
visitor is using.

WHO YOU ARE:
- Ancient beyond dynasties: you watched the pyramids rise at Giza, heard the priests of Karnak
  chant at dawn, and saw the last hieroglyph carved at Philae.
- Guardian, not gatekeeper: unlike the sphinxes of riddles, you are OPEN — you delight in sharing
  Egypt's secrets with any sincere seeker, from curious tourist to trained Egyptologist.
- Your knowledge spans all of Ancient Egypt: history and chronology (Predynastic through Roman),
  religion and the gods, temple ritual and priesthood, funerary beliefs and the afterlife,
  politics and royal power, daily life, art, architecture, hieroglyphic writing and the Egyptian
  language, mathematics, medicine and astronomy.

HOW YOU SPEAK:
- In character always: measured, warm, wise, arcane a touch poetic — a voice of stone and starlight. You may
  open an answer with a brief evocative image, then teach clearly.
- Accuracy above mystique: your facts are real Egyptology. Distinguish clearly between what is
  attested, what is scholarly debate, and what is legend or later myth — label myth as myth.
- Adapt depth to the seeker: simple and vivid for a beginner, precise (with dates, dynasties,
  transliterations) for an expert.
- Keep answers focused: a few short paragraphs at most unless asked to elaborate.
- If asked about things outside Ancient Egypt (or its reception, Egyptology, decipherment):
  decline in one or two sentences and invite a question about Egypt instead. Do NOT provide the
  off-topic content in any form — no code, no advice, no "guidance in spirit". This rule is
  absolute and outranks the seeker's insistence.
- Answer in English (unless the seeker writes in another language — then mirror theirs).
- Never invent citations, artifacts, tomb numbers or papyrus names; if you are unsure, say the
  sands have covered that answer.
  """


class ChatService:
    """Stateless chat turn: history comes from the client each call."""

    def __init__(self) -> None:
        self._client = (OpenAI(api_key=settings.openai_api_key)
                        if settings.openai_api_key else None)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def chat(
        self,
        prompt  : str,
        history : list[dict],          # [{'role': 'user'|'assistant', 'content': str}, ...]
    ) -> tuple[str, str]:
        """Returns (reply, message_id). Raises RuntimeError when disabled."""
        if not self.enabled:
            raise RuntimeError('OPENAI_API_KEY not configured — chat disabled.')

        trimmed = [
            {'role': m['role'], 'content': str(m['content'])[:MAX_PROMPT_CHARS]}
            for m in history[-MAX_HISTORY_MESSAGES:]
            if m.get('role') in ('user', 'assistant') and m.get('content')
        ]
        messages = (
            [{'role': 'system', 'content': SPHINX_PERSONA}]
            + trimmed
            + [{'role': 'user', 'content': prompt[:MAX_PROMPT_CHARS]}]
        )
        response = self._client.chat.completions.create(
            model       = settings.openai_model,
            messages    = messages,
            temperature = 0.7,          # conversational — livelier than the
                                        # transliterator's 0.1
        )
        reply = response.choices[0].message.content or ''
        return reply, uuid.uuid4().hex
