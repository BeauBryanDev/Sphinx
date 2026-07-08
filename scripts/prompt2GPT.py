def build_egyptologist_prompt(detection_data: dict) -> str:
    ctx = detection_data["context"]
    seq = detection_data["gardiner_sequence"]
    scores = detection_data["confidence_scores"]

 
    signs_with_conf = []
    for sign, conf in zip(seq, scores):
        flag = "HIGH" if conf >= 0.80 else "MED" if conf >= 0.50 else "LOW"
        signs_with_conf.append(f"{sign}({flag}:{conf:.2f})")
    signs_str = " — ".join(signs_with_conf)

    # Natural langague Context
    period_str = f"{ctx['period']} ({ctx['period_bce']} BCE)" \
                 if ctx.get('period_bce') else ctx['period']

    prompt = f"""You are an expert Egyptologist and philologist specializing in Middle Egyptian hieroglyphic texts.

DETECTED SIGN SEQUENCE (Gardiner codes with detection confidence):
{signs_str}

ARCHAEOLOGICAL CONTEXT:
- Period: {period_str}
- Text type: {ctx['text_type'].replace('_', ' ')}
- Physical support: {ctx['support'].replace('_', ' ')}
- Site: {ctx['site']}
- Reading direction: {ctx['reading_direction']}
- layout { ctx['col'] . ctx['rows'] }
- Layout: {ctx['layout'].replace('_', ' ')}
- Contains cartouche: {ctx['contains_cartouche']}
- Context confidence: {ctx['context_confidence']}

TASK:
Based on the Gardiner sign sequence and archaeological context above, provide:

1. TRANSLITERATION: Transcribe the sequence using Unified Leiden Transliteration conventions. 
   Use the confidence levels as guidance — HIGH confidence signs are likely correct, 
   LOW confidence signs may need alternative readings.

2. ENGLISH GLOSS: A single sentence semantic approximation in plain English. 
   This is a functional gloss for non-specialists, NOT a scholarly translation.
   Prefix with ~ to indicate approximation.

3. LINGUISTIC NOTES: Maximum 2 sentences explaining any ambiguity or notable features.

4. CONFIDENCE ASSESSMENT: Rate overall translation confidence as HIGH / MEDIUM / LOW
   based on sequence completeness and context quality.

CRITICAL CONSTRAINTS:
- Do NOT provide multiple alternative translations unless absolutely necessary
- Do NOT hedge excessively — commit to the most probable reading
- If a sign has LOW detection confidence, note it but still provide the best reading
- Base your reading on {ctx['text_type'].replace('_', ' ')} conventions for {period_str}

OUTPUT FORMAT — JSON only, no preamble:
{{
  "transliteration": "...",
  "english_gloss": "~ ...",
  "linguistic_notes": "...",
  "confidence": "HIGH|MEDIUM|LOW",
  "period_note": "..."
}}"""

    return prompt
