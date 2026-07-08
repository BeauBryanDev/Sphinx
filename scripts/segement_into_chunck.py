from .prompt2GPT import build_egyptologist_prompt

def segment_into_chunks(ordered_detections, max_signs=20):
    """
    Divide la secuencia ordenada de signos en chunks
    de máximo max_signs para enviar a GPT.
    Respeta los límites de línea detectados por el DSA.
    """
    chunks = []
    current_chunk = []

    for detection in ordered_detections:
        # Si es un cartucho, siempre va solo en su chunk
        if detection['class'] == 'cartouche':
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            chunks.append([detection])  # cartucho solo
            continue

        current_chunk.append(detection)

        # Cortar al llegar al límite
        if len(current_chunk) >= max_signs:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def transliterate_wall(ordered_detections, context, openai_client):
    """
    Transliterar una pared completa dividida en chunks.
    Retorna la transliteración completa concatenada.
    """
    chunks = segment_into_chunks(ordered_detections, max_signs=20)
    results = []

    for i, chunk in enumerate(chunks):
        # Contexto acumulativo — GPT sabe qué vino antes
        previous_text = " | ".join([r['transliteration']
                                    for r in results]) if results else None

        prompt = build_egyptologist_prompt({
            "gardiner_sequence": [d['class'] for d in chunk],
            "confidence_scores": [d['conf'] for d in chunk],
            "context": context,
            "previous_context": previous_text,  # chunk anterior
            "chunk_info": f"Segment {i+1} of {len(chunks)}"
        })

        response = openai_client.chat.completions.create(
            model    = "gpt-4o",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.1  # bajo — queremos consistencia, no creatividad
        )

        import json
        result = json.loads(
            response.choices[0].message.content
        )
        results.append(result)

    return results