from __future__ import annotations

import re


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _estimate_tokens(text: str) -> int:
    # Cheap token approximation good enough for chunk boundary control.
    return max(1, int(len(text) / 4))


def _split_into_sentences(block: str) -> list[str]:
    block = block.strip()
    if not block:
        return []

    parts = SENTENCE_SPLIT_RE.split(block)
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if cleaned:
        return cleaned
    return [block]


def _split_long_sentence(sentence: str, max_chars: int, overlap: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    if len(words) <= 1:
        # Fall back to fixed-width slicing with overlap instead of zero-overlap stride
        return [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars - overlap)]

    result: list[str] = []
    buff: list[str] = []
    current_len = 0

    for word in words:
        next_len = current_len + len(word) + (1 if buff else 0)
        if next_len > max_chars and buff:
            result.append(" ".join(buff))
            buff = [word]
            current_len = len(word)
        else:
            buff.append(word)
            current_len = next_len

    if buff:
        result.append(" ".join(buff))

    return result


def deepdoc_chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
    max_tokens_per_chunk: int = 500,
) -> list[str]:
    """
    High-accuracy chunking with multi-level boundaries:
    1) paragraph boundary
    2) sentence boundary
    3) fallback word split for long sentence
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text) if p and p.strip()]

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_chars = 0

    for paragraph in paragraphs:
        for sentence in _split_into_sentences(paragraph):
            for segment in _split_long_sentence(sentence, chunk_size, chunk_overlap):
                seg_len = len(segment)
                candidate_chars = current_chars + seg_len + (1 if current_sentences else 0)
                candidate_tokens = _estimate_tokens(" ".join(current_sentences + [segment]))

                if current_sentences and (candidate_chars > chunk_size or candidate_tokens > max_tokens_per_chunk):
                    chunk_text = " ".join(current_sentences).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                    if chunk_overlap > 0 and chunks:
                        overlap_text = chunks[-1][-chunk_overlap:].strip()
                        current_sentences = [overlap_text, segment] if overlap_text else [segment]
                        current_chars = len(" ".join(current_sentences))
                    else:
                        current_sentences = [segment]
                        current_chars = seg_len
                else:
                    current_sentences.append(segment)
                    current_chars = candidate_chars

    if current_sentences:
        final_chunk = " ".join(current_sentences).strip()
        if final_chunk:
            chunks.append(final_chunk)

    # de-duplicate overlap-only duplicates while preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = chunk.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)

    return deduped
