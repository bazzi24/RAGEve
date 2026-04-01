from __future__ import annotations

import re

from rag.chunking.high_accuracy import deepdoc_chunk_text
from rag.deepdoc.quality_scorer import ChunkProfile, PROFILE_PRESETS


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
TABLE_SPLIT_RE = re.compile(r"\n(?=\[Sheet:|\||\d+,)")
CODE_BLOCK_RE = re.compile(r"(?s)```[\s\S]*?```|`[^`\n]+`")


def _split_by_tables(text: str) -> list[str]:
    """Split text preserving table blocks as single units."""
    parts = TABLE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_code_blocks(text: str) -> list[str]:
    """Split text preserving code blocks as single units."""
    parts = CODE_BLOCK_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def adaptive_chunk_text(
    text: str,
    profile: ChunkProfile,
    *,
    override_size: int | None = None,
    override_overlap: int | None = None,
    override_tokens: int | None = None,
) -> list[str]:
    """
    Adaptive chunking that respects document structure.
    Uses the selected chunk profile, but allows overrides.
    """
    text = (text or "").strip()
    if not text:
        return []

    config = PROFILE_PRESETS[profile]
    chunk_size = override_size or config.chunk_size
    chunk_overlap = override_overlap or config.chunk_overlap
    max_tokens = override_tokens or config.max_tokens_per_chunk

    # Profile-specific pre-processing
    if profile == ChunkProfile.TABLE_HEAVY:
        # Split by table boundaries first, then run standard chunking
        table_parts = _split_by_tables(text)
        all_chunks: list[str] = []
        for part in table_parts:
            chunks = deepdoc_chunk_text(
                part,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_tokens_per_chunk=max_tokens,
            )
            all_chunks.extend(chunks)
        return _dedupe_chunks(all_chunks)

    if profile == ChunkProfile.CODE_MIXED:
        # Split by code blocks, chunk code parts smaller, text parts normally
        code_parts = _split_by_code_blocks(text)
        all_chunks: list[str] = []
        for part in code_parts:
            if CODE_BLOCK_RE.search(part):
                # This is a code block — treat as one chunk if small enough
                if len(part) <= chunk_size * 2:
                    all_chunks.append(part)
                else:
                    sub_chunks = deepdoc_chunk_text(
                        part,
                        chunk_size=max(chunk_size // 2, 300),
                        chunk_overlap=chunk_overlap,
                        max_tokens_per_chunk=max(max_tokens // 2, 150),
                    )
                    all_chunks.extend(sub_chunks)
            else:
                chunks = deepdoc_chunk_text(
                    part,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_tokens_per_chunk=max_tokens,
                )
                all_chunks.extend(chunks)
        return _dedupe_chunks(all_chunks)

    # Standard profiles (CLEAN_TEXT, OCR_NOISY, GENERAL)
    return deepdoc_chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_tokens_per_chunk=max_tokens,
    )


def _dedupe_chunks(chunks: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for chunk in chunks:
        key = chunk.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
