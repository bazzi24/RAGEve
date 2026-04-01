"""
Context builder — converts retrieved SearchResult chunks into an LLM prompt string.

This is the single canonical interface between the retrieval layer and the generation
layer. Keeping it here (instead of inside rag_pipeline.py) means it can be unit-tested
without running Ollama or Qdrant.

Schema expectations
-----------------
Every chunk stored in Qdrant carries these fields in its payload:

  Required (always present):
    dataset_id   — str  — HuggingFace ID or user-supplied dataset name
    split         — str  — "train" / "test" / "manual" etc.
    quality_score — float — [0, 1] quality signal from quality_scorer
    profile       — str  — adaptive-chunking profile ("technical", "narrative", etc.)
    source_file   — str  — originating file name

  HF-only (present when ingested via hf_ingestion.py):
    text_columns_used   — list[str] — columns that were combined into the chunk text
    total_chunks_in_row  — int       — how many chunks this row produced
    chunk_index          — int        — position of this chunk within its row

  Manual-only (present when ingested via ingestion_service.py):
    extension           — str — source file extension e.g. ".pdf"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkAnnotation:
    """Structured metadata extracted from a SearchResult for prompt annotation."""

    dataset_id: str
    split: str
    quality: float | None
    profile: str | None
    source_file: str | None
    text_columns_used: list[str] | None = None
    chunk_index: int | None = None
    total_chunks_in_row: int | None = None


def annotation_from_chunk(chunk: "SearchResult") -> ChunkAnnotation:  # noqa: F821
    """Extract structured metadata from a SearchResult's payload."""
    meta = chunk.metadata or {}
    return ChunkAnnotation(
        dataset_id=meta.get("dataset_id", "unknown"),
        split=meta.get("split", "unknown"),
        quality=meta.get("quality_score"),
        profile=meta.get("profile"),
        source_file=meta.get("source_file"),
        text_columns_used=meta.get("text_columns_used"),
        chunk_index=meta.get("chunk_index"),
        total_chunks_in_row=meta.get("total_chunks_in_row"),
    )


def build_context(chunks: list["SearchResult"]) -> str:  # noqa: F821
    """
    Build the context string from SearchResult chunks for the LLM prompt.

    Each chunk is annotated with its dataset, split, quality score, and source
    so the LLM can reason about provenance and reliability.

    Returns
    -------
    str
        A human-readable context block formatted as:

        [Source 1 | Dataset: imdb | Split: train | Quality: 0.834 | Profile: narrative | File: train.parquet]
        The film opens with a wide shot of ...
        ---
        [Source 2 | Dataset: squad | Split: train | Quality: 0.912 | Profile: technical | File: data.parquet]
        The capital of France is Paris...
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        ann = annotation_from_chunk(chunk)

        meta_parts: list[str] = [f"Dataset: {ann.dataset_id}"]
        meta_parts.append(f"Split: {ann.split}")

        if ann.quality is not None:
            meta_parts.append(f"Quality: {float(ann.quality):.3f}")

        if ann.profile:
            meta_parts.append(f"Profile: {ann.profile}")

        if ann.source_file:
            meta_parts.append(f"File: {ann.source_file}")

        header = f"[Source {i} | {' | '.join(meta_parts)}]"
        parts.append(f"{header}\n{chunk.chunk_text}\n")

    return "\n---\n".join(parts)
