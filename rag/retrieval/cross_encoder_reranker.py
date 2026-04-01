"""
Cross-encoder reranker for query–document relevance scoring.

Uses sentence-transformers to load a local cross-encoder model that jointly
encodes (query, document) pairs.  The cross-encoder score measures relevance,
which is a different signal from bi-encoder cosine similarity used in retrieval.

Model registry — all models are downloaded from HuggingFace on first use.
Override path with HF_HUB_CACHE env var.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING

_log = logging.getLogger("rag.retrieval.cross_encoder_reranker")

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@dataclass(frozen=True)
class RerankerEntry:
    """One entry in the AVAILABLE_RERANKERS registry."""

    id: str
    display_name: str
    description: str
    approx_size_mb: int


# ----------------------------------------------------------------------
# Registry of available local cross-encoder reranker models.
# ----------------------------------------------------------------------
AVAILABLE_RERANKERS = (
    RerankerEntry(
        id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        display_name="MS MARCO MiniLM (fast)",
        description="Lightweight model, ~80 MB. Good quality for English. Fast inference.",
        approx_size_mb=80,
    ),
    RerankerEntry(
        id="cross-encoder/ms-marco-MiniLM-L-12-v2",
        display_name="MS MARCO MiniLM (accurate)",
        description="Slightly deeper 12-layer model, ~120 MB. Better accuracy than L-6.",
        approx_size_mb=120,
    ),
    RerankerEntry(
        id="cross-encoder/ms-marco-Multi-Lingual-L-6-v2",
        display_name="MS MARCO Multilingual",
        description="~220 MB. Supports 100+ languages. Use for multilingual or non-English corpora.",
        approx_size_mb=220,
    ),
    RerankerEntry(
        id="BAAI/bge-reranker-base",
        display_name="BGE Reranker Base",
        description=(
            "~220 MB. BAAI's general-purpose reranker. "
            "Strong on semantic / cross-modal relevance. Best overall quality."
        ),
        approx_size_mb=220,
    ),
    RerankerEntry(
        id="BAAI/bge-reranker-large",
        display_name="BGE Reranker Large",
        description=(
            "~1.1 GB. Largest BGE reranker. Highest accuracy but slower and "
            "memory-intensive. Only use on a machine with >4 GB VRAM / RAM headroom."
        ),
        approx_size_mb=1100,
    ),
)


# ----------------------------------------------------------------------
# In-memory model cache — avoid re-loading the same model per request.
# Key: model_id string, Value: loaded CrossEncoder instance.
# ----------------------------------------------------------------------
_model_cache: dict[str, "CrossEncoder"] = {}

# MS MARCO models output raw logits (unbounded ±∞); BGE models apply sigmoid (∈ [0,1]).
# We normalize all scores to [0, 1] per-query using min-max scaling so that:
#   (a) MS MARCO and BGE scores are comparable in the frontend
#   (b) score_threshold and display values are always interpretable
_MS_MARCO_PREFIX = "cross-encoder/ms-marco"


def _load_model(model_id: str) -> "CrossEncoder":
    """Lazily load (and cache) a cross-encoder model by HuggingFace model ID."""
    if model_id not in _model_cache:
        from sentence_transformers import CrossEncoder

        _log.info("Loading cross-encoder model: %s", model_id)
        t0 = _time.monotonic()
        _model_cache[model_id] = CrossEncoder(model_id)
        _log.info("Cross-encoder model loaded: %s (%.1fs)", model_id, _time.monotonic() - t0)
    return _model_cache[model_id]


def _normalize_scores(scores: list[float], model_id: str) -> list[float]:
    """
    Normalize raw cross-encoder scores to [0, 1] per query.

    BGE rerankers (BAAI/) already return sigmoid scores ∈ [0, 1] — pass through.
    MS MARCO models return unbounded logits — min-max scale within this result set.

    Normalizing within each query's result set preserves relative ordering
    (the only thing that matters for reranking) while making absolute values
    comparable across model families in the frontend SourcesPanel.
    """
    if not scores:
        return scores

    # BGE models: already in [0, 1] — no transformation needed.
    if not model_id.startswith(_MS_MARCO_PREFIX):
        return scores

    mn, mx = min(scores), max(scores)
    if mx == mn:
        # All chunks scored identically — normalize to 0.5 so something is shown
        return [0.5] * len(scores)

    return [(s - mn) / (mx - mn) for s in scores]


# ----------------------------------------------------------------------
# Scored chunk — the output of the reranker
# ----------------------------------------------------------------------


@dataclass
class ScoredChunk:
    """A retrieved chunk annotated with its cross-encoder relevance score."""

    chunk_id: str
    chunk_text: str
    score: float
    metadata: dict
    # Original cosine similarity score from the bi-encoder retrieval stage.
    # Preserved here so the frontend can show both cosine and reranker scores
    # and users can see whether the reranker meaningfully re-ordered results.
    cosine_score: float = 0.0


# ----------------------------------------------------------------------
# CrossEncoderReranker
# ----------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Wrapper around a sentence-transformers CrossEncoder model.

    Usage::

        reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
        reranked = reranker.rerank(query="What is RAG?", chunks=retrieved_chunks, top_k=5)
    """

    def __init__(self, model_id: str) -> None:
        if model_id not in {m.id for m in AVAILABLE_RERANKERS}:
            raise ValueError(
                f"Unknown reranker model '{model_id}'. "
                f"Available: {[m.id for m in AVAILABLE_RERANKERS]}"
            )
        self.model_id = model_id
        self._encoder: "CrossEncoder | None" = None

    def _ensure_loaded(self) -> None:
        if self._encoder is None:
            self._encoder = _load_model(self.model_id)

    def rerank(
        self,
        query: str,
        chunks: list,
        top_k: int,
    ) -> list[ScoredChunk]:
        """
        Score (query, chunk) pairs with the cross-encoder and return the top-k
        most relevant chunks, sorted descending by relevance score.

        Parameters
        ----------
        query:
            The user question / search query.
        chunks:
            Iterable of objects with ``chunk_id``, ``chunk_text``, ``score``,
            and ``metadata`` attributes.  Accepts both ``SearchResult`` objects
            (from QdrantStore) and ``ScoredChunk`` objects.
        top_k:
            Return only the top-k most relevant chunks after reranking.

        Returns
        -------
        list[ScoredChunk]
            Top-k chunks sorted by cross-encoder relevance score (highest first).
        """
        if not chunks:
            return []

        t0 = _time.monotonic()
        self._ensure_loaded()

        # sentence-transformers CrossEncoder.predict() accepts a list of
        # (query, document) string pairs and returns a list of scores.
        pairs: list[tuple[str, str]] = []
        for chunk in chunks:
            pairs.append((query, chunk.chunk_text))

        raw_scores: list[float] = self._encoder.predict(pairs)  # type: ignore[assignment]
        # Normalize MS MARCO logits to [0, 1]; pass through BGE scores as-is.
        scores = _normalize_scores(raw_scores, self.model_id)

        scored_chunks = [
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.chunk_text,
                score=norm_score,
                # Preserve the original bi-encoder cosine score so the frontend
                # can show both cosine and reranker scores for transparency.
                cosine_score=getattr(chunk, "cosine_score", 0.0) or chunk.score,
                metadata=getattr(chunk, "metadata", {}),
            )
            for chunk, norm_score in zip(chunks, scores)
        ]

        # Sort descending by cross-encoder score (higher = more relevant)
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        result = scored_chunks[:top_k]

        elapsed = _time.monotonic() - t0
        _log.debug(
            "CrossEncoderReranker.rerank: scored %d chunks in %.3fs — "
            "top_score=%.4f bottom_score=%.4f",
            len(chunks), elapsed,
            result[0].score if result else 0.0,
            result[-1].score if result else 0.0,
        )

        return result


# ----------------------------------------------------------------------
# Helpers exposed for the API layer
# ----------------------------------------------------------------------


def list_available_rerankers() -> list[RerankerEntry]:
    """Return the full registry as a list (JSON-serialisable)."""
    return list(AVAILABLE_RERANKERS)
