"""
Sparse embedding provider using fastembed.

fastembed is Qdrant's own embedding library — lightweight, no GPU required,
and ships with off-the-shelf sparse models (SpladePP, BM25).

Models available:
  - "Qdrant/仆-IPLP-sparse-uncased_L-6_H-384_ak"       — default, Splade-style
  - "Qdrant/naver-splade-v2-conventional_Embedding"  — Splade v2 conventional
  - "sentence-transformers/gtr-t5-base"               — not sparse (dense)

The default model (Splade++) produces sparse vectors where non-zero entries
represent weighted term importance (SPLADE style — a learned sparse encoding).
This is ideal for hybrid search because:
  1. It encodes lexical + semantic relevance in one sparse vector
  2. BM25-like exact keyword matching is preserved
  3. Term weights are soft-capped (ReLU → sigmoid), reducing noise vs raw TF-IDF

Returns {indices: list[int], values: list[float]} compatible with Qdrant's
SparseVector format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Model registry — add new sparse models here.
# ----------------------------------------------------------------------

# Default model — Splade++ (SPLADE++ v1), the best general-purpose sparse model in fastembed.
# Correct fastembed model IDs confirmed via SparseTextEmbedding.list_supported_models().
DEFAULT_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

# Available sparse models keyed by short name for config / UI.
AVAILABLE_SPARSE_MODELS: dict[str, dict[str, Any]] = {
    "splade++": {
        "id": "prithivida/Splade_PP_en_v1",
        "display_name": "Splade++ (recommended)",
        "description": (
            "SPLADE++ v1 — best general-purpose sparse model in fastembed. "
            "Encodes learned term importance weights. Strong on both exact keywords "
            "and semantic similarity. ~530 MB, CPU-compatible."
        ),
        "approx_size_mb": 530,
    },
    "bm42": {
        "id": "Qdrant/bm42-all-minilm-l6-v2-attentions",
        "display_name": "BM42 (lightweight)",
        "description": (
            "Lightweight sparse model based on MiniLM-L6. Fast inference, smaller "
            "memory footprint. Good for English text with moderate keyword needs. "
            "~90 MB."
        ),
        "approx_size_mb": 90,
    },
    "bm25": {
        "id": "Qdrant/bm25",
        "display_name": "BM25 (pure keyword)",
        "description": (
            "Pure BM25 — exact term frequency / inverse document frequency. "
            "No learned weights. Best for strict keyword matching with no semantic "
            "generalization. ~10 MB."
        ),
        "approx_size_mb": 10,
    },
}


# ----------------------------------------------------------------------
# SparseVector — the dict shape Qdrant expects
# ----------------------------------------------------------------------


@dataclass
class SparseVector:
    """
    A sparse vector as understood by Qdrant SparseVector.

    Sparse vectors store only non-zero dimensions:
      indices: list of integer positions
      values:  parallel list of float values at those positions

    This is identical to the format returned by fastembed's SparseEmbedding.
    """
    indices: list[int]
    values: list[float]


# ----------------------------------------------------------------------
# SparseEmbedder — lazy-loaded fastembed wrapper
# ----------------------------------------------------------------------


class SparseEmbedder:
    """
    Sparse embedding wrapper around fastembed.

    Lazily initialises the fastembed model on first call to avoid import
    overhead and model loading during app startup.

    Usage::

        embedder = SparseEmbedder()
        vec = embedder.embed("What is RAG?")
        # vec.indices  = [312, 4891, 1204, ...]
        # vec.values   = [0.72, 0.55, 0.43, ...]
    """

    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_id = model or DEFAULT_SPARSE_MODEL
        self.cache_dir = cache_dir
        self._encoder: Any = None  # set by _ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._encoder is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is required for sparse embeddings. "
                    "Install it with: uv pip install fastembed"
                ) from exc

            _log.info("Loading sparse model '%s' (first call — may take ~10-30s)…", self.model_id)
            self._encoder = SparseTextEmbedding(
                model_name=self.model_id,
                cache_dir=self.cache_dir,
                lazy_load=True,
                max_workers=4,
            )
            _log.info("Sparse model '%s' loaded.", self.model_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> SparseVector:
        """
        Encode a single text into a sparse vector.

        Parameters
        ----------
        text:
            The text to encode.

        Returns
        -------
        SparseVector
            indices and values arrays ready for Qdrant SparseVector storage.
        """
        self._ensure_loaded()
        results: list[Any] = list(self._encoder.query_embed([text]))
        raw = results[0]

        # fastembed returns numpy arrays; convert to Python lists for Qdrant
        indices: list[int] = [int(i) for i in raw.indices]  # type: ignore[attr-defined]
        values: list[float] = [float(v) for v in raw.values]  # type: ignore[attr-defined]

        return SparseVector(indices=indices, values=values)

    def embed_batch(self, texts: list[str]) -> list[SparseVector]:
        """
        Encode multiple texts into sparse vectors in one batch call.

        Parameters
        ----------
        texts:
            List of texts to encode.

        Returns
        -------
        list[SparseVector]
            One SparseVector per input text.
        """
        self._ensure_loaded()
        results: list[Any] = list(self._encoder.query_embed(texts))
        return [
            SparseVector(
                indices=[int(i) for i in r.indices],  # numpy.int → Python int
                values=[float(v) for v in r.values],   # numpy.float → Python float
            )  # type: ignore[attr-defined]
            for r in results
        ]

    @property
    def is_loaded(self) -> bool:
        """True once the model has been loaded (after first embed call)."""
        return self._encoder is not None


# ----------------------------------------------------------------------
# Convenience helpers — used by ingestion and rag_pipeline
# ----------------------------------------------------------------------


def get_sparse_embedding(text: str, model: str | None = None) -> SparseVector:
    """
    Standalone helper: embed a single text with the default sparse model.

    Creates a temporary embedder internally. For batch workloads, create a
    SparseEmbedder instance and call embed_batch() directly.

    Parameters
    ----------
    text:
        The text to encode.

    model:
        Override the sparse model ID (uses default if None).

    Returns
    -------
    SparseVector
        The sparse encoding of the input text.
    """
    embedder = SparseEmbedder(model=model)
    return embedder.embed(text)


def get_sparse_batch(texts: list[str], model: str | None = None) -> list[SparseVector]:
    """
    Embed a batch of texts using the default sparse model.

    Parameters
    ----------
    texts:
        List of texts to encode.

    model:
        Override the sparse model ID (uses default if None).

    Returns
    -------
    list[SparseVector]
        One SparseVector per input text.
    """
    embedder = SparseEmbedder(model=model)
    return embedder.embed_batch(texts)
