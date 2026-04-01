"""
Hybrid Retrieval + RRF RAG Pipeline.

Retrieval strategy:
  1. Compute query embedding in both dense (Ollama) and sparse (fastembed) spaces.
  2. Use Qdrant's prefetch + RRF to fuse dense and sparse rankings in one call.
  3. Optionally refine results through a cross-encoder reranker (Phase 7).

RRF (Reciprocal Rank Fusion):
  Score_A(i) = Σ  1 / (k + rank_i)
  Fused(i)    = Score_A(i) + Score_B(i)

Where k=60 is the standard RRF constant. Higher k reduces the "synergy bonus"
for appearing in both lists. Qdrant's native RRF prefetch implements this internally.

Why hybrid matters:
  - Dense (Ollama): Captures semantic similarity, handles synonyms, paraphrases.
    Weaker on exact keywords and technical terms.
  - Sparse (fastembed/Splade++): Captures exact keyword matches and term frequency
    signals. Strong on technical queries (e.g. "TypeError: cannot read property 'x'")
    and domain-specific terminology (e.g. "mitochondrial matrix").
  - RRF fusion: Chunks ranked highly in BOTH spaces bubble up; chunks that are
    top-heavy in only one space are preserved. Result: robust across the full
    spectrum from "what is RAG?" (semantic) to "TypeError 'x' is not defined"
    (keyword-heavy).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.embedding.sparse_embedder import SparseEmbedder
from rag.generation.ollama_chat import ChatMessage, OllamaChat
from rag.llm.context_builder import build_context
from rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag.storage.qdrant_store import SearchResult
from rag.storage.qdrant_store import QdrantStore

_log = logging.getLogger(__name__)

# How many extra chunks to retrieve before reranking.
# top_k=5 with OVERFETCH_MULTIPLIER=3 → retrieve 15, rerank down to 5.
OVERFETCH_MULTIPLIER = 3

# RRF k constant — passed to Qdrant's native RRF prefetch.
# Standard values: 60 (Qdrant default). Range [1, 1000].
RRF_K = 60


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class SourceChunk:
    chunk_id: str
    text: str
    score: float
    source: str | None
    # Cross-encoder reranker score ∈ [0, 1] (0.0 when reranking is disabled).
    # When reranking is enabled, `score` = normalized reranker score.
    cosine_score: float = 0.0
    # Sparse retrieval score ∈ [0, 1] (0.0 when dense-only).
    sparse_score: float = 0.0
    # Search mode used to retrieve this chunk.
    # "hybrid" | "dense" | "sparse"
    search_type: str = "dense"


@dataclass
class RAGAnswer:
    answer: str
    sources: list[SourceChunk]
    metadata: dict[str, Any]


SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. "
    "If the context does not contain enough information to answer the question, say so clearly. "
    "Always cite which source(s) you used when providing specific information.\n\n"
    "## Context\n{context}\n\n"
    "## Instruction\n{question}"
)


# ----------------------------------------------------------------------
# Helpers
def _results_to_sources(
    chunks: list[SearchResult],
    search_type: str,
) -> list[SourceChunk]:
    """
    Convert SearchResult list to SourceChunk list for the SSE / response.

    cosine_score and sparse_score are taken from the SearchResult.score field
    when dedicated attributes are not available. This handles both pre-reranking
    (rich scores from Qdrant) and post-reranking (cross-encoder normalised score)
    cases correctly.
    """
    def _to_source(c: SearchResult) -> SourceChunk:
        # Prefer named dense/sparse attributes when present; fall back to score.
        dense   = getattr(c, "dense_score", None) or c.score
        sparse  = getattr(c, "sparse_score", None) or 0.0
        return SourceChunk(
            chunk_id=c.chunk_id,
            text=c.chunk_text,
            score=c.score,
            source=c.metadata.get("source_file"),
            cosine_score=dense,
            sparse_score=sparse,
            search_type=search_type,
        )
    return [_to_source(c) for c in chunks]


# ----------------------------------------------------------------------
# RAG Pipeline
# ----------------------------------------------------------------------


class RAGPipeline:
    """
    Full RAG pipeline with optional hybrid sparse+dense retrieval and
    cross-encoder reranking.

    Usage::

        pipeline = RAGPipeline(
            qdrant_store=qdrant_store,
            embedder=ollama_embedder,       # dense — Ollama
            sparse_embedder=sparse_embedder, # sparse — fastembed
            chat_model=ollama_chat,
        )

        # Hybrid search with reranking
        result = await pipeline.query_stream(
            collection_name="my-dataset",
            question="What is retrieval-augmented generation?",
            use_hybrid=True,
            use_reranker=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: OllamaEmbedder,
        chat_model: OllamaChat,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.chat = chat_model

    # ------------------------------------------------------------------
    # Model mismatch detection
    # ------------------------------------------------------------------

    def _check_model_mismatch(self, first_result_metadata: dict | None) -> None:
        """Log a warning if the query embedder differs from the ingest embedder."""
        if not first_result_metadata:
            return
        stored = first_result_metadata.get("embedding_model")
        if stored and stored != self.embedder.model:
            _log.warning(
                "Embedding model mismatch: this query uses '%s' but chunks were "
                "ingested with '%s'. Results may be degraded or incorrect "
                "(dimension mismatch or semantic incoherence).",
                self.embedder.model,
                stored,
            )

    # ------------------------------------------------------------------
    # Query (non-streaming)
    # ------------------------------------------------------------------

    async def query(
        self,
        collection_name: str,
        question: str,
        *,
        system_prompt: str | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        temperature: float = 0.7,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        reranker_model: str | None = None,
    ) -> RAGAnswer:
        chunks = await self._retrieve(
            collection_name=collection_name,
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
            use_hybrid=use_hybrid,
        )

        # Warn if the ingest embedder differs from the current query embedder
        self._check_model_mismatch(chunks[0].metadata if chunks else None)

        # ── Optional reranking ─────────────────────────────────────────
        if use_reranker and reranker_model:
            reranker = CrossEncoderReranker(model_id=reranker_model)
            reranked = reranker.rerank(query=question, chunks=chunks, top_k=top_k)
            _log.info(
                "[%s] Reranking: %d → %d chunks using %s",
                collection_name, len(chunks), top_k, reranker_model,
            )
            context_chunks = reranked
        else:
            context_chunks = chunks[:top_k]

        # ── Build context ────────────────────────────────────────────────
        context = build_context(context_chunks)

        if system_prompt:
            augmented_system = f"{system_prompt}\n\n## Context\n{context}"
        else:
            augmented_system = SYSTEM_PROMPT_TEMPLATE.format(
                context=context, question=question
            )

        # ── Generate ─────────────────────────────────────────────────────
        messages = [ChatMessage(role="user", content=question)]
        response = await self.chat.chat(
            messages=messages,
            system_prompt=augmented_system,
            temperature=temperature,
        )

        # ── Package sources ──────────────────────────────────────────────
        sources = _results_to_sources(
            context_chunks,
            search_type="hybrid" if use_hybrid else "dense",
        )

        return RAGAnswer(
            answer=response.message.content,
            sources=sources,
            metadata={
                "chunks_retrieved": len(chunks),
                "chunks_reranked": len(context_chunks) if use_reranker and reranker_model else None,
                "reranker_model": reranker_model if use_reranker else None,
                "use_hybrid": use_hybrid,
                "collection": collection_name,
            },
        )

    # ------------------------------------------------------------------
    # Query (streaming)
    # ------------------------------------------------------------------

    async def query_stream(
        self,
        collection_name: str,
        question: str,
        *,
        system_prompt: str | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        temperature: float = 0.7,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        reranker_model: str | None = None,
    ):
        chunks = await self._retrieve(
            collection_name=collection_name,
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
            use_hybrid=use_hybrid,
        )

        # Warn if the ingest embedder differs from the current query embedder
        self._check_model_mismatch(chunks[0].metadata if chunks else None)

        # ── Optional reranking ─────────────────────────────────────────
        if use_reranker and reranker_model:
            reranker = CrossEncoderReranker(model_id=reranker_model)
            reranked = reranker.rerank(query=question, chunks=chunks, top_k=top_k)
            _log.info(
                "[%s] Reranking: %d → %d chunks using %s",
                collection_name, len(chunks), top_k, reranker_model,
            )
            context_chunks = reranked
        else:
            context_chunks = chunks[:top_k]

        # ── Build context ────────────────────────────────────────────────
        context = build_context(context_chunks)

        if system_prompt:
            augmented_system = f"{system_prompt}\n\n## Context\n{context}"
        else:
            augmented_system = SYSTEM_PROMPT_TEMPLATE.format(
                context=context, question=question
            )

        # ── Stream answer ────────────────────────────────────────────────
        messages = [ChatMessage(role="user", content=question)]
        full_response: list[str] = []
        t0 = time.monotonic()

        async for delta in self.chat.chat_stream(
            messages=messages,
            system_prompt=augmented_system,
            temperature=temperature,
        ):
            full_response.append(delta)
            yield delta

        elapsed = time.monotonic() - t0
        response_len = len("".join(full_response))

        _log.info(
            "[%s] Generation complete: model=%s temp=%.1f top_k=%d "
            "chunks=%d chars=%d elapsed=%.1fs",
            collection_name,
            self.chat.model,
            temperature,
            top_k,
            len(context_chunks),
            response_len,
            elapsed,
        )

        sources = _results_to_sources(
            context_chunks,
            search_type="hybrid" if use_hybrid else "dense",
        )

        yield {
            "sources": [s.__dict__ for s in sources],
            "done": True,
            "reranker_model": reranker_model if use_reranker else None,
            "use_hybrid": use_hybrid,
        }

    # ------------------------------------------------------------------
    # Core retrieval — private
    # ------------------------------------------------------------------

    async def _retrieve(
        self,
        collection_name: str,
        question: str,
        top_k: int,
        score_threshold: float,
        use_hybrid: bool,
    ) -> list[SearchResult]:
        """
        Retrieve chunks using either hybrid or dense-only search.

        Hybrid path:
          1. Dense embed (Ollama)
          2. Sparse embed (fastembed)
          3. Qdrant hybrid_search() → RRF fusion internally
          4. Return top_k × OVERFETCH_MULTIPLIER for reranking

        Dense-only path:
          1. Dense embed (Ollama)
          2. Qdrant dense_search()
          3. Return top_k × OVERFETCH_MULTIPLIER for reranking
        """
        overfetch_k = top_k * OVERFETCH_MULTIPLIER

        if use_hybrid and self.sparse_embedder is not None:
            # ── Hybrid retrieval ────────────────────────────────────────
            _log.info(
                "[%s] Retrieval: hybrid (dense+sparse) top_k=%d → %d chunks",
                collection_name, top_k, overfetch_k,
            )

            dense_vec, sparse_vec = await self._embed_both(question)

            chunks = await self.qdrant.hybrid_search(
                collection_name=collection_name,
                dense_query=dense_vec,
                sparse_query={"indices": sparse_vec.indices, "values": sparse_vec.values},
                top_k=overfetch_k,
                rrf_k=RRF_K,
            )
            _log.info(
                "[%s] Hybrid search: %d results, max_score=%.4f",
                collection_name, len(chunks), chunks[0].score if chunks else 0.0,
            )

        else:
            # ── Dense-only retrieval (backward-compatible) ───────────────
            _log.info(
                "[%s] Retrieval: dense-only top_k=%d → %d chunks",
                collection_name, top_k, overfetch_k,
            )
            dense_vec = await self.embedder.embed_single(question)

            chunks = await self.qdrant.dense_search(
                collection_name=collection_name,
                query_vector=dense_vec,
                top_k=overfetch_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
            )
            _log.info(
                "[%s] Dense search: %d results, max_score=%.4f",
                collection_name, len(chunks), chunks[0].score if chunks else 0.0,
            )

        return chunks

    async def _embed_both(self, text: str):
        """
        Compute both dense and sparse embeddings for a single text.

        Returns (dense_vector, sparse_vector) where sparse_vector is a
        SparseVector dataclass from sparse_embedder.
        """
        import asyncio

        dense_task = self.embedder.embed_single(text)
        sparse_task = asyncio.to_thread(self.sparse_embedder.embed, text)

        dense_vec, sparse_vec = await asyncio.gather(dense_task, sparse_task)
        return dense_vec, sparse_vec
