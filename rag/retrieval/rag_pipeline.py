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

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.embedding.sparse_embedder import SparseEmbedder
from rag.generation.ollama_chat import ChatMessage, OllamaChat
from rag.llm.context_builder import build_context
from rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag.storage.qdrant_store import QdrantStore, SearchResult
from backend.services.cache_service import get_cache_service

try:
    from backend.config_loader import get_settings
    from backend.utils.log_sanitizer import sanitize_key
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False

_log = logging.getLogger(__name__)

# Get configurable settings
if HAS_CONFIG_LOADER:
    try:
        _settings = get_settings()
        OVERFETCH_MULTIPLIER = getattr(_settings, "overfetch_multiplier", 3)
        RERANKER_CACHE_SIZE = getattr(_settings, "reranker_cache_size", 10)
    except Exception:
        OVERFETCH_MULTIPLIER = 3
        RERANKER_CACHE_SIZE = 10
else:
    OVERFETCH_MULTIPLIER = 3
    RERANKER_CACHE_SIZE = 10

# RRF k constant — passed to Qdrant's native RRF prefetch.
# Standard values: 60 (Qdrant default). Range [1, 1000].
RRF_K = 60

# Only include PDF block metadata for chunks with score >= this threshold.
# Low-score chunks are likely irrelevant; showing full PDF preview is noise.
# Thresholds differ by search type because score scales differ:
#   - dense: cosine similarity ∈ [0,1] → threshold 0.75
#   - hybrid: RRF sum ∈ [0, ~0.033] → threshold 0.02 (≈60% of max)
PREVIEW_BLOCK_SCORE_THRESHOLD = {
    "hybrid": 0.02,
    "dense": 0.75,
    "sparse": 0.02,  # treat like hybrid
}


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
    # PDF layout metadata (if available): list of page numbers this chunk spans
    pages: list[int] | None = None
    # Detailed block-level bounding boxes: list of {page, bbox: {x0,y0,x1,y1}, type}
    blocks: list[dict] | None = None
    # Dataset ID (for constructing file URLs)
    datasetId: str | None = None


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

# Mandatory Markdown formatting rules appended to every system prompt.
# These ensure clean, readable output in the UI regardless of LLM defaults.
MARKDOWN_FORMATTING_RULES = (
    "\n\n"
    "CRITICAL FORMATTING RULES — STRICTLY ENFORCED:\n"
    "1. NEVER output a 'wall of text.' Always use paragraph breaks (blank lines) between distinct ideas.\n"
    "2. ALWAYS put a blank line BEFORE any list (numbered or bulleted) when it follows a sentence.\n"
    "   Example: 'Here are the benefits:\\n\\n1. First item\\n2. Second item'\n"
    "3. ALWAYS include a single space after the list marker: '1. Item' not '1.Item' or '1  Item'.\n"
    "4. For headers (## or ###), ensure exactly one blank line before and after.\n"
    "5. Keep paragraphs to 3-5 sentences maximum. Split long paragraphs into multiple shorter ones.\n"
    "\n"
    "Failure to follow these rules will degrade user experience significantly."
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

    Blocks (PDF highlighting) are only included for chunks with score >= PREVIEW_BLOCK_SCORE_THRESHOLD
    to avoid showing irrelevant previews.
    """

    def _to_source(c: SearchResult) -> SourceChunk:
        # Prefer named dense/sparse attributes when present; fall back to score.
        dense = getattr(c, "dense_score", None) or c.score
        sparse = getattr(c, "sparse_score", None) or 0.0
        blocks = c.metadata.get("blocks")
        # Only include block-level metadata for high-scoring chunks.
        if c.score < PREVIEW_BLOCK_SCORE_THRESHOLD[search_type]:
            blocks = None
        return SourceChunk(
            chunk_id=c.chunk_id,
            text=c.chunk_text,
            score=c.score,
            source=c.metadata.get("source_file"),
            cosine_score=dense,
            sparse_score=sparse,
            search_type=search_type,
            pages=c.metadata.get("pages"),
            blocks=blocks,
            datasetId=c.metadata.get("dataset_id"),
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

    # Class-level reranker cache with LRU eviction
    _reranker_cache: dict[str, CrossEncoderReranker] = {}
    _reranker_cache_order: list[str] = []

    # Performance metrics
    _metrics = {
        "reranker_cache_hits": 0,
        "reranker_cache_misses": 0,
        "reranker_load_times": [],  # in seconds
        "reranker_reuse_times": [],  # cache hit retrieval time
    }

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
        self.cache = get_cache_service()
        self._overfetch_multiplier = OVERFETCH_MULTIPLIER
        self._reranker_cache_size = RERANKER_CACHE_SIZE

    # ------------------------------------------------------------------
    # Reranker caching with LRU eviction
    # ------------------------------------------------------------------

    def _get_reranker(self, model_id: str) -> CrossEncoderReranker:
        """
        Get a reranker instance from cache or create a new one.

        Implements LRU eviction when cache exceeds configured size.
        """
        t0 = time.monotonic()

        # Check cache first
        if model_id in self._reranker_cache:
            self._metrics["reranker_cache_hits"] += 1
            self._metrics["reranker_reuse_times"].append(time.monotonic() - t0)
            # Move to end of order list (most recently used)
            if model_id in self._reranker_cache_order:
                self._reranker_cache_order.remove(model_id)
            self._reranker_cache_order.append(model_id)
            _log.debug(
                "[reranker] Cache HIT for model '%s' (load time: %.3fs from cache)",
                model_id,
                time.monotonic() - t0,
            )
            return self._reranker_cache[model_id]

        # Cache miss - create new reranker
        self._metrics["reranker_cache_misses"] += 1
        reranker = CrossEncoderReranker(model_id=model_id)

        # Add to cache with LRU eviction
        cache = self._reranker_cache
        order = self._reranker_cache_order
        cache_size = self._reranker_cache_size

        if len(cache) >= cache_size:
            # Evict least recently used (first in order list)
            lru_model = order.pop(0)
            del cache[lru_model]
            _log.debug("[reranker] Evicted LRU model: %s", lru_model)

        cache[model_id] = reranker
        order.append(model_id)

        load_time = time.monotonic() - t0
        self._metrics["reranker_load_times"].append(load_time)
        _log.info(
            "[reranker] Cache MISS - created and cached reranker '%s' (%.3fs)",
            model_id,
            load_time,
        )
        return reranker

    def get_reranker_metrics(self) -> dict[str, Any]:
        """Return current reranker cache metrics."""
        total_requests = (
            self._metrics["reranker_cache_hits"] + self._metrics["reranker_cache_misses"]
        )
        hit_rate = (
            self._metrics["reranker_cache_hits"] / total_requests
            if total_requests > 0
            else 0.0
        )
        avg_load_time = (
            sum(self._metrics["reranker_load_times"])
            / len(self._metrics["reranker_load_times"])
            if self._metrics["reranker_load_times"]
            else 0.0
        )
        avg_reuse_time = (
            sum(self._metrics["reranker_reuse_times"])
            / len(self._metrics["reranker_reuse_times"])
            if self._metrics["reranker_reuse_times"]
            else 0.0
        )
        return {
            "cache_size": len(self._reranker_cache),
            "cache_capacity": self._reranker_cache_size,
            "cached_models": list(self._reranker_cache.keys()),
            "hit_count": self._metrics["reranker_cache_hits"],
            "miss_count": self._metrics["reranker_cache_misses"],
            "hit_rate": round(hit_rate, 4),
            "avg_load_time_ms": round(avg_load_time * 1000, 2),
            "avg_reuse_time_ms": round(avg_reuse_time * 1000, 2),
            "total_requests": total_requests,
        }

    def clear_reranker_cache(self) -> None:
        """Clear the reranker cache (useful for testing or memory pressure)."""
        self._reranker_cache.clear()
        self._reranker_cache_order.clear()
        _log.info("[reranker] Cache cleared")

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
        # Check cache first for complete answer
        cached_answer = await self.cache.get_answer(
            collection=collection_name,
            chat_model=self.chat.model,
            system_prompt=system_prompt or "",
            question=question,
            temperature=temperature,
            top_k=top_k,
            use_reranker=use_reranker,
            use_hybrid=use_hybrid,
        )
        if cached_answer:
            _log.info(
                "[%s] Answer cache HIT for question (hybrid=%s)",
                collection_name,
                use_hybrid,
            )
            # Reconstruct RAGAnswer from cached dict
            sources = [
                SourceChunk(**s) for s in cached_answer.get("sources", [])
            ]
            return RAGAnswer(
                answer=cached_answer["answer"],
                sources=sources,
                metadata=cached_answer["metadata"],
            )

        _log.info(
            "[%s] Answer cache MISS for question (hybrid=%s)",
            collection_name,
            use_hybrid,
        )

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
            reranker = self._get_reranker(reranker_model)
            reranked = reranker.rerank(query=question, chunks=chunks, top_k=top_k)
            _log.info(
                "[%s] Reranking: %d → %d chunks using %s",
                collection_name,
                len(chunks),
                top_k,
                reranker_model,
            )
            context_chunks = [
                SearchResult(
                    chunk_id=rc.chunk_id,
                    chunk_text=rc.chunk_text,
                    score=rc.score,
                    metadata=rc.metadata,
                    dense_score=rc.cosine_score,
                    sparse_score=rc.sparse_score,
                )
                for rc in reranked
            ]
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

        # Append mandatory formatting rules
        augmented_system = f"{augmented_system}{MARKDOWN_FORMATTING_RULES}"

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

        answer = RAGAnswer(
            answer=response.message.content,
            sources=sources,
            metadata={
                "chunks_retrieved": len(chunks),
                "chunks_reranked": (
                    len(context_chunks) if use_reranker and reranker_model else None
                ),
                "reranker_model": reranker_model if use_reranker else None,
                "use_hybrid": use_hybrid,
                "collection": collection_name,
            },
        )

        # Cache the answer
        try:
            await self.cache.set_answer(
                collection=collection_name,
                chat_model=self.chat.model,
                system_prompt=system_prompt or "",
                question=question,
                temperature=temperature,
                top_k=top_k,
                use_reranker=use_reranker,
                use_hybrid=use_hybrid,
                answer={
                    "answer": answer.answer,
                    "sources": [s.__dict__ for s in answer.sources],
                    "metadata": answer.metadata,
                },
            )
            _log.debug(
                "[%s] Cached answer for question (hybrid=%s)",
                collection_name,
                use_hybrid,
            )
        except Exception as e:
            _log.warning("Failed to cache answer: %s", e)

        # Log reranker cache metrics periodically
        if use_reranker and (self._metrics["reranker_cache_hits"] + self._metrics["reranker_cache_misses"]) % 10 == 0:
            metrics = self.get_reranker_metrics()
            _log.info(
                "[reranker] Cache metrics: hits=%d, misses=%d, hit_rate=%.2f%%, size=%d/%d",
                metrics["hit_count"],
                metrics["miss_count"],
                metrics["hit_rate"] * 100,
                metrics["cache_size"],
                metrics["cache_capacity"],
            )

        return answer

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
            reranker = self._get_reranker(reranker_model)
            reranked = reranker.rerank(query=question, chunks=chunks, top_k=top_k)
            _log.info(
                "[%s] Reranking: %d → %d chunks using %s",
                collection_name,
                len(chunks),
                top_k,
                reranker_model,
            )
            # Convert ScoredChunk (from reranker) back to SearchResult for downstream processing.
            # This preserves dense_score (bi-encoder cosine) and sparse_score for UI display.
            context_chunks = [
                SearchResult(
                    chunk_id=rc.chunk_id,
                    chunk_text=rc.chunk_text,
                    score=rc.score,
                    metadata=rc.metadata,
                    dense_score=rc.cosine_score,
                    sparse_score=rc.sparse_score,
                )
                for rc in reranked
            ]
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

        # Append mandatory formatting rules to ensure clean Markdown in UI
        augmented_system = f"{augmented_system}{MARKDOWN_FORMATTING_RULES}"

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

        # Log reranker cache metrics periodically
        if use_reranker and (self._metrics["reranker_cache_hits"] + self._metrics["reranker_cache_misses"]) % 10 == 0:
            metrics = self.get_reranker_metrics()
            _log.info(
                "[reranker] Cache metrics: hits=%d, misses=%d, hit_rate=%.2f%%, size=%d/%d",
                metrics["hit_count"],
                metrics["miss_count"],
                metrics["hit_rate"] * 100,
                metrics["cache_size"],
                metrics["cache_capacity"],
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


    async def _embed_single_cached(self, text: str) -> list[float]:
        """Get embedding from cache or compute and cache it."""
        cached = await self.cache.get_embedding(self.embedder.model, text)
        if cached is not None:
            _log.debug(
                "[%s] Embedding cache HIT",
                self.embedder.model,
            )
            return cached

        embedding = await self.embedder.embed_single(text)
        await self.cache.set_embedding(self.embedder.model, text, embedding)
        _log.debug(
            "[%s] Embedding cached (len=%d)",
            self.embedder.model,
            len(embedding),
        )
        return embedding

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
        """
        overfetch_k = top_k * self._overfetch_multiplier
        query_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]

        # Check cache first (before computing embeddings)
        cached_chunks = await self.cache.get_search_results(
            collection=collection_name,
            embedding_model=self.embedder.model,
            top_k=overfetch_k,
            score_threshold=score_threshold,
            use_hybrid=use_hybrid,
            query_hash=query_hash,
        )
        if cached_chunks:
            _log.info(
                "[%s] Cache HIT for search results (hybrid=%s)",
                collection_name,
                use_hybrid,
            )
            return [
                SearchResult(
                    chunk_id=c["chunk_id"],
                    chunk_text=c["chunk_text"],
                    score=c["score"],
                    metadata=c["metadata"],
                    dense_score=c.get("dense_score", 0.0),
                    sparse_score=c.get("sparse_score", 0.0),
                )
                for c in cached_chunks
            ]

        # Cache miss - compute embeddings and search
        if use_hybrid and self.sparse_embedder is not None:
            _log.info(
                "[%s] Retrieval: hybrid (dense+sparse) top_k=%d → %d chunks [CACHE MISS]",
                sanitize_key(collection_name),
                top_k,
                overfetch_k,
            )
            dense_vec = await self._embed_single_cached(question)
            import asyncio
            sparse_vec = await asyncio.to_thread(self.sparse_embedder.embed, question)

            chunks = await self.qdrant.hybrid_search(
                collection_name=collection_name,
                dense_query=dense_vec,
                sparse_query={
                    "indices": sparse_vec.indices,
                    "values": sparse_vec.values,
                },
                top_k=overfetch_k,
                rrf_k=RRF_K,
            )
            _log.info(
                "[%s] Hybrid search: %d results, max_score=%.4f",
                sanitize_key(collection_name),
                len(chunks),
                chunks[0].score if chunks else 0.0,
            )

        else:
            _log.info(
                "[%s] Retrieval: dense-only top_k=%d → %d chunks [CACHE MISS]",
                sanitize_key(collection_name),
                top_k,
                overfetch_k,
            )
            dense_vec = await self._embed_single_cached(question)

            chunks = await self.qdrant.dense_search(
                collection_name=collection_name,
                query_vector=dense_vec,
                top_k=overfetch_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
            )
            _log.info(
                "[%s] Dense search: %d results, max_score=%.4f",
                sanitize_key(collection_name),
                len(chunks),
                chunks[0].score if chunks else 0.0,
            )

        # Cache the search results
        if chunks:
            try:
                cacheable = [
                    {
                        "chunk_id": c.chunk_id,
                        "chunk_text": c.chunk_text,
                        "score": c.score,
                        "metadata": c.metadata,
                        "dense_score": getattr(c, "dense_score", 0.0),
                        "sparse_score": getattr(c, "sparse_score", 0.0),
                    }
                    for c in chunks
                ]
                await self.cache.set_search_results(
                    collection=collection_name,
                    embedding_model=self.embedder.model,
                    top_k=overfetch_k,
                    score_threshold=score_threshold,
                    use_hybrid=use_hybrid,
                    query_hash=query_hash,
                    results=cacheable,
                )
                _log.debug(
                    "[%s] Cached %d search results (hybrid=%s)",
                    sanitize_key(collection_name),
                    len(chunks),
                    use_hybrid,
                )
            except Exception as e:
                _log.warning("Failed to cache search results: %s", e)

        return chunks

    async def _embed_both(self, text: str):
        """
        Compute both dense and sparse embeddings for a single text.

        Returns (dense_vector, sparse_vector) where sparse_vector is a
        SparseVector dataclass from sparse_embedder.
        """
        import asyncio

        if self.sparse_embedder is None:
            raise RuntimeError("Sparse embedder is not configured for hybrid search")

        dense_task = self.embedder.embed_single(text)
        sparse_task = asyncio.to_thread(self.sparse_embedder.embed, text)

        dense_vec, sparse_vec = await asyncio.gather(dense_task, sparse_task)
        return dense_vec, sparse_vec
