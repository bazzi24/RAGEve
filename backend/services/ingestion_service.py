from __future__ import annotations

import asyncio
import gc
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from rag.chunking.adaptive import adaptive_chunk_text
from rag.chunking.high_accuracy import deepdoc_chunk_text
from rag.deepdoc.analyzer import analyze_text
from rag.deepdoc.layout_parser import layout_to_readable_text, parse_pdf_layout
from rag.deepdoc.quality_scorer import ChunkProfile, score_and_select_profile
from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.embedding.sparse_embedder import SparseEmbedder
from rag.ingestion.doc_converter import ConversionResult
from rag.ingestion.extractors import Extractors
from rag.storage.qdrant_store import ChunkRecord, QdrantStore
from rag.utils.memory import recommend_embed_batch_size

from backend.config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Memory guard: reject PDFs larger than this before attempting any parsing.
MAX_PDF_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB

_log = logging.getLogger("backend.services.ingestion_service")

ProgressCallback = Callable[[dict], Awaitable[None] | None]


def _clear_cuda_cache() -> None:
    """Clear PyTorch CUDA memory if available. No-op otherwise."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


async def _emit_progress(callback: ProgressCallback | None, event: dict) -> None:
    if callback is None:
        return
    maybe_awaitable = callback(event)
    if maybe_awaitable is not None:
        await maybe_awaitable


class IngestionService:
    """
    End-to-end ingestion: extract -> chunk -> dense embed (Ollama) ->
    sparse embed (fastembed) -> upsert both named vectors to Qdrant.

    Memory management
    -----------------
    Embedding + upserting is done in streaming fashion per batch (not all-at-once)
    to keep RAM and VRAM bounded regardless of file size:

        chunks → batch embed (yield) → build records → upsert → gc + CUDA clear → next batch

    A 2 MB PDF producing 5,000 chunks uses at most ~batch_size chunks'
    worth of embedding vectors in memory at any given time, rather than
    all 5,000 simultaneously.
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: OllamaEmbedder,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.chunk_size = settings.default_chunk_size
        self.chunk_overlap = settings.default_chunk_overlap
        self.max_tokens_per_chunk = settings.default_max_tokens_per_chunk

    async def ingest_file(
        self,
        file_path: Path,
        dataset_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_tokens_per_chunk: int | None = None,
        force_profile: ChunkProfile | None = None,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> dict:
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            _log.error("[%s] Unsupported file type: %s", dataset_id, ext)
            raise ValueError(f"Unsupported file: {ext}")

        t0 = time.monotonic()
        _log.info("[%s] Starting: %s → %s", dataset_id, file_path.name, dataset_id)

        await _emit_progress(
            progress_callback,
            {"stage": "extracting", "message": f"Extracting text from {file_path.name}", "progress": 5},
        )

        # ── 1. Extract text ─────────────────────────────────────────────────
        extraction_meta: dict = {}
        if ext == ".pdf":
            # Size guard — reject oversized PDFs before allocating huge buffers
            file_size = file_path.stat().st_size
            if file_size > MAX_PDF_SIZE_BYTES:
                raise ValueError(
                    f"PDF file is {file_size / (1024*1024):.1f} MB — exceeds the "
                    f"{MAX_PDF_SIZE_BYTES / (1024*1024):.0f} MB size limit."
                )

            # Extract raw text, then parse layout — both wrapped in a timeout
            try:
                raw_text = await asyncio.wait_for(
                    asyncio.to_thread(Extractors.from_pdf, file_path),
                    timeout=300.0,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"PDF extraction timed out after 300s — "
                    f"the file may be corrupted or extremely large."
                )

            extraction_meta = {"extractor": "pymupdf", "layout_aware": True}

            # Layout-aware conversion (can be slow for very large files)
            try:
                layouts = await asyncio.wait_for(
                    asyncio.to_thread(parse_pdf_layout, file_path),
                    timeout=600.0,
                )
                raw_text = layout_to_readable_text(layouts)
            except asyncio.TimeoutError:
                # Fall back to the plain text extraction if layout parsing times out
                _log.warning(
                    "[%s] PDF layout parsing timed out after 600s — using plain text",
                    dataset_id,
                )
        elif ext == ".doc":
            raw_text, conv_result = Extractors.from_doc(file_path)
            extraction_meta = {
                "extractor": "doc_converter",
                "converter": conv_result.converter.value,
                "converted": conv_result.success,
            }
        elif ext == ".docx":
            raw_text = Extractors.from_docx(file_path)
            extraction_meta = {"extractor": "python-docx"}
        elif ext == ".xlsx":
            raw_text = Extractors.from_xlsx(file_path)
            extraction_meta = {"extractor": "pandas"}
        elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            raw_text = Extractors.from_image(file_path)
            extraction_meta = {"extractor": "pytesseract-ocr"}
        else:
            raise ValueError(f"Unsupported: {ext}")

        _log.debug("[%s] Extracted %d chars", dataset_id, len(raw_text))

        # ── 2. Quality analysis → profile selection ────────────────────────
        await _emit_progress(
            progress_callback,
            {"stage": "analyzing", "message": "Scoring document quality", "progress": 18},
        )

        quality_report = score_and_select_profile(raw_text)
        signals = quality_report.signals
        selected_profile = force_profile or quality_report.profile.profile
        config = quality_report.profile

        _log.info(
            "[%s] Stage=analyzing — profile=%s quality_score=%.3f",
            dataset_id, selected_profile.value, quality_report.quality_score,
        )

        eff_chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        eff_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        eff_tokens = max_tokens_per_chunk if max_tokens_per_chunk is not None else config.max_tokens_per_chunk

        # ── 3. Adaptive chunking ────────────────────────────────────────────
        await _emit_progress(
            progress_callback,
            {"stage": "chunking", "message": "Splitting into semantic chunks", "progress": 30},
        )

        chunks = adaptive_chunk_text(
            raw_text,
            profile=selected_profile,
            override_size=eff_chunk_size,
            override_overlap=eff_overlap,
            override_tokens=eff_tokens,
        )

        chunk_count = len(chunks)
        _log.info(
            "[%s] Stage=chunking — %d chunks (size=%d overlap=%d)",
            dataset_id, chunk_count, eff_chunk_size, eff_overlap,
        )

        # raw_text is no longer needed — free it immediately so GC can reclaim
        # the large string before embedding starts.
        del raw_text
        gc.collect()
        _clear_cuda_cache()

        # ── 4. Embed + sparse + upsert per batch ──────────────────────────
        #
        # Instead of loading ALL embeddings into memory, we:
        #   1. Embed one batch → get list[float] vectors
        #   2. Sparse-embed that batch
        #   3. Build ChunkRecords for that batch
        #   4. Upsert to Qdrant
        #   5. Delete batch locals + gc.collect() + CUDA cache clear
        #
        # Peak memory = max(embed_batch_size × 768 × 8 bytes × 2 stages)
        # For batch_size=16 with float64 in Python: ~200 KB per batch.
        # Even with a 5,000-chunk PDF: peak ~25 MB instead of 120 MB all-at-once.

        await _emit_progress(
            progress_callback,
            {
                "stage": "embedding",
                "message": f"Embedding {chunk_count} chunks",
                "progress": 40,
                "chunks_done": 0,
                "chunks_total": chunk_count,
            },
        )

        _log.info("[%s] Stage=embedding — streaming %d chunks per batch", dataset_id, chunk_count)

        # Ensure collection exists before streaming upserts
        if not self.qdrant.collection_exists(dataset_id):
            self.qdrant.create_collection(dataset_id)

        total_upserted = 0
        # Adaptive batch size based on available VRAM (GPU) or safe CPU default.
        # recommend_embed_batch_size() reads actual CUDA free memory, so on GPU
        # this auto-scales to 8–128; on CPU it returns 32.
        embed_batch_size = recommend_embed_batch_size()

        async def on_embed_progress(done: int, total: int) -> None:
            if total <= 0:
                mapped = 40
            else:
                ratio = done / total
                mapped = int(40 + ratio * 50)  # 40–90%
            await _emit_progress(
                progress_callback,
                {
                    "stage": "embedding",
                    "message": f"Embedding chunks ({done}/{total})",
                    "progress": mapped,
                    "chunks_done": done,
                    "chunks_total": total,
                },
            )

        # ── 4a. Sparse embedder: pre-load once outside the batch loop ────
        #    (fastembed loads model on first call; subsequent calls reuse it)
        if self.sparse_embedder is not None:
            # Trigger lazy load before the streaming loop so it doesn't
            # happen inside the time-critical embedding batch loop.
            _ = self.sparse_embedder._ensure_loaded  # type: ignore[attr-defined]

        # Pre-compute per-chunk metadata list (only strings — tiny memory)
        # These keys are intentionally aligned with the HF ingest path so that
        # build_context() in rag_pipeline.py always finds the same fields:
        #   dataset_id, split, quality_score, source_file, profile
        base_metadata = {
            "source_file": file_path.name,
            "dataset_id": dataset_id,
            "extension": ext,
            # HF ingest path stores: split, quality_score, profile, text_columns_used,
            # total_chunks_in_row. We mirror all shared fields here.
            "split": "manual",
            "quality_score": quality_report.quality_score,
            "profile": selected_profile.value,
            "text_columns_used": None,  # N/A for single-file manual ingest; HF path uses a list
            # Embedding model tracking — stored so query-time can detect mismatches.
            "embedding_model": self.embedder.model,
            "sparse_model": self.sparse_embedder.model_id if self.sparse_embedder else None,
        }

        # Stream over batches: embed → sparse → upsert → cleanup → next
        async for batch_embs, batch_start, batch_end in self.embedder.embed_batches(
            chunks, batch_size=embed_batch_size, on_progress=on_embed_progress
        ):
            batch_chunks = chunks[batch_start:batch_end]
            batch_size_actual = len(batch_chunks)

            # ── Sparse embed this batch ──────────────────────────────────
            batch_sparse: list[dict[str, Any] | None] = [None] * batch_size_actual
            if self.sparse_embedder is not None:
                sparse_results = await asyncio.to_thread(
                    self.sparse_embedder.embed_batch, batch_chunks
                )
                for idx, sv in enumerate(sparse_results):
                    if sv.indices:
                        batch_sparse[idx] = {"indices": sv.indices, "values": sv.values}

            # ── Build ChunkRecords for this batch ─────────────────────────
            records = [
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    chunk_text=batch_chunks[idx],
                    metadata={
                        **base_metadata,
                        "chunk_index": batch_start + idx,
                    },
                    dense_vector=batch_embs[idx],
                    sparse_vector=batch_sparse[idx],
                )
                for idx in range(batch_size_actual)
            ]

            # ── Upsert ────────────────────────────────────────────────
            try:
                upserted = self.qdrant.upsert_chunks(
                    collection_name=dataset_id,
                    chunks=records,
                )
                total_upserted += upserted
            except Exception as exc:
                _log.error("[%s] Upsert failed at batch %d-%d: %s",
                           dataset_id, batch_start, batch_end, exc)
                raise

            # ── Memory release for this batch ───────────────────────────
            # Delete all large intermediate objects before the next batch.
            # `chunks` list itself is kept (referenced by next slice) but
            # the sliced strings for this batch become eligible for GC.
            del batch_embs, batch_sparse, batch_chunks, records, sparse_results
            gc.collect()
            _clear_cuda_cache()

        _log.info("[%s] Stage=upserting — %d points total upserted", dataset_id, total_upserted)

        # ── 5. All batches done ───────────────────────────────────────────────
        await _emit_progress(
            progress_callback,
            {"stage": "completed", "message": "Ingestion completed", "progress": 100,
             "chunks_done": chunk_count, "chunks_total": chunk_count},
        )

        # Final cleanup — the `chunks` list itself can now be freed
        del chunks
        gc.collect()
        _clear_cuda_cache()

        elapsed = time.monotonic() - t0
        _log.info(
            "[%s] Completed: %s — %d chunks in %.1fs",
            dataset_id, file_path.name, chunk_count, elapsed,
        )

        quality_dict = {
            "quality_score": quality_report.quality_score,
            "selected_profile": selected_profile.value,
            "profile_reason": config.reason,
            "signals": {
                "alpha_ratio": signals.alpha_ratio,
                "ocr_noise_ratio": signals.ocr_noise_ratio,
                "broken_line_ratio": signals.broken_line_ratio,
                "header_footer_ratio": signals.header_footer_ratio,
                "table_density": signals.table_density,
                "avg_sentence_length": signals.avg_sentence_length,
                "language_script_changes": signals.language_script_changes,
                "repeated_word_ratio": signals.repeated_word_ratio,
                "code_delimiter_ratio": signals.code_delimiter_ratio,
                "issue_tags": signals.issue_tags,
            },
        }

        return {
            "dataset_id": dataset_id,
            "filename": file_path.name,
            "extension": ext,
            "chars": 0,  # raw_text deleted; char count not needed post-extraction
            "chunks": chunk_count,
            "collection": dataset_id,
            "document_analysis": {},
            "sample_chunk_analysis": [],
            "quality_report": quality_dict,
            "layout_summary": None,
            "extraction": extraction_meta,
        }
