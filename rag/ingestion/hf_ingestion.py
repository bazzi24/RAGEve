from __future__ import annotations

import asyncio as _asyncio
import gc
import json
import logging
import uuid as uuid_lib
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

from rag.chunking.adaptive import adaptive_chunk_text
from rag.deepdoc.quality_scorer import score_and_select_profile
from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.embedding.sparse_embedder import SparseEmbedder
from rag.storage.qdrant_store import ChunkRecord, QdrantStore

logger = logging.getLogger(__name__)

# Callback type for progress reporting during ingestion
HFProgressCallback = Callable[[dict], Awaitable[None] | None]

# Supported HF dataset file formats
SUPPORTED_FORMATS = {".parquet", ".json", ".jsonl", ".csv"}

# Default text-like column names to auto-detect if text_column is not specified
TEXT_COLUMN_CANDIDATES = [
    "text", "content", "document", "passage", "body", "article",
    "context", "story", "summary", "description",
    # Common HF dataset column names
    "review", "sentence", "query", "input", "output",
]


def _detect_text_column(columns: list[str]) -> str | None:
    """Pick the best text column from a list of column names."""
    for candidate in TEXT_COLUMN_CANDIDATES:
        for col in columns:
            if col.strip().lower() == candidate:
                return col
    return None


def _load_hf_dataset(local_path: Path, *, row_limit: int | None = None) -> pd.DataFrame:
    """
    Load a HuggingFace-format dataset from a local directory.
    Tries Parquet first (most common), then JSON/JSONL, then CSV.

    When row_limit is set, uses PyArrow for efficient row skipping on parquet files
    (avoids loading the entire dataset when only a small sample is needed).
    """
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {local_path}")

    all_files: list[Path] = []
    for ext in SUPPORTED_FORMATS:
        all_files.extend(local_path.rglob(f"*{ext}"))

    if not all_files:
        raise ValueError(
            f"No supported files (.parquet, .json, .jsonl, .csv) found in {local_path}"
        )

    all_files = sorted(all_files, key=lambda f: f.name)

    dfs: list[pd.DataFrame] = []
    rows_loaded = 0

    for file_path in all_files:
        ext = file_path.suffix.lower()
        try:
            file_df: pd.DataFrame | None = None

            if ext == ".parquet" and row_limit is not None:
                import pyarrow.parquet as pq

                with pq.ParquetFile(file_path) as pf:
                    total_rows_in_file = pf.metadata.num_rows
                    rows_to_read = min(row_limit - rows_loaded, total_rows_in_file)
                    if rows_to_read <= 0:
                        break

                    if rows_to_read >= total_rows_in_file:
                        file_df = pd.read_parquet(file_path)
                    else:
                        # Accumulate chunks from each row group into a temporary list,
                        # then append a single concatenated DataFrame — avoids appending
                        # the same object reference multiple times.
                        num_row_groups = pf.metadata.num_row_groups
                        remaining = rows_to_read
                        file_chunks: list[pd.DataFrame] = []
                        for rg_idx in range(num_row_groups):
                            if remaining <= 0:
                                break
                            rg_df = pf.read_row_group(rg_idx).to_pandas()
                            take = min(remaining, len(rg_df))
                            file_chunks.append(rg_df.head(take))
                            remaining -= take
                            rows_loaded += take
                            logger.info(
                                "Loaded %d/%d rows from %s via row-group %d/%d",
                                take, total_rows_in_file, file_path.name, rg_idx + 1, num_row_groups,
                            )
                        file_df = pd.concat(file_chunks, ignore_index=True) if file_chunks else None
                        # Continue past the parquet block; file_df will be appended below
                if isinstance(file_df, pd.DataFrame) and not file_df.empty:
                    file_df["_source_file"] = file_path.name
                    dfs.append(file_df)
                    rows_loaded += len(file_df)
                    logger.info("Loaded %d rows from %s", len(file_df), file_path.name)
                    if row_limit is not None and rows_loaded >= row_limit:
                        break
                    continue
            elif ext == ".parquet":
                file_df = pd.read_parquet(file_path)
            elif ext == ".json":
                file_df = pd.read_json(file_path, encoding="utf-8")
                if not isinstance(file_df, pd.DataFrame):
                    file_df = pd.DataFrame(file_df if isinstance(file_df, list) else [file_df])
            elif ext == ".jsonl":
                if row_limit is not None:
                    rows: list[dict] = []
                    with open(file_path, encoding="utf-8") as fh:
                        for line in fh:
                            rows.append(json.loads(line))
                            if len(rows) >= row_limit - rows_loaded:
                                break
                    file_df = pd.DataFrame(rows)
                else:
                    file_df = pd.read_json(file_path, lines=True, encoding="utf-8")
            elif ext == ".csv":
                if row_limit is not None:
                    file_df = pd.read_csv(
                        file_path, encoding="utf-8", low_memory=False, nrows=row_limit,
                    )
                else:
                    file_df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
            else:
                continue

            if isinstance(file_df, pd.DataFrame) and not file_df.empty:
                file_df["_source_file"] = file_path.name
                dfs.append(file_df)
                rows_loaded += len(file_df)
                logger.info("Loaded %d rows from %s", len(file_df), file_path.name)
                if row_limit is not None and rows_loaded >= row_limit:
                    break
        except Exception as exc:
            logger.warning("Failed to load %s: %s — skipping", file_path.name, exc)
            continue

    if not dfs:
        raise ValueError(f"Could not load any supported files from {local_path}")

    merged = pd.concat(dfs, ignore_index=True)
    logger.info("Total dataset rows: %d", len(merged))
    return merged


def _combine_columns(row: pd.Series, text_columns: list[str]) -> str:
    """Combine selected text columns into a single rich string for embedding."""
    parts: list[str] = []
    for col in text_columns:
        val = row.get(col)
        if val is None:
            continue
        if isinstance(val, str):
            if val.strip():
                parts.append(f"{col.replace('_', ' ').title()}: {val.strip()}")
        elif isinstance(val, (int, float, bool)):
            parts.append(f"{col.replace('_', ' ').title()}: {val}")
    return "\n".join(parts)


def _rows_from_df(
    df: pd.DataFrame,
    text_columns: list[str],
    all_columns: list[str],
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame to a list of row dicts for embedding and metadata storage.
    """
    text_col_set = set(text_columns)
    result: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        combined = _combine_columns(row, text_columns)
        if not combined:
            continue

        meta: dict[str, Any] = {}
        for col in all_columns:
            if col in ("_source_file",) or col in text_col_set:
                continue
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, (str, int, float, bool)):
                meta[col] = val
        meta["_source_file"] = row.get("_source_file", "unknown")
        result.append({"text": combined, "metadata": meta})
    return result


def _clear_cuda_cache() -> None:
    """Clear PyTorch CUDA memory cache. No-op if torch not available."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


async def ingest_hf_dataset(
    local_path: Path,
    dataset_id: str,
    qdrant_store: QdrantStore,
    embedder: OllamaEmbedder,
    *,
    split: str = "train",
    text_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    batch_size: int = 16,  # memory-safe default
    chunk_overlap: int = 180,
    max_tokens_per_chunk: int = 500,
    row_limit: int | None = None,
    sparse_embedder: SparseEmbedder | None = None,
    progress_callback: HFProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Ingest a HuggingFace-format dataset from local disk into Qdrant.

    Memory management
    ----------------
    Embedding + upserting is done in streaming fashion per batch:

        all_rows → chunk → accumulate → batch embed → sparse → upsert → gc → next batch

    Peak memory is bounded by batch_size (default 16) regardless of dataset size.
    A dataset producing 50,000 chunks uses at most 16 chunks' worth of
    embeddings in memory at any time (~200 KB of float vectors).
    """

    local_path = Path(local_path)

    async def _emit(stage: str, pct: int, msg: str, **kw: Any) -> None:
        if progress_callback:
            maybe = progress_callback({"stage": stage, "progress": pct, "message": msg, **kw})
            if _asyncio.iscoroutine(maybe):
                await maybe

    # ── Load dataset ─────────────────────────────────────────────────
    split_path = local_path / split
    if not split_path.exists():
        split_path = local_path

    await _emit("loading", 5, f"Loading dataset from {split_path.name}…")
    df = _load_hf_dataset(split_path, row_limit=row_limit)
    all_columns: list[str] = list(df.columns)

    # ── Resolve text_columns ─────────────────────────────────────────
    if text_columns is None or text_columns == []:
        auto_detected = _detect_text_column(all_columns)
        if auto_detected:
            tc: list[str] = [auto_detected]
            logger.info("Auto-detected text column: %s", auto_detected)
        else:
            for col in all_columns:
                if df[col].dtype == object and col != "_source_file":
                    tc = [col]
                    break
            else:
                raise ValueError(
                        f"Could not auto-detect text column. Columns: {all_columns}. "
                        "Please specify text_columns explicitly."
                    )
    else:
        tc = text_columns

    missing = [c for c in tc if c not in all_columns]
    if missing:
        raise ValueError(f"text_column(s) not found: {missing}. Available: {all_columns}")

    if row_limit:
        df = df.head(row_limit)

    total_rows = len(df)
    await _emit("loading", 20, f"Loaded {total_rows:,} rows — processing chunks…")

    # Ensure collection exists before any embedding
    if not qdrant_store.collection_exists(dataset_id):
        qdrant_store.create_collection(dataset_id)

    rows_list = _rows_from_df(df, tc, all_columns)
    # Free the DataFrame immediately after extracting rows.
    del df
    gc.collect()

    # ── Stream in bounded sub-batches ─────────────────────────────────
    # Instead of chunking ALL rows into all_chunks (OOM risk for large datasets),
    # we accumulate up to SUB_BATCH_THRESHOLD chunks, then embed+upsert that window,
    # then clear it before loading the next window.
    SUB_BATCH_THRESHOLD = 5000  # ~5k chunks ≈ ~20 MB of text; safe for any RAM
    all_chunks: list[str] = []
    all_meta: list[dict[str, Any]] = []
    quality_scores: list[float] = []
    profiles_used: dict[str, int] = {}
    total_rows_processed = 0
    total_upserted = 0

    async def _flush_batch(window_chunks: list[str], window_meta: list[dict[str, Any]]) -> int:
        """Embed + sparse + upsert one sub-batch, then clear it from memory."""
        nonlocal total_upserted
        if not window_chunks:
            return 0
        batch_idx = 0  # used only for progress math below

        async for batch_embs, batch_start, batch_end in embedder.embed_batches(
            window_chunks, batch_size=batch_size,
        ):
            bc = window_chunks[batch_start:batch_end]
            bm = window_meta[batch_start:batch_end]
            bl = len(bc)

            bsparse: list[dict[str, Any] | None] = [None] * bl
            if sparse_embedder is not None:
                sres = await _asyncio.to_thread(sparse_embedder.embed_batch, bc)
                for ix, sv in enumerate(sres):
                    if sv.indices:
                        bsparse[ix] = {"indices": sv.indices, "values": sv.values}

            records = [
                ChunkRecord(
                    chunk_id=str(uuid_lib.uuid4()),
                    chunk_text=bc[i],
                    metadata=bm[i],
                    dense_vector=batch_embs[i],
                    sparse_vector=bsparse[i],
                )
                for i in range(bl)
            ]
            upserted = qdrant_store.upsert_chunks(dataset_id, records)
            total_upserted += upserted

            # Release this batch's memory before the next one
            del batch_embs, bsparse, bc, bm, records, sres
            gc.collect()
            _clear_cuda_cache()

        flushed = len(window_chunks)
        window_chunks.clear()
        window_meta.clear()
        return flushed

    await _emit("chunking", 25, f"Chunking {total_rows:,} rows…")

    for row_data in rows_list:
        text = row_data["text"]
        meta = row_data["metadata"]
        total_rows_processed += 1

        report = score_and_select_profile(text)
        quality_scores.append(report.quality_score)
        profile_key = report.profile.profile.value
        profiles_used[profile_key] = profiles_used.get(profile_key, 0) + 1

        row_chunks = adaptive_chunk_text(
            text,
            profile=report.profile.profile,
            override_overlap=chunk_overlap,
            override_tokens=max_tokens_per_chunk,
        )

        for chunk_idx, _chunk_text in enumerate(row_chunks):
            chunk_meta: dict[str, Any] = {
                "dataset_id": dataset_id,
                "split": split,
                "source_file": meta.get("_source_file", "unknown"),
                "chunk_index": chunk_idx,
                "total_chunks_in_row": len(row_chunks),
                "quality_score": report.quality_score,
                "profile": profile_key,
                "text_columns_used": tc,
                "extension": None,  # N/A for HF datasets; manual ingest uses the file extension
                # Embedding model tracking — stored so query-time can detect mismatches.
                "embedding_model": embedder.model,
                "sparse_model": sparse_embedder.model_id if sparse_embedder else None,
            }
            if metadata_columns:
                for col in metadata_columns:
                    if col in meta:
                        chunk_meta[col] = meta[col]
            all_chunks.append(_chunk_text)
            all_meta.append(chunk_meta)

        # Flush when the window is full — release memory before the next row
        if len(all_chunks) >= SUB_BATCH_THRESHOLD:
            await _emit(
                "chunking", 30,
                f"Flushing {len(all_chunks):,} chunks to embed…",
            )
            flushed = await _flush_batch(all_chunks, all_meta)
            await _emit(
                "embedding", 40,
                f"Embedded {total_upserted:,} chunks so far…",
            )

    # Free the rows list now that iteration is complete
    del rows_list
    gc.collect()

    total_chunks = total_upserted  # final count after all flushes

    if total_chunks == 0:
        return {
            "dataset_id": dataset_id,
            "collection": dataset_id,
            "rows_processed": 0,
            "chunks_embedded": 0,
            "message": "No valid text rows found in dataset.",
        }

    # Final flush for any remaining chunks in the last partial window
    if all_chunks:
        await _flush_batch(all_chunks, all_meta)

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    del all_chunks, all_meta
    gc.collect()

    logger.info(
        "Dataset '%s': ingested %d rows → %d chunks into '%s'",
        dataset_id, total_rows_processed, total_upserted, dataset_id,
    )

    return {
        "dataset_id": dataset_id,
        "collection": dataset_id,
        "rows_processed": total_rows_processed,
        "chunks_embedded": total_upserted,
        "total_chunks": total_upserted,
        "avg_quality_score": round(avg_quality, 4),
        "profiles_used": profiles_used,
        "text_columns_used": tc,
        "message": (
            f"Ingested {total_rows_processed} rows ({total_upserted} chunks) from '{dataset_id}' "
            f"into collection '{dataset_id}'. Columns: {tc}. "
            f"Avg quality: {avg_quality:.2f}. Profiles: {profiles_used}"
        ),
    }
