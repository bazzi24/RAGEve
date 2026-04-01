"""
End-to-end RAG retrieval test.

Steps:
  1. Load a few rows from squad (local parquet).
  2. Adaptive-chunk + quality-score each row.
  3. Batch-embed chunks via Ollama.
  4. Upsert into Qdrant (collection = "squad").
  5. Embed a natural-language question via Ollama.
  6. Retrieve top-k chunks using the fixed QdrantStore.search().
  7. Print retrieved chunks with scores.

Usage:
  uv run python _test_search.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from rag.chunking.adaptive import adaptive_chunk_text
from rag.deepdoc.quality_scorer import score_and_select_profile
from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.ingestion.hf_ingestion import _load_hf_dataset, _detect_text_column
from rag.storage.qdrant_store import QdrantStore, ChunkRecord

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"

DATASET_PATH = Path("data/hf/squad")
DATASET_ID = "squad"
COLLECTION_NAME = DATASET_ID          # one collection per dataset
ROW_LIMIT = 50                        # ingest N rows for the test
QUERY = "Who did Beyonce marry?"
TOP_K = 4

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {title}\n{sep}")


def fmt_score(score: float) -> str:
    return f"{score:.4f}"


# ── Main test ─────────────────────────────────────────────────────────────────

async def main() -> None:
    # 1. Initialise stores
    print_section("Initialising stores")
    qdrant = QdrantStore(url=QDRANT_URL)
    embedder = OllamaEmbedder(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)

    print(f"  Qdrant  -> {QDRANT_URL}")
    print(f"  Ollama  -> {OLLAMA_BASE_URL}  ({OLLAMA_EMBED_MODEL})")
    print(f"  Dataset -> {DATASET_PATH}")

    # 2. Load dataset
    print_section("Loading dataset")
    df = _load_hf_dataset(DATASET_PATH)

    # Auto-detect text column; squad uses "context"
    text_col = _detect_text_column(list(df.columns))
    if text_col is None:
        text_col = "context"
    print(f"  Total rows  : {len(df)}")
    print(f"  Text column : {text_col}")
    print(f"  Row limit  : {ROW_LIMIT}")

    df_subset = df.head(ROW_LIMIT)

    # 3. Chunk + score
    print_section("Chunking & quality scoring")
    all_chunks: list[str] = []
    all_meta: list[dict] = []

    for idx, row in df_subset.iterrows():
        text = str(row.get(text_col, "")).strip()
        if not text:
            continue

        report = score_and_select_profile(text)
        chunks = adaptive_chunk_text(
            text,
            profile=report.profile.profile,
            override_overlap=180,
            override_tokens=500,
        )

        for c_idx, chunk_text in enumerate(chunks):
            all_chunks.append(chunk_text)
            all_meta.append({
                "dataset_id": DATASET_ID,
                "split": "train",
                "source_row": int(idx),
                "chunk_index": c_idx,
                "total_chunks_in_row": len(chunks),
                "quality_score": round(report.quality_score, 4),
                "profile": report.profile.profile.value,
            })

    print(f"  Rows processed : {len(df_subset)}")
    print(f"  Chunks created  : {len(all_chunks)}")

    # 4. Embed chunks
    print_section("Embedding chunks via Ollama")
    embeddings = await embedder.embed_batch(all_chunks, batch_size=16)
    print(f"  Embeddings : {len(embeddings)} vectors, dim={len(embeddings[0]) if embeddings else 0}")

    # 5. Upsert to Qdrant -- wipe collection first for a clean test
    print_section("Upserting to Qdrant")
    deleted = qdrant.delete_collection(COLLECTION_NAME)
    print(f"  Collection deleted (was present) : {deleted}")
    qdrant.create_collection(COLLECTION_NAME)
    print(f"  Collection created               : {COLLECTION_NAME}")

    records: list[ChunkRecord] = [
        ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            chunk_text=chunk,
            metadata=meta,
            vector=emb,
        )
        for chunk, meta, emb in zip(all_chunks, all_meta, embeddings)
    ]
    upserted = qdrant.upsert_chunks(COLLECTION_NAME, records)
    print(f"  Records upserted : {upserted}")

    # Verify collection
    info = qdrant.get_collection_info(COLLECTION_NAME)
    if info:
        print(f"  Collection status : {info['status']}  |  points: {info['points_count']}")

    # 6. Embed the query
    print_section("Retrieving")
    print(f"  Query : {QUERY!r}")
    print(f"  top_k : {TOP_K}")
    print()

    query_vector = await embedder.embed_single(QUERY)
    print(f"  Query embedding dim : {len(query_vector)}")

    # 7. Search (the fixed sync search method)
    hits = await qdrant.dense_search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        top_k=TOP_K,
    )

    if not hits:
        print("  [No hits returned -- check Qdrant is running and collection is indexed]")
        return

    print(f"  Retrieved {len(hits)} chunks:\n")

    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        preview = hit.chunk_text[:120].replace("\n", " ")
        print(f"  [{i}] score={fmt_score(hit.score)}  profile={meta.get('profile', '?')}  quality={meta.get('quality_score', 0):.3f}")
        print(f"       [row {meta.get('source_row', '?')} | chunk {meta.get('chunk_index', '?')}/{meta.get('total_chunks_in_row', '?')}]")
        print(f"       \"{preview}{'...' if len(hit.chunk_text) > 120 else ''}\"")
        print()

    # 8. Done
    print_section("Done")
    final_info = qdrant.get_collection_info(COLLECTION_NAME)
    pts = final_info["points_count"] if final_info else "?"
    print(f"  Collection '{COLLECTION_NAME}' has {pts} points.")
    print("  Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as exc:
        print(f"\n[Error] {exc}", file=sys.stderr)
        raise
