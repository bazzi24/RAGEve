#!/usr/bin/env python3
"""Restore squad Qdrant collection after it was wiped during hybrid schema rebuild."""
import asyncio
import gc
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.storage.qdrant_store import ChunkRecord, QdrantStore

OLLAMA_BASE = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "squad"
PARQUET = PROJECT_ROOT / "data" / "hf" / "squad" / "train" / "data.parquet"


async def restore():
    print(f"Reading {PARQUET}...")
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} rows")

    texts = [str(row.get("context", "") for _, row in df.iterrows()]
    print(f"Embedding {len(texts)} chunks with Ollama...")
    embedder = OllamaEmbedder(base_url=OLLAMA_BASE, model="nomic-embed-text:latest")
    vecs = await embedder.embed_batch_api(texts, api_batch_size=64)
    print(f"Embeddings done: {len(vecs)} vectors")

    print(f"Building ChunkRecord list...")
    records = []
    for row, vec in zip(df.iterrows(), vecs):
        idx, row_data = row
        records.append(ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            chunk_text=str(row_data.get("context", ""),
            metadata={
                "dataset_id": COLLECTION,
                "split": "train",
                "quality_score": 0.85,
                "profile": "natural_text",
                "source_file": "squad.parquet",
            },
            dense_vector=vec,
        ))

    print(f"Upserting {len(records)} records...")
    qdrant = QdrantStore(url=QDRANT_URL)
    qdrant.upsert_chunks(COLLECTION, records)

    info = qdrant.get_collection_info(COLLECTION)
    pts = info.get("points_count", 0) if info else 0
    print(f"squad restored: {pts} points")


if __name__ == "__main__":
    asyncio.run(restore())
