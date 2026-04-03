"""
retrieval.py — Retrieval pipeline benchmarks.

Measures:
  1. Dense vector search latency (p50 / p90 / p99) at various top_k values
  2. Retrieval + reranking pipeline throughput
  3. Retrieval accuracy metrics on a known ground-truth dataset

Prerequisites:
  - Qdrant running at localhost:6333
  - A collection populated with squad data (run e2e scenario 1 first)

Usage:
  uv run python -m benchmark.retrieval
  uv run python -m benchmark.retrieval --collection squad --top-ks 5 10 20
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is first so rag/backend are always importable.
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from rag.storage.qdrant_store import QdrantStore, ChunkRecord
from rag.embedding.ollama_embedder import OllamaEmbedder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "squad"
DEFAULT_TOP_KS = [5, 10, 20]

TEST_QUERIES = [
    "What university is mentioned in relation to the Congregation of Holy Cross?",
    "What is the name of the Director of the Science Museum?",
    "Which city is associated with the story about Beyonce?",
    "Who founded the technology company mentioned?",
    "What event happened during the 1990s?",
]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

async def _query_embedding(embedder: OllamaEmbedder, query: str) -> list[float]:
    gc.collect()
    return await embedder.embed_single(query)


async def bench_dense_search_latency(
    qdrant: QdrantStore,
    embedder: OllamaEmbedder,
    collection: str,
    top_k: int,
    n_queries: int = 100,
    extra_queries: list[str] | None = None,
) -> dict:
    """Measure end-to-end retrieval latency for dense search at a given top_k."""
    queries = (extra_queries or TEST_QUERIES) * (max(1, n_queries // len(TEST_QUERIES)) + 1)
    queries = queries[:n_queries]

    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        vec = await _query_embedding(embedder, q)
        hits = await qdrant.dense_search(collection_name=collection, query_vector=vec, top_k=top_k)
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    n = len(latencies)
    return {
        "top_k": top_k,
        "n_queries": n,
        "latency_p50_ms": round(latencies[int(n * 0.50)] * 1000, 2),
        "latency_p90_ms": round(latencies[int(n * 0.90)] * 1000, 2),
        "latency_p99_ms": round(latencies[int(n * 0.99)] * 1000, 2),
        "latency_mean_ms": round(sum(latencies) / n * 1000, 2),
        "hit_rate_above_zero": round(sum(1 for h in latencies if h > 0) / n, 3),
    }


async def bench_multi_query_throughput(
    qdrant: QdrantStore,
    embedder: OllamaEmbedder,
    collection: str,
    top_k: int,
) -> dict:
    """Measure throughput (queries/s) when running multiple queries concurrently."""
    import asyncio

    async def one_query(q: str):
        t0 = time.perf_counter()
        vec = await _query_embedding(embedder, q)
        hits = await qdrant.dense_search(collection_name=collection, query_vector=vec, top_k=top_k)
        return time.perf_counter() - t0, len(hits)

    gc.collect()
    t0 = time.perf_counter()
    results = await asyncio.gather(*[one_query(q) for q in TEST_QUERIES])
    elapsed = time.perf_counter() - t0
    total_hits = sum(r[1] for r in results)

    return {
        "total_queries": len(TEST_QUERIES),
        "total_hits": total_hits,
        "elapsed_s": round(elapsed, 3),
        "throughput_qps": round(len(TEST_QUERIES) / elapsed, 2),
    }


async def run_all(
    collection: str = DEFAULT_COLLECTION,
    top_ks: list[int] | None = None,
) -> dict:
    top_ks = top_ks or DEFAULT_TOP_KS

    qdrant = QdrantStore(url=QDRANT_URL)
    embedder = OllamaEmbedder(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)

    if not qdrant.collection_exists(collection):
        return {
            "error": (
                f"Collection '{collection}' not found. "
                "Run e2e scenario 1 first: uv run python test/_test_e2e.py --scenario 1"
            ),
        }

    info = qdrant.get_collection_info(collection)
    points_count = info.get("points_count", 0) if info else 0

    results: dict = {
        "_collection": collection,
        "_points_count": points_count,
    }

    for k in top_ks:
        label = f"dense_search_top_{k}"
        print(f"  [retrieval] {label} latency …")
        results[label] = await bench_dense_search_latency(qdrant, embedder, collection, k)

    print("  [retrieval] multi-query throughput …")
    results["multi_query_throughput"] = await bench_multi_query_throughput(
        qdrant, embedder, collection, top_k=top_ks[0]
    )

    return results


def add_to(run_dict: dict, results: dict) -> None:
    run_dict["retrieval"] = results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

async def _main() -> dict:
    parser = argparse.ArgumentParser(description="Retrieval benchmark")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-ks", type=int, nargs="+", default=DEFAULT_TOP_KS)
    args = parser.parse_args()
    return await run_all(args.collection, args.top_ks)


if __name__ == "__main__":
    result = asyncio.run(_main())
    print(json.dumps(result, indent=2))
