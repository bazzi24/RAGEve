"""
embedding.py — Embedding throughput and latency benchmarks.

Measures:
  1. Batch embedding throughput (chunks/s and MB/s text throughput)
  2. Single-query embedding latency (p50 / p95 / p99)
  3. Effect of batch size on throughput (diminishing-returns curve)
  4. Embedding dimension verification

Usage:
  uv run python -m benchmark.embedding
  uv run python -m benchmark.embedding --batch-sizes 32 64 128 256
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import time
from pathlib import Path

# Ensure project root is first so rag/backend are always importable.
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from rag.embedding.ollama_embedder import OllamaEmbedder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_BATCH_SIZES = [32, 64, 128, 256]
CHUNK_COUNT_SMALL = 100
CHUNK_COUNT_LARGE = 1000   # used to measure sustained throughput

# Representative text samples of varying length (characters).
SAMPLE_TEXTS = [
    # Short sentence
    "The quick brown fox jumps over the lazy dog.",
    # Medium paragraph
    "RAG systems retrieve relevant context from a knowledge base before "
    "passing it to a large language model. This improves factual accuracy "
    "and reduces hallucination by grounding responses in real documents.",
    # Long passage
    "Adaptive chunking analyses document structure — headings, paragraphs, "
    "tables, and code blocks — and applies different splitting rules per "
    "section type. A high-accuracy mode splits sentences at word boundaries "
    "while preserving complete sentences within each chunk. Quality scoring "
    "classifies documents into profiles such as research_paper, documentation, "
    "conversation, or natural_text, and adjusts chunk size and overlap "
    "accordingly. This approach ensures that critical context is not severed "
    "mid-sentence and that each chunk carries sufficient semantic coherence "
    "for retrieval. Overlap is added between adjacent chunks to prevent "
    "important information from sitting exactly at a boundary.",
] * 40   # tile to reach desired chunk count


def _make_chunks(n: int) -> list[str]:
    """Return *n* representative text chunks, cycling the samples."""
    return [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(n)]


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

async def bench_batch_throughput(
    embedder: OllamaEmbedder,
    batch_sizes: list[int],
    chunk_count: int,
) -> dict:
    """
    Measure embedding throughput (chunks/s) for a range of batch sizes
    against the same set of chunks.
    """
    chunks = _make_chunks(chunk_count)

    results: dict[str, dict] = {}
    for bs in batch_sizes:
        gc.collect()
        t0 = time.perf_counter()
        n_processed = 0
        for i in range(0, len(chunks), bs):
            batch = chunks[i : i + bs]
            vecs = await embedder.embed_batch_api(batch, api_batch_size=bs)
            n_processed += len(vecs)
        elapsed = time.perf_counter() - t0
        chunks_per_sec = n_processed / elapsed
        results[f"batch_{bs}"] = {
            "batch_size": bs,
            "total_chunks": n_processed,
            "elapsed_s": round(elapsed, 3),
            "throughput_chunks_per_sec": round(chunks_per_sec, 1),
        }
    return results


async def bench_query_latency(
    embedder: OllamaEmbedder,
    n_queries: int = 100,
) -> dict:
    """
    Measure single-query embedding latency at p50 / p90 / p99.
    """
    queries = _make_chunks(n_queries)
    latencies: list[float] = []

    for q in queries:
        gc.collect()
        t0 = time.perf_counter()
        await embedder.embed_single(q)
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    n = len(latencies)
    return {
        "n_queries": n_queries,
        "latency_p50_ms": round(latencies[int(n * 0.50)] * 1000, 2),
        "latency_p90_ms": round(latencies[int(n * 0.90)] * 1000, 2),
        "latency_p99_ms": round(latencies[int(n * 0.99)] * 1000, 2),
        "latency_mean_ms": round(sum(latencies) / n * 1000, 2),
    }


async def bench_dimension_check(embedder: OllamaEmbedder) -> dict:
    """Verify embedding vector dimension is consistent and expected."""
    chunks = _make_chunks(10)
    vecs = await embedder.embed_batch_api(chunks, api_batch_size=10)
    dims = [len(v) for v in vecs]
    return {
        "embedding_model": embedder.model,
        "vector_count": len(vecs),
        "dimensions": dims,
        "all_same_dimension": len(set(dims)) == 1,
        "dimension": dims[0] if dims else None,
    }


async def run_all(
    embedder: OllamaEmbedder | None = None,
    batch_sizes: list[int] | None = None,
) -> dict:
    embedder = embedder or OllamaEmbedder(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)
    batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES

    print("\n  [embedding] batch throughput …")
    batch_results = await bench_batch_throughput(embedder, batch_sizes, CHUNK_COUNT_LARGE)

    print("  [embedding] query latency …")
    latency_results = await bench_query_latency(embedder)

    print("  [embedding] dimension check …")
    dimension_results = await bench_dimension_check(embedder)

    return {
        "batch_throughput": batch_results,
        "query_latency": latency_results,
        "dimension": dimension_results,
    }


def add_to(run_dict: dict, embedder_results: dict) -> None:
    """Merge embedding results into the top-level run dict."""
    for key, value in embedder_results.items():
        run_dict[key] = value


# ---------------------------------------------------------------------------
# CLI entry-point (run standalone or imported)
# ---------------------------------------------------------------------------

async def _main() -> dict:
    parser = argparse.ArgumentParser(description="Embedding benchmark")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to test (default: 32 64 128 256)",
    )
    args = parser.parse_args()
    return await run_all(embedder=None, batch_sizes=args.batch_sizes)


if __name__ == "__main__":
    result = asyncio.run(_main())
    import json
    print(json.dumps(result, indent=2))
