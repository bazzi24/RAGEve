"""
chunking.py — Chunking throughput and quality benchmarks.

Measures:
  1. Adaptive chunking throughput (chunks/s) across different text profiles
  2. High-accuracy chunking throughput
  3. Chunk size distribution statistics
  4. Quality profile detection accuracy on known document types

Usage:
  uv run python -m benchmark.chunking
"""

from __future__ import annotations

import gc
import random
import sys
import time
from pathlib import Path

# Ensure project root is first so rag/backend are always importable.
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from rag.chunking.adaptive import adaptive_chunk_text
from rag.chunking.high_accuracy import deepdoc_chunk_text as ha_chunk_text
from rag.deepdoc.quality_scorer import score_and_select_profile

# ---------------------------------------------------------------------------
# Sample documents per profile
# ---------------------------------------------------------------------------

DOCUMENTS: dict[str, list[str]] = {
    "natural_text": [
        "RAG systems retrieve relevant context from a knowledge base before "
        "passing it to a large language model. This improves factual accuracy "
        "and reduces hallucination by grounding responses in real documents. "
        "The retrieval step uses semantic similarity between the query and "
        "stored chunks, typically via dense vector embeddings. "
        "A well-designed chunking strategy is essential for balancing "
        "semantic coherence with retrieval granularity. "
        "Adaptive chunking analyses document structure and applies different "
        "splitting rules per section type."
        for _ in range(10)
    ],
    "documentation": [
        "# Configuration Reference\n\n"
        "## `chunk_size`\n\n"
        "**Type:** `int`  **Default:** `512`\n\n"
        "Maximum number of tokens per chunk. Smaller values increase recall "
        "at the cost of context completeness. For code-heavy documents, "
        "consider `256` to avoid splitting across function boundaries.\n\n"
        "## `chunk_overlap`\n\n"
        "**Type:** `int`  **Default:** `50`\n\n"
        "Number of overlapping tokens between adjacent chunks. Overlap "
        "prevents critical information from sitting exactly at a boundary. "
        "Values between 32 and 128 are recommended for most use cases."
        for _ in range(10)
    ],
    "research_paper": [
        "Abstract — We present a novel approach to retrieval-augmented generation "
        "that adapts its chunking strategy based on document structure analysis. "
        "Experiments on three benchmark datasets show that adaptive chunking "
        "improves top-k recall by 12.3% compared to fixed-size baselines. "
        "Related work — Prior approaches to document segmentation include "
        "recursive character splitting and semantic sentence clustering. "
        "Methodology — We evaluate on SQuAD, Natural Questions, and TriviaQA. "
        "Results — Our method achieves state-of-the-art on all three benchmarks."
        for _ in range(10)
    ],
    "conversation": [
        "User: How does adaptive chunking work?\n"
        "Assistant: It analyses document structure and applies different "
        "splitting rules per section type. For example, code blocks are kept "
        "together while paragraphs are split at sentence boundaries.\n"
        "User: What about tables?\n"
        "Assistant: Tables are treated as atomic units unless they exceed "
        "the maximum chunk size, in which case they are split by row."
        for _ in range(10)
    ],
}

PROFILES = list(DOCUMENTS.keys())
NUM_DOCS_PER_PROFILE = 10   # each profile has 10 docs × ~300 chars ≈ 3 KB total


def _flatten(docs: list[str]) -> list[str]:
    return [d for doc in docs for d in doc]


def bench_adaptive_chunking(num_runs: int = 3) -> dict:
    """Measure adaptive chunking throughput across all profiles."""
    gc.collect()

    per_profile: dict[str, dict] = {}
    all_chunks: list[list[str]] = []
    total_text_chars = 0

    for profile in PROFILES:
        docs = DOCUMENTS[profile]
        text_chars = sum(len(d) for d in docs)
        total_text_chars += text_chars

        timings: list[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            chunks: list[str] = []
            for doc in docs:
                report = score_and_select_profile(doc)
                chs = adaptive_chunk_text(doc, profile=report.profile.profile)
                chunks.extend(chs)
            timings.append(time.perf_counter() - t0)

        avg_elapsed = sum(timings) / len(timings)
        all_chunks.append(chunks)
        chars_per_sec = text_chars / avg_elapsed if avg_elapsed > 0 else 0

        per_profile[profile] = {
            "docs": len(docs),
            "total_chars": text_chars,
            "chunks_produced": len(chunks),
            "avg_elapsed_s": round(avg_elapsed, 4),
            "throughput_chars_per_sec": round(chars_per_sec, 1),
        }

    total_chunks = sum(len(c) for c in all_chunks)
    return {
        "profiles": per_profile,
        "total_docs": sum(len(docs) for docs in DOCUMENTS.values()),
        "total_chars": total_text_chars,
        "total_chunks": total_chunks,
        "num_runs": num_runs,
    }


def bench_high_accuracy_chunking(num_runs: int = 3) -> dict:
    """Measure high-accuracy chunking throughput."""
    gc.collect()
    docs = [d for ds in DOCUMENTS.values() for d in ds]
    total_chars = sum(len(d) for d in docs)

    timings: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        chunks: list[str] = []
        for doc in docs:
            chs = ha_chunk_text(doc)
            chunks.extend(chs)
        timings.append(time.perf_counter() - t0)

    avg_elapsed = sum(timings) / len(timings)
    return {
        "total_docs": len(docs),
        "total_chars": total_chars,
        "total_chunks": len(chunks),
        "avg_elapsed_s": round(avg_elapsed, 4),
        "throughput_chars_per_sec": round(total_chars / avg_elapsed, 1) if avg_elapsed > 0 else 0,
        "num_runs": num_runs,
    }


def bench_chunk_size_distribution() -> dict:
    """Report chunk length statistics for adaptive chunking across all profiles."""
    all_lengths: list[int] = []
    for profile in PROFILES:
        for doc in DOCUMENTS[profile]:
            report = score_and_select_profile(doc)
            chunks = adaptive_chunk_text(doc, profile=report.profile.profile)
            all_lengths.extend(len(c) for c in chunks)

    all_lengths.sort()
    n = len(all_lengths)
    if not all_lengths:
        return {"error": "No chunks produced"}

    return {
        "total_chunks": n,
        "min_chars": min(all_lengths),
        "max_chars": max(all_lengths),
        "mean_chars": round(sum(all_lengths) / n, 1),
        "p50_chars": all_lengths[int(n * 0.50)],
        "p90_chars": all_lengths[int(n * 0.90)],
        "p99_chars": all_lengths[int(n * 0.99)],
    }


def run_all() -> dict:
    print("\n  [chunking] adaptive benchmark …")
    adaptive_results = bench_adaptive_chunking()

    print("  [chunking] high-accuracy benchmark …")
    ha_results = bench_high_accuracy_chunking()

    print("  [chunking] size distribution …")
    dist_results = bench_chunk_size_distribution()

    return {
        "adaptive": adaptive_results,
        "high_accuracy": ha_results,
        "chunk_size_distribution": dist_results,
    }


def add_to(run_dict: dict, results: dict) -> None:
    run_dict["chunking"] = results


if __name__ == "__main__":
    import json
    result = run_all()
    print(json.dumps(result, indent=2))
