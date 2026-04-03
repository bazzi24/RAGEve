"""
retrieval.py — Retrieval quality metrics: Precision@K, Recall@K, MRR, NDCG@K.

All metrics are computed from:
  - questions          : list[str]
  - retrieved_chunks   : list[list[SearchResult]]  (one list per question)
  - binary_relevance   : list[list[int]]           (1=relevant, 0=irrelevant)

Binary relevance is derived by `_ground_truth.derive_binary_relevance()`.
"""

from __future__ import annotations

import gc
import sys
import time
from math import log
from pathlib import Path
from typing import Any

# Ensure project root is first.
# __file__ = .../project/test/benchmark/evaluation/FILENAME.py  (4 levels deep)
_project_root = Path(__file__).resolve()
for _ in range(4):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.storage.qdrant_store import QdrantStore

from test.benchmark.evaluation._config import EvalConfig
from test.benchmark.evaluation._ground_truth import SquadSample, derive_binary_relevance


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def _dcg(gains: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    dcg = 0.0
    for i, g in enumerate(gains[:k]):
        dcg += g / log(i + 2, 2)   # i=0 → position 1 → denominator log(2)
    return dcg


def _ndcg(relevances: list[list[int]], k: int) -> float:
    """
    Normalised DCG averaged over all queries.
    relevances[r][i] ∈ {0, 1}
    """
    idcgs = [_dcg([1.0] * min(k, len(rel)), k) for rel in relevances]
    dcg_vals = [_dcg(rel, k) for rel in relevances]
    ndcg_vals = [
        dcg / idcg if idcg > 0 else 0.0
        for dcg, idcg in zip(dcg_vals, idcgs)
    ]
    return float(np.mean(ndcg_vals))


def compute_retrieval_metrics(
    questions: list[str],
    retrieved_chunks: list[list[Any]],
    ground_truth_answers: list[str],
    k: int,
) -> dict[str, Any]:
    """
    Compute Precision@K, Recall@K, MRR, NDCG@K from retrieval results.

    Args:
        questions           : list of query strings
        retrieved_chunks   : per-question list of SearchResult objects
        ground_truth_answers: per-question ground-truth answer strings
        k                  : cutoff rank

    Returns:
        dict with per-metric scalar values and per-query detail arrays.
    """
    n = len(questions)
    if n == 0:
        return {"error": "No samples provided"}

    # Derive binary relevance per question
    all_relevances: list[list[int]] = []
    for gt, chunks in zip(ground_truth_answers, retrieved_chunks):
        rels = derive_binary_relevance(gt, chunks)
        all_relevances.append(rels)

    # ── Precision@K ────────────────────────────────────────────────────────
    precisions: list[float] = []
    for rels in all_relevances:
        top = rels[:k]
        precisions.append(sum(top) / k if k > 0 else 0.0)

    # ── Recall@K ────────────────────────────────────────────────────────────
    recalls: list[float] = []
    total_rel_per_q: list[int] = []   # how many relevant chunks exist in the full set
    for rels in all_relevances:
        total_rel_per_q.append(sum(rels))
        top = rels[:k]
        recalls.append(sum(top) / sum(rels) if sum(rels) > 0 else 0.0)

    # ── MRR ─────────────────────────────────────────────────────────────────
    rr: list[float] = []   # reciprocal rank
    for rels in all_relevances:
        for rank, rel in enumerate(rels[:k], start=1):
            if rel == 1:
                rr.append(1.0 / rank)
                break
        else:
            rr.append(0.0)

    # ── NDCG@K ──────────────────────────────────────────────────────────────
    ndcg = _ndcg(all_relevances, k)

    # ── Aggregate ───────────────────────────────────────────────────────────
    def _mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    return {
        "k": k,
        "n_queries": n,
        "precision_at_k": round(_mean(precisions), 4),
        "recall_at_k": round(_mean(recalls), 4),
        "mrr": round(_mean(rr), 4),
        "ndcg_at_k": round(ndcg, 4),
        # per-query detail
        "per_query": [
            {
                "question": q[:80],
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mrr_contribution": round(r, 4),
                "relevant_in_top_k": sum(rel[:k]),
                "total_relevant": sum(rel),
            }
            for q, p, r, rel
            in zip(questions, precisions, recalls, all_relevances)
        ],
    }


# ---------------------------------------------------------------------------
# Full retrieval run (Qdrant search per question)
# ---------------------------------------------------------------------------

async def run_retrieval_eval(
    cfg: EvalConfig,
    samples: list[SquadSample],
    top_k: int,
) -> dict[str, Any]:
    """
    Run dense search for each sample's question and compute retrieval metrics.
    """
    print(f"  [retrieval] embedding {len(samples)} queries …")
    qdrant = QdrantStore(url=cfg.qdrant_url)
    embedder = OllamaEmbedder(base_url=cfg.ollama_base_url, model=cfg.ollama_embed_model)

    if not qdrant.collection_exists(cfg.dataset_id):
        return {
            "error": (
                f"Collection '{cfg.dataset_id}' not found. "
                "Run: uv run python test/_test_e2e.py --scenario 1"
            ),
        }

    info = qdrant.get_collection_info(cfg.dataset_id)
    points = info.get("points_count", 0) if info else 0
    print(f"  [retrieval] collection '{cfg.dataset_id}' has {points} points")

    questions: list[str] = []
    retrieved_chunks: list[list] = []
    ground_truth_answers: list[str] = []

    step = max(1, len(samples) // 10)

    for i, sample in enumerate(samples):
        gc.collect()
        qvec = await embedder.embed_single(sample.question)
        hits = await qdrant.dense_search(
            collection_name=cfg.dataset_id,
            query_vector=qvec,
            top_k=top_k,
        )
        questions.append(sample.question)
        retrieved_chunks.append(hits)
        ground_truth_answers.append(sample.ground_truth)

        if (i + 1) % step == 0:
            pct = (i + 1) * 100 // len(samples)
            print(f"  [retrieval] {pct}% ({i+1}/{len(samples)})")

    print(f"  [retrieval] computing metrics …")
    metrics = compute_retrieval_metrics(
        questions=questions,
        retrieved_chunks=retrieved_chunks,
        ground_truth_answers=ground_truth_answers,
        k=top_k,
    )

    metrics["_collection"] = cfg.dataset_id
    metrics["_points_count"] = points
    return metrics


def add_to(run_dict: dict, results: dict) -> None:
    run_dict["retrieval_metrics"] = results
