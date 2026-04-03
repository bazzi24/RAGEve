"""
matrix.py — Full cross-model evaluation matrix.

Generates a 16-cell evaluation table:
  Embed models: nomic-embed-text (768d), qwen3-embedding (4096d)
  LLM models : llama3.2:latest, SmolLM2-1.7B-Instruct
  Search modes: basic, hybrid, hybrid+rerank, rerank

Collections used:
  squad       — existing 768-dim collection (nomic)
  squad-qwen3 — parallel 4096-dim collection (qwen3)
  hybrid      — both collections support bm25 sparse vectors for hybrid

Usage:
  uv run python test/benchmark/evaluation/matrix.py --samples 100
  uv run python test/benchmark/evaluation/matrix.py --samples 50 --workers 2
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import log
from pathlib import Path
from typing import Any, Literal

# Resolve project root (4 levels: matrix.py → evaluation/ → benchmark/ → test/ → project/)
_project_root = Path(__file__).resolve()
for _ in range(4):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import httpx
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from openai import OpenAI

from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.storage.qdrant_store import QdrantStore

# =============================================================================
# Model & search-mode definitions
# =============================================================================

@dataclass
class EmbedModel:
    name: str          # Ollama model name
    ollama_id: str     # Ollama API model ID
    dim: int           # vector dimension
    collection: str    # Qdrant collection name
    base_url: str = "http://localhost:11434"


@dataclass
class LLMModel:
    name: str          # display name
    ollama_id: str     # Ollama API model ID
    is_vision: bool = False  # placeholder for vision models


@dataclass
class SearchMode:
    id: str          # internal key
    label: str        # display label
    use_sparse: bool = False   # include bm25 sparse
    use_rerank: bool = False   # apply bge reranker
    top_k_fetch: int = 20     # fetch this many before reranking


EMBED_MODELS = [
    EmbedModel(
        name="nomic-embed-text",
        ollama_id="nomic-embed-text:latest",
        dim=768,
        collection="squad",
    ),
    EmbedModel(
        name="qwen3-embedding",
        ollama_id="qwen3-embedding:latest",
        dim=4096,
        collection="squad-qwen3",
    ),
]

LLM_MODELS = [
    LLMModel(name="llama3.2",  ollama_id="llama3.2:latest"),
    LLMModel(name="SmolLM2-1.7B", ollama_id="thirdeyeai/SmolLM2-1.7B-Instruct-Uncensored.gguf:Q4_0"),
]

SEARCH_MODES = [
    SearchMode(id="basic",      label="Dense",          use_sparse=False, use_rerank=False, top_k_fetch=5),
    SearchMode(id="hybrid",      label="Hybrid",         use_sparse=True,  use_rerank=False, top_k_fetch=15),
    SearchMode(id="hybrid_rerank", label="Hybrid+Rerank", use_sparse=True,  use_rerank=True,  top_k_fetch=15),
    SearchMode(id="rerank",      label="Dense+Rerank",   use_sparse=False, use_rerank=True,  top_k_fetch=15),
]

# Default config
DEFAULT_N_SAMPLES = 100
DEFAULT_TOP_K = 5
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_V1_URL = f"{OLLAMA_BASE_URL}/v1"
QDRANT_URL = "http://localhost:6333"
API_BASE_URL = "http://localhost:8000"
OUTPUT_DIR = Path("data/benchmarks")

# BM25 sparse model (for hybrid search)
BM25_MODEL_ID = "Qdrant/bm25"

# Reranker
# bge-reranker-base on Ollama uses :latest tag; HuggingFace sentence-transformers
# uses the plain HF model name (no :latest).
RERANKER_ID_HF = "BAAI/bge-reranker-base"


# =============================================================================
# Squad data
# =============================================================================

def load_squad(n: int | None = None) -> list[dict]:
    """Load n squad rows. Returns list of {question, ground_truth, context}."""
    parquet_path = _project_root / "data" / "hf" / "squad" / "train" / "data.parquet"
    df = pd.read_parquet(parquet_path)
    samples = []
    for _, row in df.iterrows():
        try:
            ans = row["answers"]
            if isinstance(ans, dict) and ans.get("text"):
                texts = ans["text"]
                if isinstance(texts, list) and texts:
                    gt = str(texts[0]).strip()
                else:
                    gt = str(texts).strip()
            else:
                continue
        except Exception:
            continue
        q = str(row.get("question", "")).strip()
        ctx = str(row.get("context", "")).strip()
        if q and gt:
            samples.append({"question": q, "ground_truth": gt, "context": ctx})
        if n and len(samples) >= n:
            break
    return samples[:n]


# =============================================================================
# Retrieval: embed + search + rerank
# =============================================================================

class RetrievalRunner:
    """
    Encapsulates retrieval for one embed model.
    Handles: Ollama embedding, Qdrant dense/hybrid search, CrossEncoder reranking.
    """

    def __init__(self, embed_model: EmbedModel, top_k_fetch: int = 20):
        self.embed_model = embed_model
        self.top_k_fetch = top_k_fetch
        self._client_openai = OpenAI(api_key="ollama", base_url=OLLAMA_V1_URL, timeout=60.0)
        self._embedder = OllamaEmbedder(base_url=OLLAMA_BASE_URL, model=embed_model.ollama_id)
        self._qdrant = QdrantStore(url=QDRANT_URL)
        # Lazy: cross-encoder and sparse embedder
        self._cross_encoder: CrossEncoder | None = None
        self._sparse_embedder: Any | None = None

    def _ensure_sparse(self):
        if self._sparse_embedder is None:
            from fastembed import SparseTextEmbedding
            self._sparse_embedder = SparseTextEmbedding(BM25_MODEL_ID)
            print(f"      [sparse] loaded {BM25_MODEL_ID}")

    def _ensure_cross_encoder(self):
        if self._cross_encoder is None:
            # Force CPU for the small 2GB model (Ollama may hold the GPU)
            self._cross_encoder = CrossEncoder(
                RERANKER_ID_HF,
                device="cpu",
            )
            print(f"      [rerank] loaded {RERANKER_ID_HF} on CPU")

    @property
    def collection(self) -> str:
        return self.embed_model.collection

    @property
    def dim(self) -> int:
        return self.embed_model.dim

    def collection_exists(self) -> bool:
        return self._qdrant.collection_exists(self.collection)

    async def embed_query(self, query: str) -> list[float]:
        return await self._embedder.embed_single(query)

    async def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return await self._embedder.embed_batch_api(texts, api_batch_size=batch_size)

    def _sparse_encode(self, texts: list[str]) -> list[dict]:
        """Return BM25 sparse vectors for texts."""
        self._ensure_sparse()
        # SparseTextEmbedding yields dicts with indices/values
        results = list(self._sparse_embedder.query_embed(texts))
        return [{"indices": [int(x) for x in r.indices], "values": [float(x) for x in r.values]} for r in results]

    async def _dense_search(self, query_vec: list[float], top_k: int) -> list[dict]:
        hits = await self._qdrant.dense_search(
            collection_name=self.collection,
            query_vector=query_vec,
            top_k=top_k,
        )
        return [self._hit_to_dict(h) for h in hits]

    async def _hybrid_search(
        self,
        query_vec: list[float],
        query_text: str,
        top_k: int,
    ) -> list[dict]:
        sparse_vec = self._sparse_encode([query_text])[0]
        hits = await self._qdrant.hybrid_search(
            collection_name=self.collection,
            dense_query=query_vec,
            sparse_query=sparse_vec,
            top_k=top_k,
        )
        return [self._hit_to_dict(h) for h in hits]

    def _hit_to_dict(self, hit) -> dict:
        return {
            "chunk_id": hit.chunk_id,
            "text": hit.chunk_text,
            "score": hit.score,
            "metadata": hit.metadata,
        }

    def _rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        self._ensure_cross_encoder()
        if not hits:
            return []
        pairs = [(query, h["text"]) for h in hits]
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
        # Normalize scores to [0, 1]
        scores = np.array(scores)
        if scores.max() != scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        for hit, sc in zip(hits, scores):
            hit["rerank_score"] = float(sc)
        # Sort by rerank score descending
        sorted_hits = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        return sorted_hits[:top_k]

    async def search(
        self,
        query: str,
        mode: SearchMode,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Run search with the specified mode.
        Returns list of dicts: {chunk_id, text, score, rerank_score?}
        """
        qvec = await self.embed_query(query)

        if mode.id == "basic":
            raw = await self._dense_search(qvec, top_k=mode.top_k_fetch)
            return raw[:top_k]

        elif mode.id in ("hybrid", "hybrid_rerank"):
            raw = await self._hybrid_search(qvec, query, top_k=mode.top_k_fetch)
            if mode.id == "hybrid":
                return raw[:top_k]
            # hybrid_rerank
            reranked = self._rerank(query, raw, top_k=mode.top_k_fetch)
            return reranked[:top_k]

        elif mode.id == "rerank":
            raw = await self._dense_search(qvec, top_k=mode.top_k_fetch)
            reranked = self._rerank(query, raw, top_k=mode.top_k_fetch)
            return reranked[:top_k]

        return []


# =============================================================================
# Answer generation via streaming
# =============================================================================

async def stream_answer(
    agent_id: str,
    question: str,
    llm: LLMModel,
    system_prompt: str,
    timeout: float = 120.0,
) -> str:
    """
    Generate an answer by calling /chat/{agent_id}/stream.
    The agent already has the right collection configured.
    """
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=timeout) as client:
        try:
            r = await client.post(
                f"/chat/{agent_id}/stream",
                json={"question": question, "stream": True},
                headers={"Content-Type": "application/json"},
            )
        except Exception:
            return ""

        if r.status_code != 200:
            return ""

        tokens: list[str] = []
        async for line in r.aiter_lines():
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
                if evt.get("event") == "chunk":
                    tokens.append(evt.get("content", ""))
            except json.JSONDecodeError:
                continue
    return "".join(tokens)


async def direct_llm_answer(
    question: str,
    context: str,
    llm: LLMModel,
    timeout: float = 120.0,
) -> str:
    """
    Generate an answer directly via Ollama API (no agent needed).
    Bypasses the RAG pipeline entirely — used for LLM-only evaluation.
    """
    system = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the context does not contain the answer, say so clearly."
    )
    user = f"CONTEXT:\n{context[:4000]}\n\nQUESTION: {question}"
    client = OpenAI(api_key="ollama", base_url=OLLAMA_V1_URL, timeout=timeout)
    try:
        resp = client.chat.completions.create(
            model=llm.ollama_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


# =============================================================================
# LLM-as-judge metrics (direct Ollama calls)
# =============================================================================

_JUDGE_CLIENT: OpenAI | None = None

def _judge_client() -> OpenAI:
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = OpenAI(api_key="ollama", base_url=OLLAMA_V1_URL, timeout=120.0, max_retries=2)
    return _JUDGE_CLIENT


def _parse_judge_score(text: str) -> float | None:
    text = text.strip()
    # Try JSON
    try:
        obj = json.loads(text)
        for key in ("score", "rating", "faithfulness", "relevance",
                    "precision", "result", "value", "groundedness"):
            if key in obj and isinstance(obj[key], (int, float)):
                return max(0.0, min(1.0, float(obj[key])))
        if isinstance(obj, (int, float)):
            return max(0.0, min(1.0, float(obj)))
    except Exception:
        pass
    # Regex: find decimal number
    import re
    m = re.search(r'0?\.\d+', text)
    if m:
        return max(0.0, min(1.0, float(m.group())))
    m = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
    if m:
        val = float(m.group(1))
        return max(0.0, min(1.0, val if val <= 1 else val / 100.0))
    return None


async def judge_faithfulness(contexts: list[str], answer: str) -> float:
    if not answer or not contexts:
        return float("nan")
    ctx_block = "\n".join(f"[{i}] {c[:500]}" for i, c in enumerate(contexts))
    prompt = (
        "You are a precise RAG evaluator. Rate the FAITHFULNESS of the answer "
        "(does it stay within the provided contexts?) on a scale from 0.0 to 1.0.\n"
        f"Return ONLY JSON: {{\"score\": 0.0 to 1.0}}\n\n"
        f"CONTEXTS:\n{ctx_block}\n\n"
        f"ANSWER:\n{answer}"
    )
    try:
        resp = _judge_client().chat.completions.create(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=32,
        )
        score = _parse_judge_score(resp.choices[0].message.content or "")
        return score if score is not None else float("nan")
    except Exception:
        return float("nan")


async def judge_answer_relevance(question: str, answer: str) -> float:
    if not answer or not question:
        return float("nan")
    prompt = (
        "You are a precise RAG evaluator. Rate ANSWER RELEVANCE "
        "(does the answer address the question?) on a scale from 0.0 to 1.0.\n"
        f"Return ONLY JSON: {{\"score\": 0.0 to 1.0}}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:\n{answer}"
    )
    try:
        resp = _judge_client().chat.completions.create(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=32,
        )
        score = _parse_judge_score(resp.choices[0].message.content or "")
        return score if score is not None else float("nan")
    except Exception:
        return float("nan")


async def judge_context_relevance(question: str, contexts: list[str]) -> float:
    if not contexts:
        return float("nan")
    ctx_block = "\n".join(f"[{i}] {c[:500]}" for i, c in enumerate(contexts))
    prompt = (
        "You are a precise RAG evaluator. Rate CONTEXT RELEVANCE "
        "(how relevant are the contexts to the question?) on a scale from 0.0 to 1.0.\n"
        f"Return ONLY JSON: {{\"score\": 0.0 to 1.0}}\n\n"
        f"QUESTION: {question}\n\n"
        f"CONTEXTS:\n{ctx_block}"
    )
    try:
        resp = _judge_client().chat.completions.create(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=32,
        )
        score = _parse_judge_score(resp.choices[0].message.content or "")
        return score if score is not None else float("nan")
    except Exception:
        return float("nan")


async def judge_groundedness(contexts: list[str], answer: str) -> float:
    if not answer or not contexts:
        return float("nan")
    ctx_block = "\n".join(f"[{i}] {c[:500]}" for i, c in enumerate(contexts))
    prompt = (
        "You are a precise RAG evaluator. Rate GROUNDEDNESS "
        "(can facts in the answer be traced to the contexts?) on a scale from 0.0 to 1.0.\n"
        f"Return ONLY JSON: {{\"score\": 0.0 to 1.0}}\n\n"
        f"CONTEXTS:\n{ctx_block}\n\n"
        f"ANSWER:\n{answer}"
    )
    try:
        resp = _judge_client().chat.completions.create(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=32,
        )
        score = _parse_judge_score(resp.choices[0].message.content or "")
        return score if score is not None else float("nan")
    except Exception:
        return float("nan")


# =============================================================================
# Retrieval metrics
# =============================================================================

def _dcg(gains: list[float], k: int) -> float:
    dcg_val = 0.0
    for i, g in enumerate(gains[:k]):
        dcg_val += g / log(i + 2, 2)
    return dcg_val


def compute_retrieval_metrics(
    questions: list[str],
    retrieved_chunks: list[list[dict]],
    ground_truths: list[str],
    k: int,
) -> dict[str, Any]:
    """Compute Precision@K, Recall@K, MRR, NDCG@K."""
    n = len(questions)
    if n == 0:
        return {}

    # Binary relevance: chunk relevant if it contains any 3+ char token from the answer
    import re
    all_rels: list[list[int]] = []
    for gt, chunks in zip(ground_truths, retrieved_chunks):
        gt_lower = gt.lower()
        tokens = [t for t in re.split(r'\s+', gt_lower) if len(t) >= 3]
        rels = []
        for chunk in chunks:
            text = chunk["text"].lower()
            rels.append(1 if any(t in text for t in tokens) else 0)
        all_rels.append(rels)

    # Precision@K
    precisions = [sum(r[:k]) / k for r in all_rels]

    # Recall@K
    recalls = []
    for rels in all_rels:
        total = sum(rels)
        recalls.append(sum(rels[:k]) / total if total > 0 else 0.0)

    # MRR
    rr = []
    for rels in all_rels:
        for rank, rel in enumerate(rels[:k], 1):
            if rel == 1:
                rr.append(1.0 / rank)
                break
        else:
            rr.append(0.0)

    # NDCG@K
    idcgs = [_dcg([1.0] * min(k, len(r)), k) for r in all_rels]
    dcg_vals = [_dcg(r, k) for r in all_rels]
    ndcgs = [d / i if i > 0 else 0.0 for d, i in zip(dcg_vals, idcgs)]

    def mean(vals):
        nums = [x for x in vals if x == x]  # filter NaN
        return round(float(np.mean(nums)), 4) if nums else 0.0

    def std(vals):
        nums = [x for x in vals if x == x]
        if len(nums) < 2:
            return 0.0
        m = sum(nums) / len(nums)
        return round(float(np.std(nums, ddof=0)), 4)

    return {
        "k": k,
        "n_queries": n,
        "precision_at_k": mean(precisions),
        "precision_std": std(precisions),
        "recall_at_k": mean(recalls),
        "recall_std": std(recalls),
        "mrr": mean(rr),
        "mrr_std": std(rr),
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
        "ndcg_std": round(float(np.std(ndcgs, ddof=0)), 4),
    }


# =============================================================================
# Per-cell evaluation
# =============================================================================

async def evaluate_cell(
    cell_id: str,
    samples: list[dict],
    embed_model: EmbedModel,
    llm_model: LLMModel,
    search_mode: SearchMode,
    top_k: int,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Evaluate one cell of the matrix.
    1. Retrieval → chunks
    2. Generate answers via streaming chat
    3. LLM-as-judge metrics
    4. Retrieval metrics
    """
    n = len(samples)
    runner = RetrievalRunner(embed_model, top_k_fetch=search_mode.top_k_fetch)

    # Check/create agent
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        try:
            r = await client.get("/agents/")
            if r.status_code == 200:
                for a in r.json().get("agents", []):
                    if a.get("config", {}).get("dataset_id") == embed_model.collection:
                        agent_id = a["agent_id"]
                        break
                else:
                    agent_id = None
            else:
                agent_id = None
        except Exception:
            agent_id = None

        if agent_id is None:
            try:
                r = await client.post("/agents/", json={
                    "name": f"Matrix-{embed_model.name}",
                    "description": "Auto-created by benchmark/evaluation/matrix.py",
                    "config": {
                        "system_prompt": (
                            "You are a helpful assistant. Answer based ONLY on the provided context. "
                            "If the context does not contain enough information, say so clearly."
                        ),
                        "dataset_id": embed_model.collection,
                        "embedding_model": embed_model.ollama_id,
                        "chat_model": llm_model.ollama_id,
                        "temperature": 0.3,
                        "top_k": top_k,
                    },
                })
                if r.status_code in (200, 201):
                    agent_id = r.json().get("agent_id")
            except Exception:
                agent_id = None

    system_prompt = (
        "You are a helpful assistant. Answer based ONLY on the provided context. "
        "If the context does not contain enough information, say so clearly."
    )

    retrieved: list[list[dict]] = []
    answers: list[str] = []

    # --- Retrieval pass ---
    for i, sample in enumerate(samples):
        hits = await runner.search(sample["question"], search_mode, top_k=top_k)
        retrieved.append(hits)
        if progress_callback:
            progress_callback(cell_id, "retrieval", i + 1, n)

    # --- Answer generation pass ---
    for i, sample in enumerate(samples):
        if agent_id:
            ans = await stream_answer(agent_id, sample["question"], llm_model, system_prompt)
        else:
            # Fallback: direct LLM call
            ctx_text = " ".join(h["text"] for h in retrieved[i])
            ans = await direct_llm_answer(sample["question"], ctx_text, llm_model)
        answers.append(ans)
        if progress_callback:
            progress_callback(cell_id, "answer", i + 1, n)

    # --- LLM-as-judge ---
    f_scores, ar_scores, cr_scores, g_scores = [], [], [], []
    judge_step = max(1, n // 5)
    for i in range(n):
        ctxs = [h["text"] for h in retrieved[i]]
        f, ar, cr, g = await asyncio.gather(
            judge_faithfulness(ctxs, answers[i]),
            judge_answer_relevance(samples[i]["question"], answers[i]),
            judge_context_relevance(samples[i]["question"], ctxs),
            judge_groundedness(ctxs, answers[i]),
        )
        f_scores.append(f)
        ar_scores.append(ar)
        cr_scores.append(cr)
        g_scores.append(g)
        if (i + 1) % judge_step == 0:
            pct = (i + 1) * 100 // n
            if progress_callback:
                progress_callback(cell_id, "judge", pct, 100)

    def _mean(vals: list[float]) -> float:
        nums = [v for v in vals if v == v]
        return round(float(np.mean(nums)), 4) if nums else float("nan")

    def _std(vals: list[float]) -> float:
        nums = [v for v in vals if v == v]  # filter NaN
        if len(nums) < 2:
            return float("nan")
        return round(float(np.std(nums, ddof=0)), 4)

    judge_metrics = {
        "faithfulness":       {"mean": _mean(f_scores),  "std": _std(f_scores)},
        "answer_relevance":    {"mean": _mean(ar_scores), "std": _std(ar_scores)},
        "context_relevance":  {"mean": _mean(cr_scores), "std": _std(cr_scores)},
        "groundedness":       {"mean": _mean(g_scores),   "std": _std(g_scores)},
    }

    # --- Retrieval metrics ---
    ret_metrics = compute_retrieval_metrics(
        questions=[s["question"] for s in samples],
        retrieved_chunks=retrieved,
        ground_truths=[s["ground_truth"] for s in samples],
        k=top_k,
    )

    # Clean up GPU memory
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "cell_id": cell_id,
        "embed_model": embed_model.name,
        "llm_model": llm_model.name,
        "search_mode": search_mode.id,
        "top_k": top_k,
        "n_samples": n,
        "retrieval": ret_metrics,
        "judge": judge_metrics,
        "sample_answers": [
            {"question": s["question"][:120], "answer": a[:300]}
            for s, a in zip(samples, answers)
        ],
    }


# =============================================================================
# Collection setup: squad-qwen3
# =============================================================================

async def ensure_qwen3_collection(
    samples: list[dict],
    embed_model: EmbedModel,
    batch_size: int = 32,
) -> None:
    """
    Create squad-qwen3 collection if it doesn't exist.
    Uses qwen3-embedding for dense + fastembed/bm25 for sparse.
    Uses ChunkRecord so upsert_chunks() handles named-vector schema correctly.
    """
    from rag.storage.qdrant_store import ChunkRecord

    qdrant = QdrantStore(url=QDRANT_URL)
    coll = embed_model.collection

    if qdrant.collection_exists(coll):
        info = qdrant.get_collection_info(coll)
        pts = info.get("points_count", 0) if info else 0
        if pts > 0:
            print(f"  [setup] {coll} already exists ({pts} points) — skipping")
            return
        print(f"  [setup] {coll} exists but empty — re-creating")

    print(f"  [setup] Creating {coll} with qwen3 dense + bm25 sparse …")

    # Load libraries
    from fastembed import SparseTextEmbedding
    embedder = OllamaEmbedder(base_url=OLLAMA_BASE_URL, model=embed_model.ollama_id)
    sparse_embedder = SparseTextEmbedding(BM25_MODEL_ID)

    # Create named-vector collection
    qdrant.delete_collection(coll)
    qdrant.create_collection(coll, dense_size=embed_model.dim)

    # Prepare rows
    chunk_texts = [row["context"] for row in samples]
    chunk_metas = [
        {
            "dataset_id": coll,
            "split": "train",
            "quality_score": 0.85,
            "profile": "natural_text",
            "source_file": "squad.parquet",
        }
        for row in samples
    ]

    # Embed
    print(f"  [setup] Embedding {len(chunk_texts)} chunks with qwen3 …")
    dense_vecs: list[list[float]] = await embedder.embed_batch_api(
        chunk_texts, api_batch_size=batch_size
    )

    # Sparse encode (sync)
    sparse_vecs = [
        {"indices": [int(x) for x in r.indices], "values": [float(x) for x in r.values]}
        for r in sparse_embedder.query_embed(chunk_texts)
    ]

    # Build ChunkRecords and upsert
    chunk_records: list[ChunkRecord] = [
        ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            chunk_text=ct,
            metadata=cm,
            dense_vector=dv,
            sparse_vector=sv,
        )
        for ct, cm, dv, sv in zip(chunk_texts, chunk_metas, dense_vecs, sparse_vecs)
    ]
    print(f"  [setup] Upserting {len(chunk_records)} ChunkRecords …")
    qdrant.upsert_chunks(coll, chunk_records)

    info = qdrant.get_collection_info(coll)
    pts = info.get("points_count", 0) if info else 0
    print(f"  [setup] {coll}: {pts} points")

    del embedder, sparse_embedder
    gc.collect()


# =============================================================================
# Hybrid collection setup: add sparse vectors to squad
# =============================================================================

async def ensure_hybrid_squad() -> None:
    """
    squad collection uses an unnamed schema (flat list[float] vectors).
    To support hybrid search we need named vectors ("dense"/"sparse").
    Strategy:
      1. Scroll all existing points from squad (vector + payload)
      2. Delete squad
      3. Recreate with named-vector schema (dense + sparse)
      4. Re-upsert all points using upsert_chunks (ChunkRecord)
    """
    from rag.storage.qdrant_store import ChunkRecord
    from fastembed import SparseTextEmbedding

    qdrant = QdrantStore(url=QDRANT_URL)
    coll = "squad"

    info = qdrant.get_collection_info(coll)
    if not info:
        print(f"  [setup] squad collection not found")
        return

    # Check if already named with sparse
    params = info.get("config", {}).get("params", {})
    has_named = isinstance(params.get("vectors", {}), dict) and "dense" in params.get("vectors", {})
    has_sparse = bool(params.get("sparse_vectors"))
    if has_named and has_sparse:
        print(f"  [setup] squad already has named + sparse vectors — skipping")
        return

    print(f"  [setup] squad: need named+sparse schema — rebuilding …")

    # 1. Scroll all points
    async def _scroll_all() -> list[dict]:
        all_pts: list[dict] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            offset = None
            while True:
                body: dict = {"limit": 1000, "with_vectors": True, "with_payload": True}
                if offset:
                    body["offset"] = offset
                r = await client.post(f"{QDRANT_URL}/collections/{coll}/points/scroll", json=body)
                if r.status_code != 200:
                    print(f"  [setup] scroll failed: {r.status_code} {r.text[:200]}")
                    break
                result = r.json().get("result", {})
                all_pts.extend(result.get("points", []))
                offset = result.get("next_page_offset")
                if not offset:
                    break
        return all_pts

    print(f"  [setup] scrolling existing squad points …")
    all_pts = await _scroll_all()
    print(f"  [setup] got {len(all_pts)} points")

    if not all_pts:
        print(f"  [setup] no points to preserve — creating named schema only")
        qdrant.delete_collection(coll)
        qdrant.create_collection(coll, dense_size=768)
        return

    texts = [str(p.get("payload", {}).get("text", "")) for p in all_pts]

    # 2. Compute sparse vectors
    print(f"  [setup] computing bm25 sparse vectors …")
    sparse_embedder = SparseTextEmbedding(BM25_MODEL_ID)
    sparse_vecs = [
        {"indices": [int(x) for x in r.indices], "values": [float(x) for x in r.values]}
        for r in sparse_embedder.query_embed(texts)
    ]

    # 3. Extract existing dense vectors
    # After rebuild: vector may be list[float] (unnamed) or dict (named).
    # Always use the 'dense' key if available; fall back to raw list.
    def _get_dense_vec(pt: dict) -> list[float]:
        v = pt.get("vector")
        if isinstance(v, dict):
            dv = v.get("dense") or v.get("sparse") or v.get(list(v.keys())[0])
        elif isinstance(v, list):
            dv = v
        else:
            dv = []
        return list(dv) if dv else []

    dense_vecs = [_get_dense_vec(pt) for pt in all_pts]

    # 4. Build ChunkRecords
    chunk_records: list[ChunkRecord] = []
    for pt, dv, sv in zip(all_pts, dense_vecs, sparse_vecs):
        payload = pt.get("payload", {})
        chunk_records.append(ChunkRecord(
            chunk_id=str(pt["id"]),
            chunk_text=str(payload.get("text", "")),
            metadata=dict(payload),
            dense_vector=list(dv) if dv else [],
            sparse_vector=sv,
        ))

    # 5. Delete and recreate with named schema
    print(f"  [setup] deleting squad and recreating with named+dense+sparse schema …")
    qdrant.delete_collection(coll)
    qdrant.create_collection(coll, dense_size=768)

    # 6. Re-upsert all points
    print(f"  [setup] re-upserting {len(chunk_records)} points …")
    qdrant.upsert_chunks(coll, chunk_records)

    info2 = qdrant.get_collection_info(coll)
    pts = info2.get("points_count", 0) if info2 else 0
    print(f"  [setup] squad rebuilt: {pts} points with dense+sparse vectors")

    del sparse_embedder
    gc.collect()


# =============================================================================
# Main matrix runner
# =============================================================================

async def run_matrix(
    n_samples: int = DEFAULT_N_SAMPLES,
    top_k: int = DEFAULT_TOP_K,
    embed_models: list[EmbedModel] | None = None,
    llm_models: list[LLMModel] | None = None,
    search_modes: list[SearchMode] | None = None,
) -> dict[str, Any]:
    """Run the full evaluation matrix."""
    t0 = time.perf_counter()

    embed_models = embed_models or EMBED_MODELS
    llm_models = llm_models or LLM_MODELS
    search_modes = search_modes or SEARCH_MODES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load samples
    print(f"\n  Loading {n_samples} squad samples …")
    samples = load_squad(n_samples)
    print(f"  Loaded {len(samples)} samples")

    # Setup collections
    print(f"\n  Setting up collections …")
    # squad always exists — ensure it has sparse vectors for hybrid search
    await ensure_hybrid_squad()

    for em in embed_models:
        if em.collection == "squad-qwen3":
            await ensure_qwen3_collection(samples, em)

    # Progress tracking
    total_cells = len(embed_models) * len(llm_models) * len(search_modes)
    completed = 0
    cell_times: dict[str, float] = {}

    def progress(cell_id: str, stage: str, current: int, total: int):
        pct = current * 100 // total
        print(f"    [{cell_id}] {stage}: {pct}% ({current}/{total})")

    results: list[dict[str, Any]] = []

    for em in embed_models:
        for llm in llm_models:
            for sm in search_modes:
                cell_id = f"{em.name}×{llm.name}×{sm.id}"
                print(f"\n  [{cell_id}] Starting evaluation …")
                ct0 = time.perf_counter()

                try:
                    result = await evaluate_cell(
                        cell_id=cell_id,
                        samples=samples,
                        embed_model=em,
                        llm_model=llm,
                        search_mode=sm,
                        top_k=top_k,
                        progress_callback=progress,
                    )
                except Exception as exc:
                    print(f"  [{cell_id}] ERROR: {exc}")
                    import traceback
                    traceback.print_exc()
                    result = {
                        "cell_id": cell_id,
                        "embed_model": em.name,
                        "llm_model": llm.name,
                        "search_mode": sm.id,
                        "error": str(exc),
                    }

                cell_elapsed = time.perf_counter() - ct0
                cell_times[cell_id] = cell_elapsed
                result["_elapsed_s"] = round(cell_elapsed, 1)
                results.append(result)
                completed += 1
                print(f"  [{cell_id}] Done in {cell_elapsed:.1f}s  ({completed}/{total_cells})")

    total_elapsed = time.perf_counter() - t0

    # Build summary tables
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_samples": n_samples,
            "top_k": top_k,
            "embed_models": [e.name for e in embed_models],
            "llm_models": [l.name for l in llm_models],
            "search_modes": [s.id for s in search_modes],
            "total_cells": total_cells,
        },
        "results": results,
        "_total_elapsed_s": round(total_elapsed, 2),
        "_cell_times_s": {k: round(v, 1) for k, v in cell_times.items()},
    }

    # Save JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"matrix-{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  JSON saved → {out_path}")

    _print_matrix_summary(results, top_k)

    return summary


def _print_matrix_summary(results: list[dict], top_k: int) -> None:
    """Print a formatted table of all results."""
    print(f"\n{'='*100}")
    print(f"  Evaluation Matrix — {len(results)} cells, top_k={top_k}")
    print(f"{'='*100}")

    # Header
    modes = list(dict.fromkeys(r["search_mode"] for r in results))
    mode_labels = {
        "basic":         "Dense",
        "hybrid":        "Hybrid",
        "hybrid_rerank": "Hybrid+Rerank",
        "rerank":       "Dense+Rerank",
    }
    print(f"\n{'Cell':<35}", end="")
    for m in modes:
        print(f"  {mode_labels.get(m, m):<18}", end="")
    print()
    print("-" * (35 + 20 * len(modes)))

    # Group by embed × LLM
    rows: dict[str, list[dict]] = {}
    for r in results:
        key = f"{r['embed_model']} / {r['llm_model']}"
        rows.setdefault(key, []).append(r)

    for row_key, cell_results in rows.items():
        print(f"\n{row_key}")
        for metric in ["ndcg_at_k", "mrr", "precision_at_k", "recall_at_k",
                       "faithfulness", "answer_relevance", "context_relevance", "groundedness"]:
            print(f"  {metric:<22}", end="")
            for r in cell_results:
                if "error" in r:
                    print(f"  {'ERR':<18}", end="")
                    continue
                if metric in r.get("retrieval", {}):
                    val = r["retrieval"][metric]
                    print(f"  {val:<18.4f}", end="")
                elif metric in r.get("judge", {}):
                    val = r["judge"][metric]["mean"]
                    print(f"  {val:<18.4f}", end="")
                else:
                    print(f"  {'-':<18}", end="")
            print()

    print(f"\n{'='*100}")
    print()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RAGEve Cross-Model Evaluation Matrix")
    p.add_argument("--samples", type=int, default=DEFAULT_N_SAMPLES,
                   help=f"Number of squad rows (default: {DEFAULT_N_SAMPLES})")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help=f"Retrieval top_k (default: {DEFAULT_TOP_K})")
    p.add_argument("--embed", nargs="+", choices=["nomic", "qwen3"], default=None,
                   help="Which embed models to run")
    p.add_argument("--llm", nargs="+", choices=["llama3.2", "SmolLM2"], default=None,
                   help="Which LLM models to run")
    p.add_argument("--mode", nargs="+", choices=["basic", "hybrid", "hybrid_rerank", "rerank"],
                   default=None, help="Which search modes to run")
    args = p.parse_args()

    embed_map = {"nomic": EMBED_MODELS[0], "qwen3": EMBED_MODELS[1]}
    llm_map = {"llama3.2": LLM_MODELS[0], "SmolLM2": LLM_MODELS[1]}
    mode_map = {s.id: s for s in SEARCH_MODES}

    embeds = [embed_map[k] for k in (args.embed or ["nomic", "qwen3"])]
    llms = [llm_map[k] for k in (args.llm or ["llama3.2", "SmolLM2"])]
    modes = [mode_map[k] for k in (args.mode or ["basic", "hybrid", "hybrid_rerank", "rerank"])]

    print(f"\n{'='*60}")
    print("  RAGEve Cross-Model Evaluation Matrix")
    print(f"{'='*60}")
    print(f"  Embed models  : {[e.name for e in embeds]}")
    print(f"  LLM models   : {[l.name for l in llms]}")
    print(f"  Search modes : {[s.id for s in modes]}")
    print(f"  Samples      : {args.samples}")
    print(f"  Top-K        : {args.top_k}")
    print(f"  Total cells  : {len(embeds) * len(llms) * len(modes)}")
    print()

    result = asyncio.run(run_matrix(
        n_samples=args.samples,
        top_k=args.top_k,
        embed_models=embeds,
        llm_models=llms,
        search_modes=modes,
    ))
