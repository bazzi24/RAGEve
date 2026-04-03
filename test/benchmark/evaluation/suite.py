"""
suite.py — Full evaluation pipeline orchestrator.

Steps:
  1. Load squad samples (ground truth)
  2. Run Qdrant dense retrieval for each sample  → retrieval metrics
  3. Generate answers via FastAPI streaming chat  → LLM-as-judge metrics
  4. Merge all metrics into a single report
  5. Save to data/benchmarks/eval-<timestamp>.json
  6. Print human-readable summary
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is first.
# __file__ = .../project/test/benchmark/evaluation/FILENAME.py  (4 levels deep)
_project_root = Path(__file__).resolve()
for _ in range(4):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import httpx

from test.benchmark.evaluation._config import EvalConfig
from test.benchmark.evaluation._ground_truth import SquadSample, load_squad_samples
from test.benchmark.evaluation import retrieval, judge


# ---------------------------------------------------------------------------
# Answer generation via FastAPI streaming chat
# ---------------------------------------------------------------------------

async def _stream_chat_answer(
    agent_id: str,
    question: str,
    cfg: EvalConfig,
    timeout: float = 180.0,
) -> str:
    """
    Send a streaming /chat/{agent_id}/stream request to the FastAPI backend
    and return the assembled answer string.
    """
    API_BASE_URL = "http://localhost:8000"
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
                event = json.loads(line)
                if event.get("event") == "chunk":
                    tokens.append(event.get("content", ""))
            except json.JSONDecodeError:
                continue

    return "".join(tokens)


async def _get_or_create_agent(cfg: EvalConfig) -> str | None:
    """Find an existing agent for the squad collection, or create one."""
    # NOTE: /agents/ is on the FastAPI backend, NOT Ollama
    API_BASE_URL = "http://localhost:8000"
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        try:
            r = await client.get("/agents/")
            if r.status_code == 200:
                for a in r.json().get("agents", []):
                    if a.get("config", {}).get("dataset_id") == cfg.dataset_id:
                        return a["agent_id"]
        except Exception:
            pass

        payload = {
            "name": "Eval Agent",
            "description": "Auto-created by benchmark/evaluation",
            "config": {
                "system_prompt": (
                    "You are a helpful assistant. Answer based ONLY on the provided context. "
                    "If the context does not contain enough information, say so clearly."
                ),
                "dataset_id": cfg.dataset_id,
                "embedding_model": cfg.ollama_embed_model,
                "chat_model": cfg.ollama_chat_model,
                "temperature": 0.3,
                "top_k": cfg.top_k,
            },
        }
        try:
            r = await client.post("/agents/", json=payload)
            if r.status_code in (200, 201):
                return r.json().get("agent_id")
        except Exception:
            pass
    return None


async def generate_answers(
    cfg: EvalConfig,
    samples: list[SquadSample],
    agent_id: str,
) -> list[str]:
    """Generate a RAG answer for each sample. Returns list of answer strings."""
    answers: list[str] = []
    n = len(samples)
    step = max(1, n // 10)

    print(f"  [answer-gen] generating {n} answers …")
    for i, sample in enumerate(samples):
        ans = await _stream_chat_answer(agent_id, sample.question, cfg)
        answers.append(ans)
        if (i + 1) % step == 0:
            pct = (i + 1) * 100 // n
            print(f"  [answer-gen] {pct}% ({i+1}/{n})")
        # Brief pause to avoid hammering the backend
        await asyncio.sleep(0.25)

    return answers


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_full_evaluation(
    n_samples: int | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Run the complete RAGEve evaluation pipeline.

    Args:
        n_samples : number of squad rows to evaluate (default: from config)
        top_k    : retrieval top_k (default: from config)

    Returns:
        Full evaluation report dict (also saved to JSON).
    """
    cfg = EvalConfig()
    n_samples = n_samples if n_samples is not None else cfg.n_samples
    top_k = top_k if top_k is not None else cfg.top_k

    t0 = time.perf_counter()

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_samples": n_samples,
            "top_k": top_k,
            "dataset_id": cfg.dataset_id,
            "ollama_chat_model": cfg.ollama_chat_model,
            "ollama_embed_model": cfg.ollama_embed_model,
        },
        "retrieval_metrics": {},
        "llm_judge_metrics": {},
        "answers": [],
    }

    # ── Step 1: Load squad samples ─────────────────────────────────────────
    print(f"\n  [suite] loading {n_samples} squad samples …")
    samples = load_squad_samples(n_samples)
    if not samples:
        report["error"] = "No squad samples loaded"
        return report
    print(f"  [suite] loaded {len(samples)} samples")

    # ── Step 2: Retrieval metrics ──────────────────────────────────────────
    t_retrieval = time.perf_counter()
    ret_metrics = await retrieval.run_retrieval_eval(cfg, samples, top_k)
    report["retrieval_metrics"] = ret_metrics
    t_retrieval_elapsed = time.perf_counter() - t_retrieval
    print(f"  [suite] retrieval done in {t_retrieval_elapsed:.1f}s")

    if "error" in ret_metrics and not ret_metrics.get("_collection"):
        report["error"] = ret_metrics["error"]
        return report

    # ── Step 3: Agent for answer generation ─────────────────────────────────
    print(f"  [suite] getting agent for dataset '{cfg.dataset_id}' …")
    agent_id = await _get_or_create_agent(cfg)
    if not agent_id:
        report["error"] = "Could not get or create an agent. Run e2e scenario 1 first."
        return report
    print(f"  [suite] agent: {agent_id}")

    # ── Step 4: Generate RAG answers ───────────────────────────────────────
    t_answers = time.perf_counter()
    answers = await generate_answers(cfg, samples, agent_id)
    t_answers_elapsed = time.perf_counter() - t_answers
    report["_answer_gen_elapsed_s"] = round(t_answers_elapsed, 2)

    # Prepare per-sample retrieved context strings
    retrieved_contexts: list[list[str]] = []
    for sample, metric_dict in zip(samples, ret_metrics.get("per_query", [])):
        # We need the actual chunk texts — retrieve them from the eval
        # Re-run a quick retrieval to get chunk texts for judge metrics
        from rag.storage.qdrant_store import QdrantStore
        from rag.embedding.ollama_embedder import OllamaEmbedder
        gc.collect()
        qdrant = QdrantStore(url=cfg.qdrant_url)
        embedder = OllamaEmbedder(base_url=cfg.ollama_base_url, model=cfg.ollama_embed_model)
        hits = await qdrant.dense_search(cfg.dataset_id, await embedder.embed_single(sample.question), top_k=top_k)
        retrieved_contexts.append([hit.chunk_text for hit in hits])

    # ── Step 5: LLM-as-judge metrics ───────────────────────────────────────
    t_judge = time.perf_counter()
    judge_metrics = await judge.compute_llm_judge_metrics(
        questions=[s.question for s in samples],
        contexts_per_q=retrieved_contexts,
        answers=answers,
    )
    report["llm_judge_metrics"] = judge_metrics
    t_judge_elapsed = time.perf_counter() - t_judge
    print(f"  [suite] judge metrics done in {t_judge_elapsed:.1f}s")

    # ── Step 6: Save answers (not full chunks) ─────────────────────────────
    report["answers"] = [
        {"question": s.question[:200], "answer": a[:500]}
        for s, a in zip(samples, answers)
    ]

    # ── Step 7: Total time ──────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t0
    report["_total_elapsed_s"] = round(total_elapsed, 2)
    report["_step_times_s"] = {
        "retrieval": round(t_retrieval_elapsed, 2),
        "answer_generation": round(t_answers_elapsed, 2),
        "llm_judge": round(t_judge_elapsed, 2),
    }

    # ── Step 8: Save to JSON ───────────────────────────────────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = cfg.output_dir / f"eval-{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  [suite] JSON saved → {out_path}")

    # ── Step 9: Print summary ───────────────────────────────────────────────
    _print_summary(report)

    return report


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(report: dict) -> None:
    print(f"\n{'='*60}")
    print("  Evaluation Summary")
    print(f"{'='*60}")

    cfg = report.get("config", {})
    print(f"\n  Config       : {cfg.get('n_samples')} samples, top_k={cfg.get('top_k')}, "
          f"model={cfg.get('ollama_chat_model')}")

    rm = report.get("retrieval_metrics", {})
    if "error" not in rm:
        print(f"\n  Retrieval Metrics (k={rm.get('k')}, {rm.get('n_queries')} queries):")
        print(f"    Precision@K  : {rm.get('precision_at_k', '?'):.4f}")
        print(f"    Recall@K     : {rm.get('recall_at_k', '?'):.4f}")
        print(f"    MRR          : {rm.get('mrr', '?'):.4f}")
        print(f"    NDCG@K       : {rm.get('ndcg_at_k', '?'):.4f}")

    jm = report.get("llm_judge_metrics", {})
    if "error" not in jm:
        n = jm.get("n_queries", 0)
        nan = jm.get("n_nan", {})
        print(f"\n  LLM-Judge Metrics ({n} queries, nan=unparseable):")
        for metric, stats in [
            ("  Faithfulness",      jm.get("faithfulness", {})),
            ("  Answer Relevance",  jm.get("answer_relevance", {})),
            ("  Context Precision", jm.get("context_precision", {})),
            ("  Context Relevance", jm.get("context_relevance", {})),
            ("  Groundedness",      jm.get("groundedness", {})),
        ]:
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is not None:
                print(f"    {metric:<22}: {mean:.4f}  ±{std:.4f}" if std else f"    {metric}: {mean:.4f}")
            else:
                print(f"    {metric}: nan")

    steps = report.get("_step_times_s", {})
    print(f"\n  Step times:")
    for name, secs in steps.items():
        print(f"    {name:<20}: {secs:.1f}s")
    print(f"    {'total':<20}: {report.get('_total_elapsed_s', 0):.1f}s")
    print(f"\n{'='*60}\n")


def add_to(run_dict: dict, results: dict) -> None:
    """Merge evaluation results into the benchmark run dict."""
    for key, value in results.items():
        run_dict[key] = value
