"""
streaming.py — End-to-end streaming RAG chat benchmarks.

Measures:
  1. Full RAG pipeline latency (first-token time and total response time)
  2. Token throughput (tokens/s) during streaming
  3. Chat quality: response length and source coverage
  4. Session management: create / read / delete overhead

Prerequisites:
  - Backend running at localhost:8000
  - A squad collection and agent (run e2e scenario 1 + setup-agent first)

Usage:
  uv run python -m benchmark.streaming
  uv run python -m benchmark.streaming --num-queries 10
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
import time
from pathlib import Path

# Ensure project root is first so rag/backend are always importable.
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"
DEFAULT_NUM_QUERIES = 5
REQUEST_TIMEOUT_S = 180.0

TEST_QUERIES = [
    "Summarize what you know about universities mentioned in the context.",
    "What is the Director of the Science Museum known for?",
    "Describe any stories involving major cities.",
    "What events or facts can you recall from the provided context?",
    "Give me three facts from the documents you have access to.",
]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

async def _get_or_create_agent() -> str | None:
    """Find an agent pointing at 'squad' collection, or create one."""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        try:
            r = await client.get("/agents/")
            if r.status_code == 200:
                for a in r.json().get("agents", []):
                    if a.get("config", {}).get("dataset_id") == "squad":
                        return a["agent_id"]
        except Exception:
            pass

        payload = {
            "name": "Benchmark Agent",
            "description": "Auto-created by benchmark/streaming.py",
            "config": {
                "system_prompt": (
                    "You are a helpful assistant. Answer based ONLY on the provided context. "
                    "If the context does not contain enough information, say so clearly."
                ),
                "dataset_id": "squad",
                "embedding_model": "nomic-embed-text:latest",
                "chat_model": "llama3.2:latest",
                "temperature": 0.7,
                "top_k": 5,
            },
        }
        try:
            r = await client.post("/agents/", json=payload)
            if r.status_code in (200, 201):
                return r.json().get("agent_id")
        except Exception:
            pass
    return None


async def _stream_chat(
    agent_id: str,
    question: str,
) -> dict:
    """
    Send a streaming /chat/{agent_id}/stream request and collect
    timing / token / source metrics.
    """
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT_S) as client:
        t0_request = time.perf_counter()
        first_token_ts: float | None = None
        tokens: list[str] = []
        sources: list[dict] = []
        error: str | None = None

        try:
            r = await client.post(
                f"/chat/{agent_id}/stream",
                json={"question": question, "stream": True},
                headers={"Content-Type": "application/json"},
            )
            request_latency = time.perf_counter() - t0_request

            if r.status_code != 200:
                error = f"HTTP {r.status_code}"
                return {
                    "question": question,
                    "error": error,
                    "request_latency_s": round(request_latency, 3),
                }

            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                evt = event.get("event", "")
                if evt == "chunk":
                    if first_token_ts is None:
                        first_token_ts = time.perf_counter()
                    tokens.append(event.get("content", ""))
                elif evt == "end":
                    sources = event.get("sources", [])

        except httpx.ReadTimeout:
            error = "ReadTimeout"
        except Exception as exc:
            error = str(exc)

    t_total = time.perf_counter() - t0_request
    assembled = "".join(tokens)
    first_token_latency = (first_token_ts - t0_request) if first_token_ts else None

    return {
        "question": question,
        "error": error,
        "request_latency_s": round(t_total, 3),
        "first_token_latency_s": round(first_token_latency, 3) if first_token_latency is not None else None,
        "tokens_received": len(tokens),
        "response_chars": len(assembled),
        "response_words": len(assembled.split()),
        "sources_count": len(sources),
        "avg_source_score": (
            round(sum(s.get("score", 0) for s in sources) / len(sources), 4)
            if sources else None
        ),
    }


async def bench_streaming_latency(
    agent_id: str,
    num_queries: int = DEFAULT_NUM_QUERIES,
) -> dict:
    """Run streaming queries and report per-query and aggregate latency stats."""
    queries = (TEST_QUERIES * (num_queries // len(TEST_QUERIES) + 1))[:num_queries]

    gc.collect()
    results: list[dict] = []
    for q in queries:
        r = await _stream_chat(agent_id, q)
        results.append(r)
        # Brief pause between queries to avoid hammering Ollama
        await asyncio.sleep(0.5)

    # Aggregate
    successful = [r for r in results if r.get("error") is None]
    failed = [r for r in results if r.get("error") is not None]

    total_latencies = [r["request_latency_s"] for r in successful]
    first_token_latencies = [r["first_token_latency_s"] for r in successful if r.get("first_token_latency_s") is not None]

    def _p(values: list[float], pct: float) -> float | None:
        if not values:
            return None
        s = sorted(values)
        return round(s[int(len(s) * pct)], 3)

    return {
        "total_queries": num_queries,
        "successful": len(successful),
        "failed": len(failed),
        "total_latency_s": {
            "p50": _p(total_latencies, 0.50),
            "p90": _p(total_latencies, 0.90),
            "p99": _p(total_latencies, 0.99),
            "mean": round(sum(total_latencies) / len(total_latencies), 3) if total_latencies else None,
        },
        "first_token_latency_s": {
            "p50": _p(first_token_latencies, 0.50),
            "p90": _p(first_token_latencies, 0.90),
            "mean": round(sum(first_token_latencies) / len(first_token_latencies), 3) if first_token_latencies else None,
        },
        "per_query": results,
    }


async def bench_session_crud(agent_id: str) -> dict:
    """Benchmark chat session create / list / delete overhead."""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        r = await client.post("/chat/sessions", json={"agent_id": agent_id, "title": "Benchmark session"})
        timings["create_session_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if r.status_code not in (200, 201):
            return {"error": f"create failed: HTTP {r.status_code}"}
        session_id = r.json().get("session_id")

        t0 = time.perf_counter()
        r = await client.get("/chat/sessions")
        timings["list_sessions_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if r.status_code != 200:
            return {"error": f"list failed: HTTP {r.status_code}"}
        sessions = r.json().get("sessions", [])
        found = any(s.get("session_id") == session_id for s in sessions)

        t0 = time.perf_counter()
        r = await client.delete(f"/chat/sessions/{session_id}")
        timings["delete_session_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "session_id": session_id,
            "session_found_after_create": found,
            "timings_ms": timings,
        }


async def run_all(num_queries: int = DEFAULT_NUM_QUERIES) -> dict:
    print("  [streaming] getting agent …")
    agent_id = await _get_or_create_agent()
    if not agent_id:
        return {"error": "Could not get or create an agent. Run e2e scenario 1 first."}

    print(f"  [streaming] streaming latency ({num_queries} queries) …")
    latency_results = await bench_streaming_latency(agent_id, num_queries)

    print("  [streaming] session CRUD …")
    session_results = await bench_session_crud(agent_id)

    return {
        "_agent_id": agent_id,
        "streaming_latency": latency_results,
        "session_crud": session_results,
    }


def add_to(run_dict: dict, results: dict) -> None:
    run_dict["streaming"] = results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

async def _main() -> dict:
    parser = argparse.ArgumentParser(description="Streaming benchmark")
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    args = parser.parse_args()
    return await run_all(args.num_queries)


if __name__ == "__main__":
    result = asyncio.run(_main())
    print(json.dumps(result, indent=2))
