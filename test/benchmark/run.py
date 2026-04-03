#!/usr/bin/env python3
"""
run.py — RAGEve benchmark suite orchestrator.

Runs all benchmark modules, aggregates results, saves to JSON,
and prints a summary to stdout.

Benchmark modules:
  embedding   — Ollama embedder throughput and latency
  chunking    — Adaptive / high-accuracy chunker throughput
  retrieval   — Qdrant dense search latency and throughput
  streaming   — End-to-end RAG streaming chat latency and session CRUD

Output:
  data/benchmarks/<timestamp>.json   — full structured JSON report
  stdout                              — human-readable summary

Usage:
  uv run python test/benchmark/run.py
  uv run python test/benchmark/run.py --modules embedding chunking
  uv run python test/benchmark/run.py --modules retrieval --collection squad --top-ks 5 10
  uv run python test/benchmark/run.py --modules evaluation  # RAG quality metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is first so rag/backend are always importable.
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARK_OUT_DIR = _project_root / "data" / "benchmarks"


def _ensure_out_dir() -> Path:
    d = BENCHMARK_OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def _collect_system_info() -> dict:
    info: dict = {}

    # Python / platform
    import platform
    info["platform"] = {
        "os": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }

    # CPU
    try:
        import multiprocessing
        info["cpu"] = {"logical_cores": multiprocessing.cpu_count()}
    except Exception:
        pass

    # Memory (total RAM)
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        info["memory_gb"] = round(total_kb / 1024 / 1024, 1)
                        break
        elif sys.platform == "darwin":
            import subprocess as _s
            r = _s.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            info["memory_gb"] = round(int(r.stdout.strip()) / 1024**3, 1)
    except Exception:
        pass

    # Ollama models available
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            r = client.get("http://localhost:11434/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                info["ollama_models"] = [m.get("name") for m in models]
    except Exception:
        info["ollama_models"] = []

    return info


# ---------------------------------------------------------------------------
# Result merging helpers
# ---------------------------------------------------------------------------

def _merge(source: dict, target: dict) -> None:
    """Deep-merge source dict into target dict."""
    for key, value in source.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict) and isinstance(target[key], dict):
            _merge(value, target[key])
        # else: skip (target already has this key)


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

BENCHMARK_MODULES = {
    "embedding":  ("rag.embedding.ollama_embedder", "Ollama embedder throughput & latency"),
    "chunking":   ("rag.chunking.adaptive",          "Adaptive & high-accuracy chunking throughput"),
    "retrieval":  ("rag.storage.qdrant_store",       "Qdrant dense search latency & throughput"),
    "streaming":   ("httpx",                          "End-to-end streaming RAG chat latency"),
    "evaluation":  ("rag",                             "RAG quality: Precision@K, Recall@K, MRR, NDCG@K, Faithfulness, Answer Relevance, Context Precision, Context Relevance, Groundedness"),
}

ALL_MODULES = list(BENCHMARK_MODULES)


def _run_module(module: str, extra_args: list[str] | None = None) -> dict:
    """
    Import and call a benchmark module's `run_all()` function directly.

    Module files live under test/benchmark/ and each exposes run_all().
    "evaluation" is a package (directory) — import its __init__ instead.

    Returns the dict from run_all(), or {"error": ...} on failure.
    """
    import importlib.util
    import asyncio

    bench_dir = Path(__file__).parent
    module_file = bench_dir / f"{module}.py"

    # evaluation/ is a package — use its __init__ as the entry point
    if module == "evaluation":
        spec = importlib.util.spec_from_file_location(
            f"benchmark_evaluation",
            bench_dir / "evaluation" / "__init__.py",
        )
    else:
        spec = importlib.util.spec_from_file_location(
            f"benchmark_{module}",
            module_file,
        )

    if spec is None or spec.loader is None:
        return {"error": f"Could not load spec for {module_file}"}

    mod = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return {"error": f"exec_module failed: {exc}"}

    if not hasattr(mod, "run_all"):
        return {"error": f"benchmark.{module} has no run_all() function"}

    try:
        # run_all may be sync or async — always run in a fresh thread so we never
        # nest asyncio event loops.
        def _call() -> dict:
            result_or_coro = mod.run_all()
            if asyncio.iscoroutine(result_or_coro):
                return asyncio.run(result_or_coro)
            return result_or_coro

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(_call).result(timeout=600)
        return result
    except concurrent.futures.TimeoutError:
        return {"error": f"module '{module}' timed out after 600s"}
    except Exception as exc:
        import traceback
        return {"error": f"run_all() raised: {exc}\n{traceback.format_exc()[-500]}"}


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def _check_prereqs(modules: list[str]) -> tuple[bool, list[str]]:
    """
    Check that required services are up for the requested modules.
    Returns (all_ok, list_of_warnings).
    """
    import httpx

    warnings: list[str] = []
    all_ok = True

    def ping(label: str, url: str) -> bool:
        nonlocal all_ok
        try:
            r = httpx.get(url, timeout=5.0)
            ok = r.status_code < 500
            if not ok:
                all_ok = False
            return ok
        except Exception:
            warnings.append(f"  {label} not reachable at {url}")
            all_ok = False
            return False

    needs_ollama = "embedding" in modules or "retrieval" in modules or "streaming" in modules
    needs_qdrant = "retrieval" in modules
    needs_backend = "streaming" in modules

    if needs_ollama:
        ping("Ollama", "http://localhost:11434/api/tags")
    if needs_qdrant:
        ping("Qdrant", "http://localhost:6333/collections")
    if needs_backend:
        ping("FastAPI", "http://localhost:8000/health")

    return all_ok, warnings


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RAGEve benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--modules",
        nargs="+",
        choices=ALL_MODULES,
        default=ALL_MODULES,
        help="Which benchmark modules to run (default: all)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write JSON reports (default: data/benchmarks/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing benchmarks",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the JSON file (print-only mode)",
    )
    return p


async def _main() -> dict:
    p = _build_parser()
    args, unknown = p.parse_known_args()

    # Parse module-specific args (everything after `--` is forwarded)
    module_args: dict[str, list[str]] = {}
    if "--" in unknown:
        sep = unknown.index("--")
        raw_extra = unknown[sep + 1:]
        # Forward everything after `--` to all modules
        module_args = {m: raw_extra for m in args.modules}

    modules = args.modules

    # Output directory
    global_out_dir = args.output_dir or _ensure_out_dir()

    print(f"\n{'='*60}")
    print("  RAGEve Benchmark Suite")
    print(f"{'='*60}")
    print(f"  Modules : {', '.join(modules)}")
    print(f"  Output  : {global_out_dir}")
    print(f"  Time    : {datetime.now(timezone.utc).isoformat()}")
    print()

    # Prerequisites
    prereq_ok, prereq_warnings = _check_prereqs(modules)
    if prereq_warnings:
        for w in prereq_warnings:
            print(f"  WARNING {w}")
    if not prereq_ok:
        print("\n  Prerequisites check failed. Fix the issues above and re-run.")
        return {"error": "Prerequisites not met", "warnings": prereq_warnings}

    # System info
    system_info = _collect_system_info()
    print(f"  System  : {system_info.get('platform', {}).get('os')} / "
          f"{system_info.get('platform', {}).get('python_version')} / "
          f"{system_info.get('cpu', {}).get('logical_cores', '?')} cores")

    # Run benchmarks
    t0_run = time.perf_counter()
    run_results: dict = {
        "benchmark_version": "1.0.0",
        "modules_run": modules,
        "system_info": system_info,
    }

    for module in modules:
        print(f"\n  Running benchmark.{module} …")
        t0 = time.perf_counter()
        result = _run_module(module, module_args.get(module))
        elapsed = time.perf_counter() - t0
        print(f"  benchmark.{module} done in {elapsed:.1f}s")

        if "error" in result and not any(k in result for k in ("embedding", "chunking", "retrieval", "streaming")):
            print(f"    ERROR: {result['error'][:200]}")
        run_results[module] = result

    total_run_s = time.perf_counter() - t0_run
    run_results["_total_elapsed_s"] = round(total_run_s, 2)

    # Build final report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_s": round(total_run_s, 2),
        "modules": modules,
        "system_info": system_info,
        "results": run_results,
    }

    # Write JSON
    if not args.no_save:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = global_out_dir / f"run-{ts}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  JSON saved → {out_path}")

    # Print human-readable summary
    _print_summary(report)

    return report


def _print_summary(report: dict) -> None:
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")

    results = report.get("results", {})
    total_s = report.get("total_elapsed_s", 0)
    print(f"\n  Total run time : {total_s:.1f}s")

    # Embedding
    emb = results.get("embedding", {})
    if "batch_throughput" in emb:
        print("\n  [embedding]")
        for key, val in emb["batch_throughput"].items():
            tput = val.get("throughput_chunks_per_sec", "?")
            print(f"    {key}: {tput} chunks/s")
        lat = emb.get("query_latency", {})
        if lat:
            print(f"    query latency: p50={lat.get('latency_p50_ms')}ms  "
                  f"p90={lat.get('latency_p90_ms')}ms  "
                  f"p99={lat.get('latency_p99_ms')}ms")
        dim = emb.get("dimension", {})
        if dim:
            print(f"    dimension     : {dim.get('dimension')}  model={dim.get('embedding_model')}")

    # Chunking
    chk = results.get("chunking", {})
    if "adaptive" in chk:
        print("\n  [chunking]")
        for profile, data in chk["adaptive"].get("profiles", {}).items():
            tput = data.get("throughput_chars_per_sec", "?")
            print(f"    {profile}: {tput} chars/s  ({data.get('total_chunks', 0)} chunks)")

    # Retrieval
    ret = results.get("retrieval", {})
    if "error" in ret and not ret["error"]:
        pass
    if "dense_search_top_5" in ret:
        print("\n  [retrieval]")
        for key, val in ret.items():
            if key.startswith("dense_search_top_"):
                lat = val
                print(f"    {key}: p50={lat.get('latency_p50_ms')}ms  "
                      f"p90={lat.get('latency_p90_ms')}ms  "
                      f"p99={lat.get('latency_p99_ms')}ms  ({lat.get('n_queries')} queries)")
        tput = ret.get("multi_query_throughput", {})
        if tput:
            print(f"    multi-query throughput: {tput.get('throughput_qps')} qps")
        pts = ret.get("_points_count")
        if pts:
            print(f"    collection '{ret.get('_collection')}' points: {pts}")

    # Streaming
    stm = results.get("streaming", {})
    if "error" in stm:
        print(f"\n  [streaming] ERROR: {stm['error']}")
    lat = stm.get("streaming_latency", {})
    if lat:
        total_lat = lat.get("total_latency_s", {})
        print("\n  [streaming]")
        print(f"    queries      : {lat.get('successful', 0)}/{lat.get('total_queries', 0)} successful")
        print(f"    total latency: p50={total_lat.get('p50')}s  "
              f"p90={total_lat.get('p90')}s  "
              f"p99={total_lat.get('p99')}s")
        ftl = lat.get("first_token_latency_s", {})
        if ftl:
            print(f"    first token  : p50={ftl.get('p50')}s  p90={ftl.get('p90')}s")
        scrud = stm.get("session_crud", {})
        if scrud and "error" not in scrud:
            ms = scrud.get("timings_ms", {})
            print(f"    session CRUD : create={ms.get('create_session_ms')}ms  "
                  f"list={ms.get('list_sessions_ms')}ms  "
                  f"delete={ms.get('delete_session_ms')}ms")

    print(f"\n{'='*60}")
    print()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = asyncio.run(_main())
    if "error" in result and result.get("error") == "Prerequisites not met":
        sys.exit(1)
