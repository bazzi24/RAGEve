#!/usr/bin/env python3
"""
run.py — CLI entry-point for the RAGEve evaluation suite.

Usage:
  uv run python test/benchmark/evaluation/run.py
  uv run python test/benchmark/evaluation/run.py --samples 50 --top-k 5

The suite requires:
  - Ollama running at localhost:11434
  - Qdrant running at localhost:6333
  - FastAPI backend running at localhost:8000
  - A "squad" collection in Qdrant (run: uv run python test/_test_e2e.py --scenario 1)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Resolve project root before any imports that need rag/backend.
# __file__ = .../project/test/benchmark/evaluation/run.py
# parent.parent.parent.parent = project root
_project_root = Path(__file__).resolve()
for _ in range(4):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from test.benchmark.evaluation.suite import run_full_evaluation
from test.benchmark.evaluation._config import EvalConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RAGEve RAG Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help=f"Number of squad rows to evaluate (default: {EvalConfig().n_samples})",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Retrieval top_k (default: {EvalConfig().top_k})",
    )
    return p


async def _main() -> dict:
    p = _build_parser()
    args = p.parse_args()

    print(f"\n{'='*60}")
    print("  RAGEve Evaluation Suite")
    print(f"{'='*60}")
    print(f"  Samples : {args.samples or EvalConfig().n_samples}")
    print(f"  Top-K   : {args.top_k or EvalConfig().top_k}")
    print()

    return await run_full_evaluation(
        n_samples=args.samples,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    result = asyncio.run(_main())
    if "error" in result:
        print(f"\nERROR: {result['error']}")
        sys.exit(1)
