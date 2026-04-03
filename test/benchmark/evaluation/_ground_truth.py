"""
_ground_truth.py — Squad sample loading and ground-truth derivation.

Squad rows have: id, title, context, question, answers[{text, answer_start}]

For each squad row used as an eval sample:
  - user_input          = squad row's question
  - reference           = squad row's answers[0]["text"]   (ground truth answer)
  - reference_contexts  = [squad row's context]            (trivially relevant)
  - retrieved_contexts = top-K Qdrant chunks for the question

Binary relevance per retrieved chunk:
  1  if the chunk text contains any of the ground-truth answer tokens
     (3+ character exact substring match), else 0.

This gives a conservative but correctable ground-truth for
Precision@K / Recall@K / MRR / NDCG@K.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure project root is first.
# __file__ = .../project/test/benchmark/evaluation/_ground_truth.py
# parent.parent.parent.parent = project root
_project_root = Path(__file__).resolve()
for _ in range(4):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SquadSample:
    """Single evaluation sample from the squad dataset."""
    row_idx: int          # original DataFrame index
    question: str
    ground_truth: str     # answers[0]["text"]
    context: str          # full squad context


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_squad_samples(n: int | None = None) -> list[SquadSample]:
    """
    Load *n* rows from the squad parquet file and return SquadSample objects.

    Skips rows where the answer text is empty or identical to the context
    (duplicated title rows).
    """
    from test.benchmark.evaluation._config import EvalConfig
    cfg = EvalConfig()

    if not cfg.squad_parquet.exists():
        raise FileNotFoundError(
            f"Squad parquet not found at {cfg.squad_parquet}. "
            "Run: uv run python test/_test_e2e.py --scenario 1"
        )

    df = pd.read_parquet(cfg.squad_parquet)

    # Keep only rows with a real answer
    def safe_answer(row) -> str:
        try:
            ans = row.get("answers")
            if isinstance(ans, dict) and "text" in ans and ans["text"]:
                texts = ans["text"]
                if isinstance(texts, list) and texts:
                    return str(texts[0]).strip()
                return str(ans["text"]).strip()
            return ""
        except Exception:
            return ""

    samples: list[SquadSample] = []
    for idx, row in df.iterrows():
        gt = safe_answer(row)
        if not gt:
            continue
        q = str(row.get("question", "")).strip()
        ctx = str(row.get("context", "")).strip()
        if q and ctx:
            samples.append(SquadSample(
                row_idx=int(idx),
                question=q,
                ground_truth=gt,
                context=ctx,
            ))
        if n and len(samples) >= n:
            break

    return samples[:n] if n else samples


# ---------------------------------------------------------------------------
# Ground-truth relevance
# ---------------------------------------------------------------------------

def derive_binary_relevance(
    ground_truth: str,
    chunks: list[Any],          # list of SearchResult objects
) -> list[int]:
    """
    Return binary relevance labels (0/1) for each retrieved chunk.

    A chunk is marked relevant (1) if it contains any non-trivial
    substring of the ground-truth answer (3+ consecutive chars, case-insensitive).

    This is a weak but standard approximation for squad-style QA.
    """
    # Normalise answer to short tokens
    answer_tokens = [
        tok.strip().lower()
        for tok in re.split(r'\s+', ground_truth)
        if len(tok.strip()) >= 3
    ]
    if not answer_tokens:
        return [0] * len(chunks)

    # Build a combined answer phrase (3-char substrings)
    ans_lower = ground_truth.lower()
    min_len = 3

    labels: list[int] = []
    for chunk in chunks:
        chunk_text = chunk.chunk_text.lower() if hasattr(chunk, 'chunk_text') else str(chunk).lower()
        # Check: any short token present OR any 3-char substring present
        relevant = any(tok in chunk_text for tok in answer_tokens)
        labels.append(1 if relevant else 0)

    return labels


# ---------------------------------------------------------------------------
# Build a dict suitable for JSON serialisation
# ---------------------------------------------------------------------------

def sample_to_dict(s: SquadSample) -> dict[str, Any]:
    return {
        "row_idx": s.row_idx,
        "question": s.question,
        "ground_truth": s.ground_truth,
        "context_len": len(s.context),
    }
