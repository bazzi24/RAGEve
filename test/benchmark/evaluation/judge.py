"""
judge.py — LLM-as-judge metrics computed via direct Ollama API calls.

Metrics computed:
  - Faithfulness     : does the answer stay faithful to the retrieved contexts?
  - Answer Relevance : is the answer relevant to the user's question?
  - Context Precision: are the retrieved contexts actually relevant to the question?
  - Context Relevance: how relevant are the retrieved contexts to the question?
  - Groundedness     : can the answer be fully grounded in the retrieved contexts?

All metrics use a raw OpenAI-compatible API call to Ollama so that strict
InstructorLLM schema-following requirements do not apply.

The Ollama LLM acts as an zero-shot judge. Each metric uses a carefully
crafted prompt that asks the LLM to return a JSON object with a single
score field.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is first.
# __file__ = .../project/test/benchmark/evaluation/FILENAME.py  (4 levels deep)
_project_root = Path(__file__).resolve()
for _ in range(4):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from openai import OpenAI, RateLimitError, APITimeoutError

from test.benchmark.evaluation._config import EvalConfig


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = EvalConfig()
        _client = OpenAI(
            api_key="ollama",
            base_url=cfg.ollama_v1_url,
            timeout=120.0,
            max_retries=2,
        )
    return _client


# ---------------------------------------------------------------------------
# Score parsing helpers
# ---------------------------------------------------------------------------

def _parse_score(text: str) -> float | None:
    """Extract a float score ∈ [0, 1] from LLM response text."""
    text = text.strip()
    try:
        obj = json.loads(text)
        for key in ("score", "rating", "faithfulness", "relevance", "precision",
                    "result", "value", "groundedness"):
            if key in obj:
                val = obj[key]
                if isinstance(val, (int, float)):
                    return max(0.0, min(1.0, float(val)))
        # If the JSON is just a number
        if isinstance(obj, (int, float)):
            return max(0.0, min(1.0, float(obj)))
    except Exception:
        pass

    # Look for a decimal number anywhere in the text.
    # Negative lookbehind/lookahead ensure we match a single number token,
    # not a fragment of a larger number or a standalone "1" (ambiguous scale).
    m = re.search(r'(?<![0-9.])(0?\.[0-9]+)(?![0-9.])', text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass

    # Normalise integers/floors > 1 as 0-100 scale
    m = re.search(r'(?<![0-9.])([0-9]+\.[0-9]+)(?![0-9.])', text)
    if m:
        val = float(m.group(1))
        if val > 1:
            val = val / 100.0
        return max(0.0, min(1.0, val))

    return None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_FAITHFULNESS_TEMPLATE = """\
You are an expert evaluator of RAG system outputs.
Given the retrieved contexts and a generated answer, assess whether the answer
is **faithful** (i.e., does NOT hallucinate or contradict information in the contexts).

Score the faithfulness on a scale from 0.0 to 1.0:
  - 1.0 = answer is fully faithful; every claim is supported by the contexts
  - 0.5 = answer is partially faithful; some claims unsupported
  - 0.0 = answer is unfaithful; contradicts or invents information

IMPORTANT: Return ONLY a valid JSON object with a single key "score" (float).
No explanation, no markdown, no extra text.

Example response: {{"score": 0.95}}

---
RETRIEVED CONTEXTS:
{contexts}

---
GENERATED ANSWER:
{answer}
"""

_ANSWER_RELEVANCE_TEMPLATE = """\
You are an expert evaluator of RAG system outputs.
Given the user's question and the generated answer, assess whether the answer
is **relevant** (i.e., directly addresses what was asked).

Score the answer relevance on a scale from 0.0 to 1.0:
  - 1.0 = answer directly addresses the question
  - 0.5 = answer partially addresses the question
  - 0.0 = answer is off-topic or does not address the question

IMPORTANT: Return ONLY a valid JSON object with a single key "score" (float).
No explanation, no markdown, no extra text.

Example response: {{"score": 0.85}}

---
USER QUESTION:
{question}

---
GENERATED ANSWER:
{answer}
"""

_CONTEXT_PRECISION_TEMPLATE = """\
You are an expert evaluator of RAG system outputs.
Given the user question and a list of retrieved contexts, assess which contexts
are **actually relevant** to answering the question.

For each context, indicate whether it contains useful information to answer the question.
Respond with a JSON object mapping context index to relevance score (0=irrelevant, 1=relevant).

Example: {{"0": 1, "1": 0, "2": 1}}  means context 0 is relevant, 1 is not, 2 is relevant.

Only include contexts that appear in the list below.

---
USER QUESTION:
{question}

---
RETRIEVED CONTEXTS (in order):
{contexts}

---
Respond with a single JSON object only, no explanation.
"""

_CONTEXT_RELEVANCE_TEMPLATE = """\
You are an expert evaluator of RAG system outputs.
Given the user's question and the retrieved contexts, assess how **relevant**
the overall set of contexts is to answering the question.

Score context relevance on a scale from 0.0 to 1.0:
  - 1.0 = all contexts are highly relevant and useful
  - 0.5 = some contexts are relevant; others are noise
  - 0.0 = none of the contexts help answer the question

IMPORTANT: Return ONLY a valid JSON object with a single key "score" (float).
No explanation, no markdown, no extra text.

Example response: {{"score": 0.72}}

---
USER QUESTION:
{question}

---
RETRIEVED CONTEXTS:
{contexts}
"""

_GROUNDEDNESS_TEMPLATE = """\
You are an expert fact-checker evaluating RAG system outputs.
Given the retrieved contexts and a generated answer, verify that every factual
claim in the answer can be traced back to the contexts.

Score groundedness on a scale from 0.0 to 1.0:
  - 1.0 = every factual claim is fully supported by the contexts
  - 0.5 = most claims supported; a few cannot be verified
  - 0.0 = most or all claims are unsupported / hallucinated

IMPORTANT: Return ONLY a valid JSON object with a single key "score" (float).
No explanation, no markdown, no extra text.

Example response: {{"score": 0.90}}

---
RETRIEVED CONTEXTS:
{contexts}

---
GENERATED ANSWER:
{answer}
"""


# ---------------------------------------------------------------------------
# Low-level judge call
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_DELAY = [2, 5, 15]   # seconds between retries


async def _judge(
    prompt: str,
    system: str = "You are a precise, unbiased RAG evaluator. Always respond with valid JSON only.",
    temperature: float = 0.1,
    max_tokens: int = 128,
) -> float | None:
    """
    Call the Ollama judge LLM and extract a [0,1] score.
    Returns None if the model fails to return a parseable score.
    """
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=EvalConfig().ollama_chat_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return _parse_score(content)

        except (RateLimitError, APITimeoutError, asyncio.TimeoutError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY[attempt])
        except Exception as exc:
            last_error = exc
            break

    print(f"      [judge] failed after {MAX_RETRIES} attempts: {last_error}")
    return None


# ---------------------------------------------------------------------------
# Per-metric scorers
# ---------------------------------------------------------------------------

def _fmt_contexts(contexts: list[str]) -> str:
    lines = []
    for i, ctx in enumerate(contexts):
        truncated = ctx[:600] + ("..." if len(ctx) > 600 else "")
        lines.append(f"[{i}] {truncated}")
    return "\n".join(lines)


async def score_faithfulness(
    contexts: list[str],
    answer: str,
) -> float:
    """Faithfulness: does the answer stay faithful to the contexts?"""
    if not answer or not contexts:
        return 0.0
    prompt = _FAITHFULNESS_TEMPLATE.format(
        contexts=_fmt_contexts(contexts),
        answer=answer,
    )
    score = await _judge(prompt)
    return score if score is not None else float("nan")


async def score_answer_relevance(
    question: str,
    answer: str,
) -> float:
    """Answer Relevance: does the answer address the question?"""
    if not answer or not question:
        return 0.0
    prompt = _ANSWER_RELEVANCE_TEMPLATE.format(
        question=question,
        answer=answer,
    )
    score = await _judge(prompt)
    return score if score is not None else float("nan")


async def score_context_precision(
    question: str,
    contexts: list[str],
) -> float:
    """
    Context Precision: which of the retrieved contexts are relevant to the question?
    Returns the fraction of relevant contexts in the retrieved set.
    """
    if not contexts:
        return 0.0
    prompt = _CONTEXT_PRECISION_TEMPLATE.format(
        question=question,
        contexts=_fmt_contexts(contexts),
    )
    score = await _judge(prompt, max_tokens=256)
    if score is None:
        return float("nan")

    # _parse_score returns a single float; context precision needs a fraction
    # Re-call to get per-context JSON
    try:
        resp_raw = await asyncio.to_thread(
            _get_client().chat.completions.create,
            model=EvalConfig().ollama_chat_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        content = resp_raw.choices[0].message.content or ""
        obj = json.loads(content)
        if isinstance(obj, dict):
            vals = list(obj.values())
            if vals:
                return sum(1 for v in vals if v >= 0.5) / len(vals)
        return float("nan")
    except Exception:
        return score  # fallback to single-score interpretation


async def score_context_relevance(
    question: str,
    contexts: list[str],
) -> float:
    """Context Relevance: overall relevance of retrieved contexts to the question."""
    if not contexts:
        return 0.0
    prompt = _CONTEXT_RELEVANCE_TEMPLATE.format(
        question=question,
        contexts=_fmt_contexts(contexts),
    )
    score = await _judge(prompt)
    return score if score is not None else float("nan")


async def score_groundedness(
    contexts: list[str],
    answer: str,
) -> float:
    """Groundedness: can every factual claim in the answer be grounded in the contexts?"""
    if not answer or not contexts:
        return 0.0
    prompt = _GROUNDEDNESS_TEMPLATE.format(
        contexts=_fmt_contexts(contexts),
        answer=answer,
    )
    score = await _judge(prompt)
    return score if score is not None else float("nan")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def compute_llm_judge_metrics(
    questions: list[str],
    contexts_per_q: list[list[str]],
    answers: list[str],
) -> dict[str, Any]:
    """
    Compute all LLM-as-judge metrics for a batch of RAG results.

    Args:
        questions       : list of user questions
        contexts_per_q  : per-question list of retrieved context strings
        answers        : per-question generated answer strings

    Returns:
        dict with mean scores and per-query detail arrays.
    """
    n = len(questions)
    if n == 0:
        return {"error": "No samples provided"}

    faithfulness_scores: list[float] = []
    answer_relevance_scores: list[float] = []
    context_precision_scores: list[float] = []
    context_relevance_scores: list[float] = []
    groundedness_scores: list[float] = []

    step = max(1, n // 10)

    for i in range(n):
        q = questions[i]
        ctxs = contexts_per_q[i] if i < len(contexts_per_q) else []
        ans = answers[i] if i < len(answers) else ""

        f_task = score_faithfulness(ctxs, ans)
        ar_task = score_answer_relevance(q, ans)
        cp_task = score_context_precision(q, ctxs)
        cr_task = score_context_relevance(q, ctxs)
        g_task = score_groundedness(ctxs, ans)

        f, ar, cp, cr, g = await asyncio.gather(
            f_task, ar_task, cp_task, cr_task, g_task
        )

        faithfulness_scores.append(f)
        answer_relevance_scores.append(ar)
        context_precision_scores.append(cp)
        context_relevance_scores.append(cr)
        groundedness_scores.append(g)

        if (i + 1) % step == 0:
            pct = (i + 1) * 100 // n
            print(f"    [judge] {pct}% ({i+1}/{n})")

    def _mean(vals: list[float]) -> float:
        nums = [v for v in vals if not (v != v)]  # filter NaN
        return float(sum(nums) / len(nums)) if nums else float("nan")

    def _std(vals: list[float]) -> float:
        nums = [v for v in vals if v == v]  # filter NaN
        if len(nums) < 2:
            return float("nan")
        m = sum(nums) / len(nums)
        return float((sum((v - m) ** 2 for v in nums) / len(nums)) ** 0.5)

    def _n_nan(vals: list[float]) -> int:
        return sum(1 for v in vals if v != v)

    return {
        "n_queries": n,
        "n_nan": {
            "faithfulness": _n_nan(faithfulness_scores),
            "answer_relevance": _n_nan(answer_relevance_scores),
            "context_precision": _n_nan(context_precision_scores),
            "context_relevance": _n_nan(context_relevance_scores),
            "groundedness": _n_nan(groundedness_scores),
        },
        "faithfulness": {
            "mean": round(_mean(faithfulness_scores), 4),
            "std": round(_std(faithfulness_scores), 4),
        },
        "answer_relevance": {
            "mean": round(_mean(answer_relevance_scores), 4),
            "std": round(_std(answer_relevance_scores), 4),
        },
        "context_precision": {
            "mean": round(_mean(context_precision_scores), 4),
            "std": round(_std(context_precision_scores), 4),
        },
        "context_relevance": {
            "mean": round(_mean(context_relevance_scores), 4),
            "std": round(_std(context_relevance_scores), 4),
        },
        "groundedness": {
            "mean": round(_mean(groundedness_scores), 4),
            "std": round(_std(groundedness_scores), 4),
        },
        # per-query detail
        "per_query": [
            {
                "question": q[:80],
                "faithfulness": round(faithfulness_scores[i], 4),
                "answer_relevance": round(answer_relevance_scores[i], 4),
                "context_precision": round(context_precision_scores[i], 4),
                "context_relevance": round(context_relevance_scores[i], 4),
                "groundedness": round(groundedness_scores[i], 4),
            }
            for i, q in enumerate(questions)
        ],
    }


def add_to(run_dict: dict, results: dict) -> None:
    run_dict["llm_judge_metrics"] = results
