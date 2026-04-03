#!/usr/bin/env python3
import json
from pathlib import Path

f = sorted(Path("data/benchmarks").glob("matrix-*.json"))[-1]
d = json.loads(f.read_text())
results = d.get("results", [])
cfg = d.get("config", {})

print(f"File : {f.name}")
print(f"Config: {cfg.get('embed_models', [])} x {cfg.get('llm_models', [])} x {cfg.get('search_modes', [])}")
print(f"Cells : {len(results)} / {d.get('config', {}).get('total_cells', '?')}")
print()
print(f"{'Embed':<22} {'LLM':<16} {'Mode':<20} {'P@K':>6} {'MRR':>6} {'NDCG@K':>8} {'Faith':>6} {'AR':>6} {'CP':>6} {'CR':>6} {'Gnd':>6}")
print("-" * 120)
for r in results:
    err = "  [ERR]" if "error" in r else ""
    coll = r.get("embed_model", "?")
    llm = r.get("llm_model", "?")
    mode = r.get("search_mode", "?")
    ret = r.get("retrieval", {})
    jdg = r.get("judge", {})
    def g(d, k, sub=None):
        v = d.get(k, {})
        return round(v.get(sub or k, 0.0), 4) if isinstance(v, dict) else v
    p5 = round(ret.get("precision_at_k", 0.0), 4)
    mrr = round(ret.get("mrr", 0.0), 4)
    ndcg = round(ret.get("ndcg_at_k", 0.0), 4)
    faith = round(jdg.get("faithfulness", {}).get("mean", 0.0), 4)
    ar = round(jdg.get("answer_relevance", {}).get("mean", 0.0), 4)
    cp = round(jdg.get("context_precision", {}).get("mean", 0.0), 4)
    cr = round(jdg.get("context_relevance", {}).get("mean", 0.0), 4)
    gnd = round(jdg.get("groundedness", {}).get("mean", 0.0), 4)
    def fmt(v):
        if v != v: return "  nan"  # NaN
        return f"{v:>6.4f}"
    print(f"{coll:<22} {llm:<16} {mode:<20} {fmt(p5)} {fmt(mrr)} {fmt(ndcg)} {fmt(faith)} {fmt(ar)} {fmt(cp)} {fmt(cr)} {fmt(gnd)}{err}")
