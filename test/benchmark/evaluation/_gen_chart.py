"""Generate benchmark comparison chart for README using user-provided results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# User-provided data — 100 SQuAD questions, top-5 retrieval, nomic-embed-text + llama3.2
modes      = ['Dense', 'Hybrid', 'Hybrid+Rerank', 'Dense+Rerank']
ndcg       = [0.30,  0.23,    0.24,            0.40]
mrr        = [0.30,  0.23,    0.25,            0.40]
recall     = [0.30,  0.30,    0.30,            0.40]
ans_rel    = [0.54,  0.36,    0.29,            0.35]

# ── Style ───────────────────────────────────────────────────────────────────
BG       = '#0d1117'
AX_BG    = '#161b22'
GRID     = '#21262d'
TEXT     = '#e6edf3'
MUTED    = '#8b949e'
ACCENT   = '#58a6ff'
GREEN    = '#3fb950'
PURPLE   = '#bc8cff'
RED      = '#f85149'
BAR_ALT  = '#1f6feb'

fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), dpi=130)
fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_facecolor(AX_BG)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    for tick in ax.get_yticklabels():
        tick.set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)

x = np.arange(len(modes))
w = 0.55
xlabels = modes

# Helper to style bar labels
def label_bars(bars, color=TEXT):
    for bar in bars:
        h = bar.get_height()
        label = f'{h:.2f}'
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, label,
                ha='center', va='bottom', color=color, fontsize=9.5, fontweight='bold')

# ── Chart 1: NDCG@K ─────────────────────────────────────────────────────────
ax = axes[0]
# Highlight best bar
best_ndcg = max(ndcg)
bars = ax.bar(x, ndcg, w, color=[BAR_ALT if v == best_ndcg else ACCENT for v in ndcg], alpha=0.92, zorder=3)
label_bars(bars)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, color=TEXT, fontsize=9.5)
ax.set_ylabel('NDCG@K', color=TEXT, fontsize=11)
ax.set_title('Retrieval Quality (NDCG@K)', color=TEXT, fontsize=11, pad=10, fontweight='bold')
ax.set_ylim(0, 0.62)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
# Star marker on best
for i, v in enumerate(ndcg):
    if v == best_ndcg:
        ax.text(i, v + 0.055, '★ Best', ha='center', va='bottom', color=GREEN, fontsize=8.5, fontweight='bold')

# ── Chart 2: MRR ────────────────────────────────────────────────────────────
ax = axes[1]
best_mrr = max(mrr)
bars = ax.bar(x, mrr, w, color=[BAR_ALT if v == best_mrr else PURPLE for v in mrr], alpha=0.92, zorder=3)
label_bars(bars)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, color=TEXT, fontsize=9.5)
ax.set_ylabel('MRR', color=TEXT, fontsize=11)
ax.set_title('Mean Reciprocal Rank', color=TEXT, fontsize=11, pad=10, fontweight='bold')
ax.set_ylim(0, 0.62)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
for i, v in enumerate(mrr):
    if v == best_mrr:
        ax.text(i, v + 0.055, '★ Best', ha='center', va='bottom', color=GREEN, fontsize=8.5, fontweight='bold')

# ── Chart 3: Recall@K ────────────────────────────────────────────────────────
ax = axes[2]
# Dense+Rerank also best for recall
best_recall = max(recall)
bars = ax.bar(x, recall, w, color=[BAR_ALT if v == best_recall else GREEN for v in recall], alpha=0.92, zorder=3)
label_bars(bars)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, color=TEXT, fontsize=9.5)
ax.set_ylabel('Recall@K', color=TEXT, fontsize=11)
ax.set_title('Recall@K (Coverage)', color=TEXT, fontsize=11, pad=10, fontweight='bold')
ax.set_ylim(0, 0.62)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
for i, v in enumerate(recall):
    if v == best_recall:
        ax.text(i, v + 0.055, '★ Best', ha='center', va='bottom', color=GREEN, fontsize=8.5, fontweight='bold')

# ── Chart 4: Answer Relevance ───────────────────────────────────────────────
ax = axes[3]
best_ar = max(ans_rel)
bars = ax.bar(x, ans_rel, w, color=[BAR_ALT if v == best_ar else RED for v in ans_rel], alpha=0.92, zorder=3)
label_bars(bars)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, color=TEXT, fontsize=9.5)
ax.set_ylabel('Answer Relevance', color=TEXT, fontsize=11)
ax.set_title('LLM Answer Relevance', color=TEXT, fontsize=11, pad=10, fontweight='bold')
ax.set_ylim(0, 0.72)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
for i, v in enumerate(ans_rel):
    if v == best_ar:
        ax.text(i, v + 0.055, '★ Best', ha='center', va='bottom', color=GREEN, fontsize=8.5, fontweight='bold')

# ── Super title ─────────────────────────────────────────────────────────────
fig.suptitle('RAGEve Retrieval Benchmark  ·  100 SQuAD Questions  ·  nomic-embed-text (768d) + llama3.2  ·  top-5',
             color=MUTED, fontsize=10.5, y=1.01, x=0.52)

plt.tight_layout(pad=2.5)
out = 'docs/assets/benchmark_chart.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
