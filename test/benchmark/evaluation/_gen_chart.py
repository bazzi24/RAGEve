"""Generate benchmark comparison chart — nomic vs qwen3, 4 metrics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# Exact numbers from benchmark runs
MODES = ['Dense', 'Hybrid', 'Hybrid+Rerank', 'Dense+Rerank']

# nomic-embed-text (10 samples)
nomic_ndcg    = [0.3000, 0.2316, 0.2384, 0.4000]
nomic_mrr     = [0.3000, 0.2333, 0.2500, 0.4000]
nomic_recall  = [0.3000, 0.3000, 0.3000, 0.4000]
nomic_ar      = [0.5400, 0.3630, 0.2930, 0.3500]

# qwen3-embedding (10 samples)
qwen_ndcg    = [0.5000, 0.2979, 0.4106, 0.5000]
qwen_mrr     = [0.5000, 0.3667, 0.5000, 0.5000]
qwen_recall  = [0.5000, 0.5000, 0.5000, 0.5000]
qwen_ar      = [0.4430, 0.7000, 0.5330, 0.6430]

# ── Style ───────────────────────────────────────────────────────────────────
BG    = '#0d1117'
AXBG  = '#161b22'
GRID  = '#21262d'
TEXT  = '#e6edf3'
MUTED = '#8b949e'
NCOL  = '#58a6ff'   # nomic bars (blue)
GCOL  = '#3fb950'   # qwen3 bars (green)

fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=130)
fig.patch.set_facecolor(BG)
for row in axes:
    for ax in row:
        ax.set_facecolor(AXBG)
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        for tick in ax.get_yticklabels():
            tick.set_color(MUTED)
        ax.tick_params(colors=MUTED, length=0)

x = np.arange(len(MODES))
w = 0.36

def label_bars(ax, bars, color):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.018, f'{h:.2f}',
                ha='center', va='bottom', color=color, fontsize=9.5, fontweight='bold')

charts = [
    (axes[0, 0], nomic_ndcg,   qwen_ndcg,   'NDCG@K',                  'Retrieval Quality (NDCG@K)',       0.68, 0.05),
    (axes[0, 1], nomic_mrr,    qwen_mrr,    'MRR',                      'Mean Reciprocal Rank',              0.68, 0.05),
    (axes[1, 0], nomic_recall, qwen_recall, 'Recall@K',                 'Recall@K (Retrieval Coverage)',    0.68, 0.05),
    (axes[1, 1], nomic_ar,     qwen_ar,     'Answer Relevance',          'LLM Answer Relevance',              0.82, 0.08),
]

for ax, nomic_vals, qwen_vals, ylabel, title, ymax, ypad in charts:
    b1 = ax.bar(x - w/2, nomic_vals, w, label='nomic-embed-text (768d)', color=NCOL, alpha=0.92, zorder=3)
    b2 = ax.bar(x + w/2, qwen_vals,  w, label='qwen3-embedding (4096d)', color=GCOL, alpha=0.92, zorder=3)
    label_bars(ax, b1, NCOL)
    label_bars(ax, b2, GCOL)
    ax.set_xticks(x)
    ax.set_xticklabels(MODES, color=TEXT, fontsize=10.5, fontweight='bold')
    ax.set_ylabel(ylabel, color=TEXT, fontsize=11.5)
    ax.set_title(title, color=TEXT, fontsize=13, pad=12, fontweight='bold')
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper left', labelcolor=TEXT, facecolor=AXBG, edgecolor=GRID, fontsize=10.5, framealpha=0.8)

    # Highlight qwen3 bars that beat nomic
    for bar, nv, qv in zip(b2, nomic_vals, qwen_vals):
        if qv > nv:
            bar.set_edgecolor('#ffffff')
            bar.set_linewidth(1.5)

fig.suptitle(
    'RAGEve Retrieval Benchmark  ·  SQuAD v1.1  ·  top-5  ·  LLM: llama3.2\n'
    'nomic-embed-text (768d)  vs  qwen3-embedding (4096d)',
    color=MUTED, fontsize=11, y=1.015, x=0.52
)

plt.tight_layout(pad=2.5)
for out, dpi in [('docs/assets/benchmark_chart.png', 150), ('docs/assets/benchmark_chart_hd.png', 200)]:
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}  ({dpi}dpi)')
