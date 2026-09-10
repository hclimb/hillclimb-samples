"""
Plot a single corpus eval run: 6 bar panels.

Layout (subplot_mosaic):
  Row 0: example-level recall | token-level hit rate | doc_access_acc
  Row 1: memory judge acc     | RAG judge acc         | RAG recall@5

Usage:
    uv run python analysis/plot_corpus_evals_single_run.py \
        --metrics path/to/metrics.json \
        --title "Base checkpoint (step 100k) — n=128" \
        --out results/corpus_evals_base.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("msmarco",        "MS MARCO"),
    ("natural_questions", "NQ"),
    ("narrativeqa",     "NarrativeQA"),
    ("2wikimultihopqa", "2Wiki"),
    ("hotpotqa",       "HotpotQA"),
    ("musique",        "MuSiQue"),
    ("dureader",       "DuReader"),
    ("popqa",          "PopQA"),
    ("triviaqa",       "TriviaQA"),
]

def v(metrics, ds_key, metric):
    # Try multiple common prefixes and variations
    search_keys = [
        f"gen_large_mem_msa_{ds_key}/{metric}",
        f"gen_large_mem_msa_{ds_key}_v1/{metric}", # for msmarco
        f"gen_large_mem_msa_{ds_key}_10m/{metric}", # for triviaqa
        f"gen_large_mem_{ds_key}/{metric}",
        f"{ds_key}/{metric}",
    ]
    for k in search_keys:
        if k in metrics:
            return metrics[k] * 100
    return 0


def bar_panel(ax, metrics, metric_key, title, ylabel, color, fmt=".1f"):
    ds_labels = [l for _, l in DATASETS]
    ds_keys   = [k for k, _ in DATASETS]
    x = np.arange(len(DATASETS))
    vals = [v(metrics, dk, metric_key) for dk in ds_keys]
    bars = ax.bar(x, vals, color=color, alpha=0.85, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:{fmt}}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=10)
    ax.set_ylim(0, max(5, max(vals) * 1.35) if max(vals) > 0 else 5)
    ax.grid(axis="y", alpha=0.3)

def plot(metrics, title, out_path):
    mosaic = [
        ["recall",    "pos_recall", "access"],
        ["mem_judge", "rag_judge",  "rag_r5"],
    ]
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(15, 8),
    )
    fig.suptitle(title, fontsize=13)

    bar_panel(axd["recall"],    metrics, "doc_hit_rate",
              "Decoding Example-Level Recall\n(any decode-step lookup hit correct doc?)",
              "% examples", "#1f77b4")

    bar_panel(axd["pos_recall"], metrics, "doc_token_hit_rate",
              "Decoding Token-Level Hit Rate\n(any head/k lookup hit correct doc)",
              "% tokens", "#17becf")

    bar_panel(axd["access"],    metrics, "doc_access_acc",
              "doc_access_acc\n(fraction of all H×S×K slots)",
              "% slots", "#2ca02c", fmt=".2f")

    bar_panel(axd["mem_judge"], metrics, "llm_judge_accuracy",
              "Memory Layers — LLM Judge Acc",
              "% correct", "#ff7f0e")

    bar_panel(axd["rag_judge"], metrics, "rag_accuracy",
              "RAG Judge Acc (Qwen3-4B)",
              "% correct", "#d62728")

    bar_panel(axd["rag_r5"],    metrics, "rag_recall@5",
              "RAG Recall@5\n(Qwen3-Embedding-0.6B, substring match)",
              "% queries", "#9467bd")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True,
                        help="Path to metrics JSON (flat {eval_key/metric: value} dict)")
    parser.add_argument("--title", default="Corpus Evals — single run")
    parser.add_argument("--out", default="results/corpus_evals_single_run.png")
    args = parser.parse_args()

    with open(args.metrics) as f:
        metrics = json.load(f)

    plot(metrics, args.title, args.out)


if __name__ == "__main__":
    main()
