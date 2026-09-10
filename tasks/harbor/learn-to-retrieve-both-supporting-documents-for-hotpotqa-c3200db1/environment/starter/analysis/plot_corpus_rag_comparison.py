"""
Plot corpus eval comparison across three checkpoints: memory layers vs RAG.

Reads three metrics JSON files from --metrics_dir:
  metrics_base.json  — pretrained (step 100000)
  metrics_128k.json  — two-pass 128K mem (step 102250)
  metrics_256k.json  — two-pass 256K mem (step 101000)

Produces:
  results/corpus_rag_three_checkpoints.png  — 3×2 grouped bar chart
  results/corpus_rag_three_checkpoints.md   — comparison table + embed link

Usage:
    python analysis/plot_corpus_rag_comparison.py \
        --metrics_dir outputs/corpus_rag_three_checkpoints \
        --results_dir results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHECKPOINTS = [
    ("base",  "Pretrained\n(step 100k)"),
    ("128k",  "Two-Pass 128K\n(step 102k)"),
    ("256k",  "Two-Pass 256K\n(step 101k)"),
]

DATASETS = [
    ("gen_large_mem_musique",  "MuSiQue"),
    ("gen_large_mem_hotpotqa", "HotpotQA"),
    ("gen_large_mem_msmarco",  "MS MARCO"),
]

# Colours
C_MEM = "#1f77b4"   # blue  — memory layers judge acc
C_RAG = "#ff7f0e"   # orange — RAG judge acc
C_RET = "#2ca02c"   # green  — doc_access_acc
C_RCL = "#d62728"   # red    — RAG recall@5


def load(metrics_dir, label):
    path = os.path.join(metrics_dir, f"metrics_{label}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path) as f:
        return json.load(f)


def get(metrics, eval_key, metric, default=None):
    return metrics.get(f"{eval_key}/{metric}", default)


def _bar_panel(ax, all_metrics, metric_key, title, ylabel, scale=100,
               fmt=".1f", pad_frac=1.2, min_ylim=0.1, default=0):
    """Generic grouped bar panel: one group per checkpoint, one bar per dataset."""
    x = np.arange(len(CHECKPOINTS))
    w = 0.35
    ckpt_labels = [label for _, label in CHECKPOINTS]
    for i, (ds_key, ds_label) in enumerate(DATASETS):
        vals = [(get(all_metrics[ck], ds_key, metric_key) or default) * scale
                for ck, _ in CHECKPOINTS]
        offset = (i - 1) * w * 0.9
        bars = ax.bar(x + offset, vals, w * 0.85, label=ds_label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + scale * 0.003,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(ckpt_labels, fontsize=9)
    ax.legend(fontsize=9)
    ylim_top = ax.get_ylim()[1]
    ax.set_ylim(0, max(min_ylim, ylim_top * pad_frac))
    ax.grid(axis="y", alpha=0.3)


def plot(all_metrics, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Corpus Evals: Memory Layers vs RAG — Three Checkpoints", fontsize=14)

    x = np.arange(len(CHECKPOINTS))
    w = 0.35
    ckpt_labels = [label for _, label in CHECKPOINTS]

    # ── Panel A: Memory Layers LLM judge accuracy ─────────────────────────────
    _bar_panel(axes[0, 0], all_metrics,
               "llm_judge_accuracy", "Memory Layers — LLM Judge Accuracy",
               "Judge accuracy (%)", scale=100, fmt=".1f", pad_frac=1.2,
               min_ylim=5)

    # ── Panel B: RAG LLM judge accuracy ──────────────────────────────────────
    _bar_panel(axes[0, 1], all_metrics,
               "rag_accuracy", "RAG (Qwen3-4B) — LLM Judge Accuracy",
               "Judge accuracy (%)", scale=100, fmt=".1f", pad_frac=1.2,
               min_ylim=5)

    # ── Panel C: Memory vs RAG delta ──────────────────────────────────────────
    ax = axes[1, 0]
    for i, (ds_key, ds_label) in enumerate(DATASETS):
        deltas = [(get(all_metrics[ck], ds_key, "llm_judge_accuracy", 0) -
                   get(all_metrics[ck], ds_key, "rag_accuracy", 0)) * 100
                  for ck, _ in CHECKPOINTS]
        offset = (i - 1) * w * 0.9
        colors = [C_MEM if d >= 0 else C_RAG for d in deltas]
        bars = ax.bar(x + offset, deltas, w * 0.85, label=ds_label, color=colors, alpha=0.8)
        for bar, v in zip(bars, deltas):
            va = "bottom" if v >= 0 else "top"
            y = max(v, 0) if v >= 0 else min(v, 0)
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{v:+.1f}", ha="center", va=va, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Mem Layers − RAG (judge acc Δ)", fontsize=11)
    ax.set_ylabel("Δ judge accuracy (pp)\n+ve = memory leads")
    ax.set_xticks(x)
    ax.set_xticklabels(ckpt_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel D: RAG recall@5 and MRR (checkpoint-independent) ───────────────
    # Use base checkpoint values; retrieval model is the same across runs.
    ax = axes[1, 1]
    ds_labels = [l for _, l in DATASETS]
    xd = np.arange(len(DATASETS))
    wd = 0.35
    r5_vals  = [(get(all_metrics["base"], ds_key, "rag_recall@5") or 0) * 100
                for ds_key, _ in DATASETS]
    mrr_vals = [(get(all_metrics["base"], ds_key, "rag_mrr") or 0) * 100
                for ds_key, _ in DATASETS]
    bars1 = ax.bar(xd - wd / 2, r5_vals,  wd, label="Recall@5", color="#2ca02c")
    bars2 = ax.bar(xd + wd / 2, mrr_vals, wd, label="MRR",      color="#9467bd")
    for bars, vals in [(bars1, r5_vals), (bars2, mrr_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("RAG Retrieval — Recall@5 & MRR\n(Qwen3-Embedding-0.6B, checkpoint-independent)",
                 fontsize=10)
    ax.set_ylabel("(%)")
    ax.set_xticks(xd)
    ax.set_xticklabels(ds_labels, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(5, ax.get_ylim()[1] * 1.25))
    ax.grid(axis="y", alpha=0.3)

    # ── Panel E: doc_hit_rate (example-level recall) ──────────────────────────
    has_hit_rate = any(
        get(all_metrics[ck], ds_key, "doc_hit_rate") is not None
        for ck, _ in CHECKPOINTS for ds_key, _ in DATASETS
    )
    _bar_panel(axes[2, 0], all_metrics,
               "doc_hit_rate",
               "Memory Layers — Example-Level Recall\n(any lookup hit correct doc)",
               "Fraction of examples (%)", scale=100, fmt=".1f", pad_frac=1.2,
               min_ylim=5)
    if not has_hit_rate:
        axes[2, 0].text(0.5, 0.5, "Re-run evals to populate\ndoc_hit_rate",
                        transform=axes[2, 0].transAxes,
                        ha="center", va="center", fontsize=10, color="gray",
                        style="italic")

    # ── Panel F: doc_hit_rate_by_position ─────────────────────────────────────
    has_hit_by_pos = any(
        get(all_metrics[ck], ds_key, "doc_hit_rate_by_position") is not None
        for ck, _ in CHECKPOINTS for ds_key, _ in DATASETS
    )
    _bar_panel(axes[2, 1], all_metrics,
               "doc_hit_rate_by_position",
               "Memory Layers — Position-Level Recall\n(any head/k hit correct doc at this position)",
               "Fraction of positions (%)", scale=100, fmt=".1f", pad_frac=1.2,
               min_ylim=5)
    if not has_hit_by_pos:
        axes[2, 1].text(0.5, 0.5, "Re-run evals to populate\ndoc_hit_rate_by_position",
                        transform=axes[2, 1].transAxes,
                        ha="center", va="center", fontsize=10, color="gray",
                        style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart → {out_path}")
    plt.close()


def write_markdown(all_metrics, png_name, out_path):
    lines = [
        "# Corpus Evals: Memory Layers vs RAG — Three Checkpoints",
        "",
        "| Model | Checkpoint | Step |",
        "|-------|-----------|------|",
        "| **Pretrained** | `4B_pretraining_cot_unfreeze_all_topk_128_norm` | 100 000 |",
        "| **Two-Pass 128K** | `4B_pretraining_two_pass-2026-04-17` | 102 250 |",
        "| **Two-Pass 256K** | `4B_pretraining_two_pass-2026-04-18` | 101 000 |",
        "",
        "n = 128 questions per dataset. "
        "Memory bank embeds full corpus (lookup_chunk_size=8192). "
        "RAG: Qwen3-Embedding-0.6B top-5 retrieval + Qwen3-4B generation.",
        "",
        f"![Corpus RAG comparison]({png_name})",
        "",
        "---",
        "",
        "## Generation Quality (LLM Judge)",
        "",
        "### Memory Layers",
        "",
        "| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |",
        "|---------|-----------|---------------|---------------|",
    ]
    for ds_key, ds_label in DATASETS:
        row = [ds_label]
        for ck, _ in CHECKPOINTS:
            v = get(all_metrics[ck], ds_key, "llm_judge_accuracy")
            row.append(f"{v:.4f}" if v is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### RAG (Qwen3-Embedding-0.6B + Qwen3-4B)",
        "",
        "| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |",
        "|---------|-----------|---------------|---------------|",
    ]
    for ds_key, ds_label in DATASETS:
        row = [ds_label]
        for ck, _ in CHECKPOINTS:
            v = get(all_metrics[ck], ds_key, "rag_accuracy")
            row.append(f"{v:.4f}" if v is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### Delta (Memory − RAG)",
        "",
        "| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |",
        "|---------|-----------|---------------|---------------|",
    ]
    for ds_key, ds_label in DATASETS:
        row = [ds_label]
        for ck, _ in CHECKPOINTS:
            mem = get(all_metrics[ck], ds_key, "llm_judge_accuracy", 0)
            rag = get(all_metrics[ck], ds_key, "rag_accuracy", 0)
            delta = mem - rag
            sign = "+" if delta >= 0 else ""
            row.append(f"{sign}{delta:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "---",
        "",
        "## Retrieval Quality",
        "",
        "### doc_access_acc — fraction of all (head × pos × top-k) slots hitting correct doc",
        "",
        "| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |",
        "|---------|-----------|---------------|---------------|",
    ]
    for ds_key, ds_label in DATASETS:
        row = [ds_label]
        for ck, _ in CHECKPOINTS:
            v = get(all_metrics[ck], ds_key, "doc_access_acc")
            row.append(f"{v:.4f}" if v is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### doc_hit_rate — fraction of examples where any lookup hit the correct doc",
        "",
        "| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |",
        "|---------|-----------|---------------|---------------|",
    ]
    for ds_key, ds_label in DATASETS:
        row = [ds_label]
        for ck, _ in CHECKPOINTS:
            v = get(all_metrics[ck], ds_key, "doc_hit_rate")
            row.append(f"{v:.4f}" if v is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### doc_hit_rate_by_position — fraction of active positions with any hit",
        "",
        "| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |",
        "|---------|-----------|---------------|---------------|",
    ]
    for ds_key, ds_label in DATASETS:
        row = [ds_label]
        for ck, _ in CHECKPOINTS:
            v = get(all_metrics[ck], ds_key, "doc_hit_rate_by_position")
            row.append(f"{v:.4f}" if v is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### RAG recall@5 / MRR (Qwen3-Embedding-0.6B, checkpoint-independent)",
        "",
        "| Dataset | recall@1 | recall@5 | MRR |",
        "|---------|---------|---------|-----|",
        "*(Same retrieval model for all checkpoints — values from base run)*",
        "",
    ]
    ck = "base"
    for ds_key, ds_label in DATASETS:
        r1  = get(all_metrics[ck], ds_key, "rag_recall@1",  "N/A")
        r5  = get(all_metrics[ck], ds_key, "rag_recall@5",  "N/A")
        mrr = get(all_metrics[ck], ds_key, "rag_mrr",       "N/A")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| {ds_label} | {fmt(r1)} | {fmt(r5)} | {fmt(mrr)} |")

    lines += [
        "",
        "---",
        "",
        "## All Metrics (JSON)",
        "",
    ]
    for ck, ck_label in CHECKPOINTS:
        ck_label_clean = ck_label.replace("\n", " ")
        lines.append(f"### {ck_label_clean}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(all_metrics[ck], indent=2))
        lines.append("```")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved markdown → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True,
                        help="Directory containing metrics_base.json, metrics_128k.json, metrics_256k.json")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    all_metrics = {}
    for ck, _ in CHECKPOINTS:
        all_metrics[ck] = load(args.metrics_dir, ck)
        print(f"Loaded metrics for '{ck}'")

    os.makedirs(args.results_dir, exist_ok=True)

    png_name = "corpus_rag_three_checkpoints.png"
    png_path = os.path.join(args.results_dir, png_name)
    md_path  = os.path.join(args.results_dir, "corpus_rag_three_checkpoints.md")

    plot(all_metrics, png_path)
    write_markdown(all_metrics, png_name, md_path)

    # Print a quick summary table
    print("\n── Generation Quality (LLM Judge) ──")
    header = f"{'Dataset':<18}" + "".join(f"  {ck:>14}" for ck, _ in CHECKPOINTS)
    print(header)
    for ds_key, ds_label in DATASETS:
        row = f"{ds_label:<18}"
        for ck, _ in CHECKPOINTS:
            mem = get(all_metrics[ck], ds_key, "llm_judge_accuracy")
            rag = get(all_metrics[ck], ds_key, "rag_accuracy")
            mem_s = f"{mem*100:.1f}%" if mem is not None else "N/A"
            rag_s = f"{rag*100:.1f}%" if rag is not None else "N/A"
            row += f"  {mem_s:>6}/RAG {rag_s:>6}"
        print(row)

    print("\n── doc_access_acc ──")
    print(header)
    for ds_key, ds_label in DATASETS:
        row = f"{ds_label:<18}"
        for ck, _ in CHECKPOINTS:
            v = get(all_metrics[ck], ds_key, "doc_access_acc")
            row += f"  {v*100:.3f}%" if v is not None else "  N/A"
        print(row)


if __name__ == "__main__":
    main()
