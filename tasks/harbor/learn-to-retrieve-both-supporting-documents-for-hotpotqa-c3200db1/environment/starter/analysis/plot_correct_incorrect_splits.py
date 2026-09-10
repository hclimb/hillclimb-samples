"""
Plot memory-layer retrieval metrics split by LLM-judge correctness.

Reads one or more saved generation result JSONs (the files with
{"metrics": ..., "samples": [...]}) and produces a 2x2 figure:
  - correct vs incorrect sample counts
  - doc_access_acc split by correctness
  - doc_hit_rate split by correctness
  - doc_token_hit_rate split by correctness

Usage:
    uv run python analysis/plot_correct_incorrect_splits.py \
        --results outputs/.../msa_musique.json \
        --out results/musique_correct_incorrect.png

    uv run python analysis/plot_correct_incorrect_splits.py \
        --results outputs/.../msa_musique.json outputs/.../msa_hotpotqa.json \
        --labels MuSiQue HotpotQA \
        --out results/correct_incorrect_splits.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASET_LABELS = {
    "msmarco": "MS MARCO",
    "natural_questions": "NQ",
    "narrativeqa": "NarrativeQA",
    "2wikimultihopqa": "2Wiki",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "dureader": "DuReader",
    "popqa": "PopQA",
    "triviaqa": "TriviaQA",
}


def infer_label(path_str: str) -> str:
    lower = path_str.lower()
    for key, label in DATASET_LABELS.items():
        if key in lower:
            return label
    path = Path(path_str)
    if path.parent.name:
        return path.parent.name
    return path.stem


def mean_for_subset(samples, metric, judge_value):
    vals = [
        s.get(metric)
        for s in samples
        if s.get("llm_judge_accuracy") == judge_value and s.get(metric) is not None
    ]
    return float(np.mean(vals)) if vals else None


def summarize_result(path_str: str, label: str) -> dict:
    with open(path_str) as f:
        data = json.load(f)
    samples = data.get("samples", [])
    correct = [s for s in samples if s.get("llm_judge_accuracy") == 1.0]
    incorrect = [s for s in samples if s.get("llm_judge_accuracy") == 0.0]

    return {
        "label": label,
        "path": path_str,
        "n_total": len(samples),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "judge_acc": (len(correct) / len(samples)) if samples else None,
        "correct_doc_access_acc": mean_for_subset(samples, "doc_access_acc", 1.0),
        "incorrect_doc_access_acc": mean_for_subset(samples, "doc_access_acc", 0.0),
        "correct_doc_hit_rate": mean_for_subset(samples, "doc_hit_rate", 1.0),
        "incorrect_doc_hit_rate": mean_for_subset(samples, "doc_hit_rate", 0.0),
        "correct_doc_token_hit_rate": mean_for_subset(samples, "doc_token_hit_rate", 1.0),
        "incorrect_doc_token_hit_rate": mean_for_subset(samples, "doc_token_hit_rate", 0.0),
    }


def _percent_or_nan(v):
    return np.nan if v is None else v * 100.0


def _annotate_bars(ax, bars, vals, fmt=".1f"):
    for bar, val in zip(bars, vals):
        if val is None or np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.5,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="gray",
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:{fmt}}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _metric_panel(ax, summaries, correct_key, incorrect_key, title, ylabel, colors):
    labels = [s["label"] for s in summaries]
    x = np.arange(len(summaries))
    width = 0.35

    correct_vals = [_percent_or_nan(s[correct_key]) for s in summaries]
    incorrect_vals = [_percent_or_nan(s[incorrect_key]) for s in summaries]

    bars1 = ax.bar(x - width / 2, np.nan_to_num(correct_vals, nan=0.0), width,
                   label="Judge correct", color=colors[0], alpha=0.9)
    bars2 = ax.bar(x + width / 2, np.nan_to_num(incorrect_vals, nan=0.0), width,
                   label="Judge incorrect", color=colors[1], alpha=0.9)

    _annotate_bars(ax, bars1, correct_vals)
    _annotate_bars(ax, bars2, incorrect_vals)

    ymax = 5.0
    finite = [v for v in correct_vals + incorrect_vals if not np.isnan(v)]
    if finite:
        ymax = max(ymax, max(finite) * 1.25)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def plot(summaries, title, out_path):
    fig, axd = plt.subplot_mosaic(
        [
            ["counts", "access"],
            ["hit", "token_hit"],
        ],
        figsize=(14, 8),
    )
    fig.suptitle(title, fontsize=13)

    labels = [s["label"] for s in summaries]
    x = np.arange(len(summaries))

    # Counts panel
    correct_counts = [s["n_correct"] for s in summaries]
    incorrect_counts = [s["n_incorrect"] for s in summaries]
    bars_correct = axd["counts"].bar(x, correct_counts, color="#2ca02c", label="Judge correct")
    bars_incorrect = axd["counts"].bar(
        x, incorrect_counts, bottom=correct_counts, color="#d62728", label="Judge incorrect"
    )
    for i, s in enumerate(summaries):
        total = s["n_total"]
        judge_acc = s["judge_acc"] * 100 if s["judge_acc"] is not None else 0.0
        axd["counts"].text(
            x[i],
            correct_counts[i] + incorrect_counts[i] + max(1, total * 0.02),
            f"{judge_acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    axd["counts"].set_title("Sample Counts\n(top label = LLM judge accuracy)", fontsize=10)
    axd["counts"].set_ylabel("# samples", fontsize=9)
    axd["counts"].set_xticks(x)
    axd["counts"].set_xticklabels(labels, fontsize=10)
    axd["counts"].legend(fontsize=9)
    axd["counts"].grid(axis="y", alpha=0.3)

    _metric_panel(
        axd["access"],
        summaries,
        "correct_doc_access_acc",
        "incorrect_doc_access_acc",
        "doc_access_acc by LLM-Judge Correctness",
        "% slots",
        ("#1f77b4", "#aec7e8"),
    )
    _metric_panel(
        axd["hit"],
        summaries,
        "correct_doc_hit_rate",
        "incorrect_doc_hit_rate",
        "doc_hit_rate by LLM-Judge Correctness",
        "% examples",
        ("#ff7f0e", "#ffbb78"),
    )
    _metric_panel(
        axd["token_hit"],
        summaries,
        "correct_doc_token_hit_rate",
        "incorrect_doc_token_hit_rate",
        "doc_token_hit_rate by LLM-Judge Correctness",
        "% tokens",
        ("#9467bd", "#c5b0d5"),
    )
    axd["access"].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="One or more generation result JSONs with a top-level samples array.")
    parser.add_argument("--labels", nargs="*",
                        help="Optional display labels matching --results order.")
    parser.add_argument("--title", default="Correct vs Incorrect Retrieval Splits")
    parser.add_argument("--out", default="results/correct_incorrect_splits.png")
    parser.add_argument("--summary_out", default=None,
                        help="Optional path to write the computed split metrics as JSON.")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.results):
        raise ValueError("--labels must have the same length as --results")

    summaries = []
    for i, path_str in enumerate(args.results):
        label = args.labels[i] if args.labels else infer_label(path_str)
        summaries.append(summarize_result(path_str, label))

    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Saved summary → {args.summary_out}")

    plot(summaries, args.title, args.out)


if __name__ == "__main__":
    main()
