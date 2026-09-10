"""
Plot doc_token_hit_rate histograms split by LLM-judge correctness.

Reads one or more saved generation result JSONs (the files with
{"metrics": ..., "samples": [...]}) and produces a multi-panel figure with
one histogram per dataset. Each panel overlays the doc_token_hit_rate
distribution for:
  - samples where llm_judge_accuracy == 1.0
  - samples where llm_judge_accuracy == 0.0

Usage:
    uv run python analysis/plot_correct_incorrect_token_hit_histograms.py \
        --results outputs/.../msa_musique.json \
        --out results/musique_doc_token_hit_hist.png

    uv run python analysis/plot_correct_incorrect_token_hit_histograms.py \
        --results outputs/.../msa_musique.json outputs/.../msa_hotpotqa.json \
        --labels MuSiQue HotpotQA \
        --out results/doc_token_hit_histograms.png
"""

import argparse
import json
import math
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


def values_for_subset(samples, metric, judge_value):
    return [
        float(s[metric])
        for s in samples
        if s.get("llm_judge_accuracy") == judge_value and s.get(metric) is not None
    ]


def summarize_result(path_str: str, label: str) -> dict:
    with open(path_str) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    correct_vals = values_for_subset(samples, "doc_token_hit_rate", 1.0)
    incorrect_vals = values_for_subset(samples, "doc_token_hit_rate", 0.0)

    return {
        "label": label,
        "path": path_str,
        "n_total": len(samples),
        "n_correct": len(correct_vals),
        "n_incorrect": len(incorrect_vals),
        "correct_doc_token_hit_rates": correct_vals,
        "incorrect_doc_token_hit_rates": incorrect_vals,
        "correct_mean": float(np.mean(correct_vals)) if correct_vals else None,
        "incorrect_mean": float(np.mean(incorrect_vals)) if incorrect_vals else None,
    }


def _panel_title(summary: dict) -> str:
    return (
        f"{summary['label']}\n"
        f"correct={summary['n_correct']}  incorrect={summary['n_incorrect']}"
    )


def plot_histograms(summaries, title, out_path, bins, density):
    n = len(summaries)
    ncols = min(3, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    correct_color = "#2ca02c"
    incorrect_color = "#d62728"

    for ax, summary in zip(axes.flat, summaries):
        correct_vals = summary["correct_doc_token_hit_rates"]
        incorrect_vals = summary["incorrect_doc_token_hit_rates"]

        if correct_vals:
            ax.hist(
                correct_vals,
                bins=bin_edges,
                density=density,
                alpha=0.55,
                color=correct_color,
                label="Judge correct",
            )
            ax.axvline(summary["correct_mean"], color=correct_color, linestyle="--", linewidth=1.5)

        if incorrect_vals:
            ax.hist(
                incorrect_vals,
                bins=bin_edges,
                density=density,
                alpha=0.55,
                color=incorrect_color,
                label="Judge incorrect",
            )
            ax.axvline(summary["incorrect_mean"], color=incorrect_color, linestyle="--", linewidth=1.5)

        if not correct_vals and not incorrect_vals:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(_panel_title(summary), fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("doc_token_hit_rate", fontsize=9)
        ax.set_ylabel("Density" if density else "# samples", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    for ax in axes.flat[len(summaries):]:
        ax.axis("off")

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
    parser.add_argument("--title", default="doc_token_hit_rate by LLM-Judge Correctness")
    parser.add_argument("--out", default="results/correct_incorrect_doc_token_hit_histograms.png")
    parser.add_argument("--summary_out", default=None,
                        help="Optional path to write per-dataset histogram inputs as JSON.")
    parser.add_argument("--bins", type=int, default=20,
                        help="Number of fixed-width bins spanning [0, 1].")
    parser.add_argument("--count_scale", action="store_true",
                        help="Plot raw counts instead of density-normalized histograms.")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.results):
        raise ValueError("--labels must have the same length as --results")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    summaries = []
    for i, path_str in enumerate(args.results):
        label = args.labels[i] if args.labels else infer_label(path_str)
        summaries.append(summarize_result(path_str, label))

    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Saved summary → {args.summary_out}")

    plot_histograms(
        summaries,
        args.title,
        args.out,
        bins=args.bins,
        density=not args.count_scale,
    )


if __name__ == "__main__":
    main()
