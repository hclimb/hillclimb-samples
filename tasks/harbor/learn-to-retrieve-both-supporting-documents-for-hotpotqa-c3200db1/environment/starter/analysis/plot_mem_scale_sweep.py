"""
Plot memory-scale sweep: memory layer tokens vs LLM judge accuracy.

Reads gen_embed_results.json from each eval key directory under the given
output root and produces a line chart per dataset.

Usage:
    python analysis/plot_mem_scale_sweep.py --output_dir outputs/2026-04-16/... \
        [--results_dir results/] [--step 100000]
"""
import argparse
import json
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Map eval-key suffix → nominal memory token count
_TOKEN_LABEL_MAP = {
    "32k":  32_768,
    "64k":  65_536,
    "128k": 131_072,
    "256k": 262_144,
    "512k": 524_288,
    "1M":   1_048_576,
}

# Dataset display names and their per-example token counts
# (num_chunks_per_doc × doc_chunk_seq_len)
_DATASETS = {
    "msmarco": {
        "label": "MS MARCO QA",
        "tokens_per_example": 4 * 256,  # 1024
        "color": "#1f77b4",
        "marker": "o",
    },
    "hotpotqa": {
        "label": "HotpotQA",
        "tokens_per_example": 16 * 256,  # 4096
        "color": "#ff7f0e",
        "marker": "s",
    },
}


def find_latest_output_dir(base="outputs"):
    """Return the most recently modified eval_results dir under base/."""
    candidates = sorted(
        glob.glob(os.path.join(base, "*", "*")),
        key=os.path.getmtime,
        reverse=True,
    )
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def load_metrics(output_dir, step=None):
    """
    Walk output_dir looking for gen_embed_{dataset}_{scale}/gen_embed_results.json.
    Returns {dataset: {token_count: accuracy}}.
    """
    # Step subdir is optional — check both patterns
    search_roots = []
    if step is not None:
        search_roots.append(os.path.join(output_dir, "eval_results", f"step_{step}"))
    # Also try direct children (no step subdir)
    search_roots.append(os.path.join(output_dir, "eval_results"))
    search_roots.append(output_dir)

    results = {ds: {} for ds in _DATASETS}

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            for ds_key in _DATASETS:
                for scale_key, mem_tokens in _TOKEN_LABEL_MAP.items():
                    expected = f"gen_embed_{ds_key}_{scale_key}"
                    if entry == expected:
                        json_path = os.path.join(root, entry, "gen_embed_results.json")
                        if not os.path.exists(json_path):
                            continue
                        with open(json_path) as f:
                            data = json.load(f)
                        acc = data.get("metrics", {}).get("llm_judge_accuracy")
                        if acc is not None:
                            results[ds_key][mem_tokens] = acc
    return results


def plot(results, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    has_any = False
    for ds_key, info in _DATASETS.items():
        pts = sorted(results[ds_key].items())  # [(tokens, acc), ...]
        if not pts:
            continue
        has_any = True
        xs, ys = zip(*pts)
        xs_k = [x / 1024 for x in xs]  # display in K tokens
        ax.plot(
            xs_k, [y * 100 for y in ys],
            marker=info["marker"],
            color=info["color"],
            label=info["label"],
            linewidth=2,
            markersize=7,
        )
        for x, y in zip(xs_k, ys):
            ax.annotate(
                f"{y*100:.1f}%",
                (x, y * 100),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
                color=info["color"],
            )

    if not has_any:
        print("No results found — nothing to plot.")
        return

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Memory layer tokens", fontsize=11)
    ax.set_ylabel("LLM judge accuracy (%)", fontsize=11)
    ax.set_title("Memory Scale vs Accuracy (gen_embed)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    # Label each power-of-2 tick with a human-readable string
    # x values are already in K (divided by 1024 before plotting)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: ("1M" if x >= 1024 else f"{int(round(x))}K")
    ))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None,
                        help="Eval output directory (e.g. outputs/2026-04-16/12-00-00). "
                             "Defaults to the most recently modified outputs/ subdir.")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (used to find the step_XXXXXX subdir).")
    parser.add_argument("--results_dir", default="results",
                        help="Where to write the output chart.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = find_latest_output_dir()
        if output_dir is None:
            raise RuntimeError("Could not find any output directory under outputs/")
        print(f"Using latest output dir: {output_dir}")

    results = load_metrics(output_dir, step=args.step)

    # Print a summary table
    all_tokens = sorted({t for ds in results.values() for t in ds})
    print(f"\n{'Eval key':<20}  " + "  ".join(f"{t//1024:>6}K" for t in all_tokens))
    print("-" * (22 + 9 * len(all_tokens)))
    for ds_key, info in _DATASETS.items():
        row = f"{info['label']:<20}  "
        row += "  ".join(
            f"{results[ds_key].get(t, float('nan'))*100:>6.1f}%"
            if t in results[ds_key] else f"{'N/A':>7}"
            for t in all_tokens
        )
        print(row)
    print()

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "mem_scale_sweep.png")
    plot(results, out_path)


if __name__ == "__main__":
    main()
