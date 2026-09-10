"""
Compile two-pass memory comparison results into a markdown table and plot.

Reads gen_embed_results.json files from two eval output directories and writes:
  - <out_dir>/two_pass_comparison.md  — table with judge acc + retrieval acc
  - <out_dir>/mem_scale_sweep.png     — line chart comparing both checkpoints

Usage:
    python analysis/write_two_pass_md.py \
        --dir_a outputs/two_pass_eval/256k_mem \
        --dir_b outputs/two_pass_eval/128k_mem \
        --label_a "256k mem" \
        --label_b "128k mem" \
        --step_a 101000 \
        --step_b 102250 \
        --out_dir results/two_pass_comparison
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_SCALES = ["32k", "256k", "512k", "1M"]
_TOKEN_MAP = {"32k": 32_768, "256k": 262_144, "512k": 524_288, "1M": 1_048_576}
_DATASETS = {
    "msmarco":  "MS MARCO QA",
    "hotpotqa": "HotpotQA",
}
_DS_COLORS = {"msmarco": "#1f77b4", "hotpotqa": "#ff7f0e"}
_DS_MARKERS = {"msmarco": "o", "hotpotqa": "s"}


def load_metrics(output_dir, step):
    """Return {ds: {scale: {metric: value}}} from eval output dir."""
    root = os.path.join(output_dir, "eval_results", f"step_{step}")
    if not os.path.isdir(root):
        root = os.path.join(output_dir, "eval_results")

    data = {ds: {} for ds in _DATASETS}
    for ds_key in _DATASETS:
        for scale in _SCALES:
            path = os.path.join(root, f"gen_embed_{ds_key}_{scale}", "gen_embed_results.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                j = json.load(f)
            metrics = j.get("metrics", {})
            data[ds_key][scale] = {
                "llm_judge_accuracy": metrics.get("llm_judge_accuracy"),
                "doc_access_acc":     metrics.get("doc_access_acc"),
            }
    return data


def fmt(v):
    return "N/A" if v is None else f"{v * 100:.1f}%"


def write_md(data_a, data_b, label_a, label_b, step_a, step_b, out_dir):
    lines = [
        "# Two-Pass Memory Training: Scale Comparison",
        "",
        f"Checkpoints: **{label_a}** (step {step_a}) vs **{label_b}** (step {step_b})",
        "Eval: gen_embed, 32k–256k memory tokens, 256 samples each.",
        "",
        f"| ![{label_a}](mem_scale_sweep_{label_a.lower().replace(' ', '_')}.png) | ![{label_b}](mem_scale_sweep_{label_b.lower().replace(' ', '_')}.png) |",
        f"|:---:|:---:|",
        f"| {label_a} | {label_b} |",
        "",
    ]

    for ds_key, ds_label in _DATASETS.items():
        lines += [
            f"## {ds_label}",
            "",
            f"| Mem tokens | {label_a} judge acc | {label_a} retrieval acc | {label_b} judge acc | {label_b} retrieval acc |",
            f"|:----------:|:-------------------:|:----------------------:|:-------------------:|:----------------------:|",
        ]
        for scale in _SCALES:
            a = data_a[ds_key].get(scale, {})
            b = data_b[ds_key].get(scale, {})
            lines.append(
                f"| {scale:>10} "
                f"| {fmt(a.get('llm_judge_accuracy')):>19} "
                f"| {fmt(a.get('doc_access_acc')):>22} "
                f"| {fmt(b.get('llm_judge_accuracy')):>19} "
                f"| {fmt(b.get('doc_access_acc')):>22} |"
            )
        lines.append("")

    out_path = os.path.join(out_dir, "two_pass_comparison.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def plot_single(data, label, out_path):
    """One chart per checkpoint: MS MARCO and HotpotQA judge acc vs memory tokens."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ds_key, ds_label in _DATASETS.items():
        pts = [
            (_TOKEN_MAP[s], data[ds_key][s]["llm_judge_accuracy"])
            for s in _SCALES
            if s in data[ds_key] and data[ds_key][s].get("llm_judge_accuracy") is not None
        ]
        if not pts:
            continue
        xs, ys = zip(*pts)
        xs_k = [x / 1024 for x in xs]
        ax.plot(
            xs_k, [y * 100 for y in ys],
            marker=_DS_MARKERS[ds_key], color=_DS_COLORS[ds_key],
            label=ds_label, linewidth=2, markersize=7,
        )
        for x, y in zip(xs_k, ys):
            ax.annotate(f"{y*100:.1f}%", (x, y * 100),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=8, color=_DS_COLORS[ds_key])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Memory layer tokens", fontsize=11)
    ax.set_ylabel("LLM judge accuracy (%)", fontsize=11)
    ax.set_title(f"Memory Scale vs Accuracy — {label}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "1M" if x >= 1024 else f"{int(round(x))}K"
    ))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart → {out_path}")
    plt.close()


def plot(data_a, data_b, label_a, label_b, out_dir):
    slug_a = label_a.lower().replace(" ", "_")
    slug_b = label_b.lower().replace(" ", "_")
    plot_single(data_a, label_a, os.path.join(out_dir, f"mem_scale_sweep_{slug_a}.png"))
    plot_single(data_b, label_b, os.path.join(out_dir, f"mem_scale_sweep_{slug_b}.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir_a",    required=True)
    p.add_argument("--dir_b",    required=True)
    p.add_argument("--label_a",  default="256k mem")
    p.add_argument("--label_b",  default="128k mem")
    p.add_argument("--step_a",   type=int, required=True)
    p.add_argument("--step_b",   type=int, required=True)
    p.add_argument("--out_dir",  default="results/two_pass_comparison")
    args = p.parse_args()

    data_a = load_metrics(args.dir_a, args.step_a)
    data_b = load_metrics(args.dir_b, args.step_b)

    for ds_key, ds_label in _DATASETS.items():
        print(f"\n{ds_label}")
        header = f"{'Scale':>8}  {args.label_a:>24}  {args.label_b:>24}"
        print(header)
        print(f"{'':>8}  {'judge':>12} {'retrieval':>10}  {'judge':>12} {'retrieval':>10}")
        print("-" * len(header))
        for scale in _SCALES:
            a = data_a[ds_key].get(scale, {})
            b = data_b[ds_key].get(scale, {})
            print(
                f"{scale:>8}  "
                f"{fmt(a.get('llm_judge_accuracy')):>12} {fmt(a.get('doc_access_acc')):>10}  "
                f"{fmt(b.get('llm_judge_accuracy')):>12} {fmt(b.get('doc_access_acc')):>10}"
            )

    os.makedirs(args.out_dir, exist_ok=True)
    write_md(data_a, data_b, args.label_a, args.label_b, args.step_a, args.step_b, args.out_dir)
    plot(data_a, data_b, args.label_a, args.label_b, args.out_dir)


if __name__ == "__main__":
    main()
