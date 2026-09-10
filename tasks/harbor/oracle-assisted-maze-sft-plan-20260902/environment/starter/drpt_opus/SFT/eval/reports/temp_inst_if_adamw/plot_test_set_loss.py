#!/usr/bin/env python3
"""Plot benchmark-reference loss against benchmark accuracy.

Reads test_set_task_loss.json (written by SFT/eval/test_set_task_loss.py) and the
accuracy already collected from the campaign, then puts them side by side per
setting. The point of the pairing: loss is teacher-forced, accuracy needs the
model to generate a parseable answer, so a method that sits low on loss and low
on accuracy is failing at generation rather than at the task.

Run after the measurement job finishes:  python plot_test_set_loss.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path(
    "/home/nakyungl/drpt-next/Dr.Post-Training-Next/drpt_opus/SFT/runs/campaigns/"
    "dolci32k-qwen3_1_7b-s42"
)
LOSS_JSON = HERE / "test_set_task_loss.json"

SETTINGS = ("reason_math", "reason_code", "mixed_math")
METHODS = ("FullTraining", "LayerwiseRaw", "LayerwiseSoft", "LayerwiseSoftP", "LayerwiseOptA")
EXTRA = "TargetOnly"
BENCH_KEY = {
    "reason_math": ("math500_results.json", "accuracy"),
    "mixed_math": ("math500_results.json", "accuracy"),
    "reason_code": ("mbpp_plus_results.json", "plus_pass_at_1"),
}
COLORS = {
    "FullTraining": "#2a78d6",
    "LayerwiseRaw": "#eb6834",
    "LayerwiseSoft": "#1baf7a",
    "LayerwiseSoftP": "#eda100",
    "LayerwiseOptA": "#e87ba4",
    EXTRA: "#4a3aa7",
}
MARKERS = {
    "FullTraining": "o", "LayerwiseRaw": "s", "LayerwiseSoft": "^",
    "LayerwiseSoftP": "D", "LayerwiseOptA": "v", EXTRA: "P",
}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def accuracy(setting: str, method: str):
    fname, key = BENCH_KEY[setting]
    hits = glob.glob(str(CAMPAIGN / f"{setting}-{method}-adamw-*" / fname))
    if not hits:
        return None
    return json.loads(Path(hits[0]).read_text()).get(key)


def main() -> int:
    if not LOSS_JSON.exists():
        raise SystemExit(f"{LOSS_JSON} not found -- run the measurement job first")
    loss = json.loads(LOSS_JSON.read_text())

    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(5.6 * len(SETTINGS), 4.8),
                             facecolor=SURFACE, squeeze=False)
    rows = []
    for ax, setting in zip(axes[0], SETTINGS):
        ax.set_facecolor(SURFACE)
        for idx, method in enumerate(list(METHODS) + [EXTRA]):
            rec = loss.get(f"{setting}|{method}") or loss.get(f"{setting}|{setting}:{method}")
            if not rec:
                continue
            acc = accuracy(setting, method)
            l = rec["test_task_loss"]
            rows.append((setting, method, l, acc))
            if acc is None:
                continue
            ax.scatter([l], [acc], s=95, color=COLORS[method], marker=MARKERS[method],
                       edgecolor=SURFACE, linewidth=1.1, zorder=3, label=method)
            ax.annotate(method.replace("Layerwise", "L."), xy=(l, acc),
                        xytext=(7, 5 if idx % 2 == 0 else -12), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_title(setting, fontsize=12, color=INK, loc="left", pad=10)
        ax.set_xlabel("teacher-forced loss on benchmark references  (lower = fits task)",
                      fontsize=8.5, color=MUTED)
        ax.set_ylabel("benchmark accuracy", fontsize=9.5, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5)

    fig.suptitle("Does the benchmark loss explain the benchmark accuracy?",
                 fontsize=14, color=INK, x=0.007, ha="left", y=0.985)
    fig.text(0.007, 0.925,
             "Bottom-left = fits the reference solutions but cannot turn that into a "
             "scored answer (a generation problem). Right side = a real capability gap.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    png = HERE / "test_set_loss_vs_accuracy.png"
    fig.savefig(png, dpi=150, facecolor=SURFACE)

    md = ["# Benchmark-reference loss vs benchmark accuracy\n",
          "| setting | method | test loss | accuracy |", "|---|---|---:|---:|"]
    for setting, method, l, acc in rows:
        md.append(f"| {setting} | {method} | {l:.4f} | "
                  + (f"{acc:.2f} |" if acc is not None else "- |"))
    (HERE / "test_set_loss_report.md").write_text("\n".join(md) + "\n")
    print(f"wrote {png}")
    print(f"wrote {HERE/'test_set_loss_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
