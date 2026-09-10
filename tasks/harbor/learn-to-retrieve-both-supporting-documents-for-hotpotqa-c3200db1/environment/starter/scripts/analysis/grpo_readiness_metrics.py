"""Post-hoc GRPO-readiness diagnostic: pass@k + intra-group reward variance.

Reads an eval-results JSON already produced by eval_msa_hybrid_multisample.sh
(evals.msa_hybrid.eval.multi_sample=true) — samples already carry per-sample
llm_judge_accuracy/llm_judge_score (written by evals/shared.py::run_metrics_pipeline as part
of the normal eval.py flow) plus group_id/sample_idx (written by
GenLargeMemRagHybridEvaluator's multi_sample path). This script does NOT call the judge or the
model — it only re-groups already-judged samples and computes statistics over each group.

Why these two numbers: pass@k is the standard "does correctness improve if the policy gets k
tries" measure (Chen et al. 2021, unbiased estimator so one n-sample group answers pass@k for
every k <= n from a single generation round). Intra-group reward variance is what determines
whether GRPO's group-normalized advantage is nonzero for a given prompt — a group where every
sample gets the same reward (all-correct OR all-incorrect) contributes exactly zero gradient
signal under GRPO; only groups with a mix of correct/incorrect samples do.

Usage:
  uv run python scripts/analysis/grpo_readiness_metrics.py <path> [--out summary.json]

<path> is either a local file or a gs:// URI (fetched via `gsutil cat`, using GCS_USER_EMAIL
from the environment/.env for the account, matching scripts/misc/read_eval_samples.py).
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np


def load_results(path):
    if path.startswith("gs://"):
        env = {**os.environ, "CLOUDSDK_CORE_ACCOUNT": os.environ.get(
            "CLOUDSDK_CORE_ACCOUNT", os.environ.get("GCS_USER_EMAIL") or "rohunagrawal@gmail.com")}
        raw = subprocess.run(["gsutil", "cat", path], capture_output=True, text=True, env=env)
        if raw.returncode != 0:
            raise RuntimeError(f"gsutil cat {path} failed: {raw.stderr}")
        return json.loads(raw.stdout)
    with open(path) as f:
        return json.load(f)


def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al. 2021): P(>=1 correct among k drawn w/o replacement from n)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def group_samples(samples):
    groups = defaultdict(list)
    for s in samples:
        groups[s.get("group_id", id(s))].append(s)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None, help="write the summary JSON here too")
    args = ap.parse_args()

    data = load_results(args.path)
    samples = data.get("samples", [])
    if not samples:
        print("No samples found — nothing to compute.", file=sys.stderr)
        sys.exit(1)

    groups = group_samples(samples)
    group_sizes = sorted(set(len(v) for v in groups.values()))
    if group_sizes == [1]:
        print("WARNING: every group has exactly 1 sample — this JSON doesn't look like a "
              "multi_sample run (group_id/sample_idx present but no repeats). pass@k and "
              "intra-group variance are undefined for singleton groups.", file=sys.stderr)

    n_groups = len(groups)
    min_n = min(group_sizes)
    ks = list(range(1, min_n + 1))

    per_group_pass_at_k = {k: [] for k in ks}
    bin_variances, score_variances = [], []
    n_all_correct, n_all_incorrect, n_mixed = 0, 0, 0
    missing_judge = 0

    for gid, members in groups.items():
        correct = [m.get("llm_judge_accuracy") for m in members]
        scores = [m.get("llm_judge_score") for m in members]
        if any(c is None for c in correct):
            missing_judge += 1
            continue
        n = len(correct)
        c = int(sum(correct))
        for k in ks:
            per_group_pass_at_k[k].append(pass_at_k(n, c, k))
        bin_variances.append(float(np.var(correct)))
        if c == n:
            n_all_correct += 1
        elif c == 0:
            n_all_incorrect += 1
        else:
            n_mixed += 1
        if all(s is not None for s in scores):
            score_variances.append(float(np.var(scores)))

    scored_groups = n_groups - missing_judge
    summary = {
        "source": args.path,
        "n_groups": n_groups,
        "group_sizes_seen": group_sizes,
        "groups_missing_judge_score": missing_judge,
        "pass_at_k": {f"pass@{k}": float(np.mean(v)) for k, v in per_group_pass_at_k.items() if v},
        "mean_intra_group_binary_variance": float(np.mean(bin_variances)) if bin_variances else None,
        "mean_intra_group_score_variance": float(np.mean(score_variances)) if score_variances else None,
        "group_reward_composition": {
            "all_correct_frac": n_all_correct / scored_groups if scored_groups else None,
            "all_incorrect_frac": n_all_incorrect / scored_groups if scored_groups else None,
            "mixed_frac": n_mixed / scored_groups if scored_groups else None,
        },
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
