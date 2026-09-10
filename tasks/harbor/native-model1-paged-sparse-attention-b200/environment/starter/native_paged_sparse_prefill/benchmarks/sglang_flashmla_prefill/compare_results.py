#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LOWER_IS_BETTER = {
    "duration",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
}
HIGHER_IS_BETTER = {
    "request_throughput",
    "input_throughput",
    "total_throughput",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _delta(candidate: float, baseline: float, lower_is_better: bool) -> float:
    if baseline == 0:
        return float("nan")
    if lower_is_better:
        return (baseline - candidate) / baseline * 100.0
    return (candidate - baseline) / baseline * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_jsonl", type=Path)
    parser.add_argument("--baseline-m", type=int, default=1)
    args = parser.parse_args()

    rows = _load(args.summary_jsonl)
    by_prompt = defaultdict(dict)
    for row in rows:
        by_prompt[row["workload"]["prompt_length"]][row["row_tile_m"]] = row

    metrics = list(LOWER_IS_BETTER) + list(HIGHER_IS_BETTER)
    for prompt_length in sorted(by_prompt):
        group = by_prompt[prompt_length]
        baseline = group.get(args.baseline_m)
        if baseline is None:
            print(f"\n{prompt_length}: missing M={args.baseline_m} baseline")
            continue
        print(f"\nPrompt {prompt_length}")
        for row_tile_m in sorted(group):
            if row_tile_m == args.baseline_m:
                continue
            row = group[row_tile_m]
            parts = [f"M={row_tile_m}"]
            for metric in metrics:
                if metric not in row or metric not in baseline:
                    continue
                if row[metric] is None or baseline[metric] is None:
                    continue
                lower = metric in LOWER_IS_BETTER
                parts.append(
                    f"{metric}={row[metric]:.3f} ({_delta(row[metric], baseline[metric], lower):+.2f}%)"
                )
            print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
