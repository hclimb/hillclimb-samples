"""V1 verification (plan §Verification): pool-uniform IndexSampler kills the
step-28k data-heterogeneity signal.

The heterogeneity that `probe_h_b_data_heterogeneity.py` detected wasn't
between-source — HF `interleave_datasets(all_exhausted)` already balances
those. It was **within-source local clustering**: the streaming pipeline's
`.shuffle(buffer_size=100_000)` draws from a rolling 100k-slice of each
source's file order, so samples nearby in raw order (which may share
difficulty, doc-length, source-provider quirks) stay nearby in the training
stream. Over a 100k-step run at bs=16 the per-window mean raw_row_id drifts
monotonically 0 → 1 across the run.

Proxy: **normalized raw_row_id within each source** (raw_row_id / N_source).
Under a pool-uniform permutation the per-window mean should sit tight around
0.5 with spread ≈ 1/√(12 · n_source_per_window) (uniform distribution). Under
the old buffered-shuffle it drifts monotonically.

Method:
  1. Load manifest; extract raw_row_id per entry and per-source counts.
  2. Materialize the IndexSampler permutation and reorder raw_row_id_norm.
  3. Slide 2000-step (bs=16 → 32k samples) windows.
  4. Per source per window, compute mean(raw_row_id_norm) restricted to that
     source's samples in the window.
  5. Assert (a) mean stays inside 5σ of 0.5, and (b) the drift across windows
     (slope of mean vs window-index) is not systematically monotonic.

Analytic drift under the old pipeline (documented, not simulated): with
`shuffle_buffer=100k`, source `s`'s samples in window `w` come from
raw_row_id positions in approx [w · Δ, w · Δ + 100k), where Δ =
(N_source − 100k) / n_windows. Per-window mean drifts 0.05 → 0.95 linearly.
This monotonic drift is exactly what would show up as a step-28k regime
change if row order correlates with any latent difficulty proxy.

Run:
  cd $HOME/memory-layers && source .venv/bin/activate
  python scripts/debug/verify_v1_heterogeneity.py \\
    --manifest gs://memory-layers-training/indexed/<hash>/manifest.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_manifest(uri: str):
    import fsspec
    with fsspec.open(uri, "r") as f:
        return json.load(f)


def indexsampler_permutation(N: int, seed: int) -> np.ndarray:
    """Materialize the IndexSampler(shuffle=True, seed, num_epochs=1)
    permutation over [0, N)."""
    import grain.python as grain
    sampler = grain.IndexSampler(
        num_records=N, shuffle=True, seed=seed, num_epochs=1,
        shard_options=grain.NoSharding(),
    )
    perm = np.empty(N, dtype=np.int64)
    for i in range(N):
        perm[i] = sampler[i].record_key
    return perm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--window-steps", type=int, default=2000)
    ap.add_argument("--min-samples-per-window", type=int, default=200,
                    help="Skip windows with fewer than this many samples from a source.")
    args = ap.parse_args()

    print(f"[v1] loading manifest from {args.manifest}...", flush=True)
    manifest = load_manifest(args.manifest)
    N = len(manifest)
    print(f"[v1] N={N} entries", flush=True)

    source_names = sorted({e["source"] for e in manifest})
    name_to_k = {n: k for k, n in enumerate(source_names)}
    K = len(source_names)
    src_ids = np.array([name_to_k[e["source"]] for e in manifest], dtype=np.int32)
    raw_row_ids = np.array([int(e["raw_row_id"]) for e in manifest], dtype=np.int64)
    n_per_source = np.bincount(src_ids, minlength=K)

    # Per-entry proxy: RANK of raw_row_id within its source's surviving set,
    # normalized to [0, 1]. Rank (not raw_row_id/max) is the right proxy because
    # per-position survival rates can be non-uniform (some sources have late
    # rows surviving filter at a higher rate), which would skew a raw-value
    # proxy even under a perfect uniform shuffle. Rank is uniform in [0,1] by
    # construction over the surviving set.
    row_norm = np.zeros(N, dtype=np.float64)
    max_row_per_source = np.zeros(K, dtype=np.int64)
    for k in range(K):
        idx = np.where(src_ids == k)[0]
        rr = raw_row_ids[idx]
        max_row_per_source[k] = int(rr.max()) + 1
        # rank of each element within its source
        order = np.argsort(rr, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        row_norm[idx] = ranks.astype(np.float64) / max(len(idx) - 1, 1)

    print(f"[v1] K={K} sources:", flush=True)
    for k, name in enumerate(source_names):
        print(f"  {name}: N={n_per_source[k]} (max raw_row_id={max_row_per_source[k]-1})", flush=True)

    # Apply IndexSampler permutation
    print(f"[v1] materializing IndexSampler perm (seed={args.seed})...", flush=True)
    perm = indexsampler_permutation(N, args.seed)
    src_ids_p = src_ids[perm]
    row_norm_p = row_norm[perm]

    # Slide windows of `window_samples` samples
    window_samples = args.window_steps * args.batch_size
    n_w = N // window_samples
    print(f"[v1] window = {args.window_steps} steps × bs={args.batch_size} "
          f"= {window_samples} samples → {n_w} windows", flush=True)

    # Per-source-per-window mean(row_norm)
    means = np.full((n_w, K), np.nan, dtype=np.float64)
    counts = np.zeros((n_w, K), dtype=np.int64)
    for w in range(n_w):
        lo = w * window_samples
        hi = lo + window_samples
        chunk_src = src_ids_p[lo:hi]
        chunk_row = row_norm_p[lo:hi]
        for k in range(K):
            mask = chunk_src == k
            counts[w, k] = int(mask.sum())
            if counts[w, k] > 0:
                means[w, k] = float(chunk_row[mask].mean())

    print(f"\n[v1] per-source per-window mean(raw_row_id_normalized) — "
          f"expect ≈ 0.5 flat under uniform pool sampling", flush=True)
    verdict_pass = True
    for k, name in enumerate(source_names):
        col = means[:, k]
        cnt = counts[:, k]
        valid = ~np.isnan(col) & (cnt >= args.min_samples_per_window)
        col_v = col[valid]
        cnt_v = cnt[valid]
        if col_v.size < 5:
            print(f"  {name:60s} SKIP (too few windows with ≥{args.min_samples_per_window} samples)")
            continue
        # Under uniform, expected 1σ = std(Uniform[0,1]) / sqrt(n_per_window) = 1/sqrt(12 n)
        sigma = 1.0 / np.sqrt(12.0 * cnt_v.mean())
        overall = float(col_v.mean())
        # Deviation from 0.5, per-window
        max_dev = float(np.max(np.abs(col_v - 0.5)))
        max_dev_sigmas = max_dev / sigma if sigma > 0 else float("inf")
        # Trend test: slope of mean vs window-index, in units of "sigmas of 0.5"
        x = np.arange(col_v.size, dtype=np.float64)
        slope, intercept = np.polyfit(x, col_v, 1)
        # A monotonic drift 0→1 across the run has slope ≈ 1/n_w. Under the null
        # (independent per-window means), slope std is sigma*sqrt(12/(n_w^3 - n_w))
        # → σ_slope ≈ sigma * sqrt(12) / n_w^(3/2). Report drift/σ_slope.
        n = col_v.size
        slope_sigma = sigma * np.sqrt(12.0) / (n ** 1.5) if n > 3 else float("nan")
        slope_sigmas = slope / slope_sigma if slope_sigma > 0 else float("inf")
        # A monotonic old-pipeline drift over the whole run would look like
        # slope ≈ (0.95 - 0.05) / n_w ≈ 0.9/n_w — vastly larger than uniform σ.
        marker = ""
        if max_dev_sigmas > 5.0:
            marker += "  <-- MAX-DEV OUT OF BAND"
            verdict_pass = False
        if abs(slope_sigmas) > 5.0:
            marker += "  <-- MONOTONIC DRIFT"
            verdict_pass = False
        print(f"  {name:60s} mean={overall:.4f} "
              f"max_dev={max_dev:.4f} ({max_dev_sigmas:.1f}σ) "
              f"slope/step={slope:+.2e} ({slope_sigmas:+.1f}σ){marker}",
              flush=True)

    print("\n[v1] === VERDICT ===", flush=True)
    if verdict_pass:
        print("[v1] PASS — per-source raw_row_id distribution is flat across the "
              "training-length window sequence (no monotonic drift, no out-of-band "
              "windows). Pool-uniform ordering behaves as expected.", flush=True)
        print("[v1] (Old buffered-shuffle would show slope ≈ 0.9/n_w ~ "
              f"{0.9/max(n_w,1):.4f}/window here — many σ.)", flush=True)
        return 0
    else:
        print("[v1] FAIL — IndexSampler output is not uniform along raw_row_id. "
              "Either the sampler is broken or the manifest is not in raw insertion order.",
              flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
