# Approx memory top-k as a model-config default

**Date:** 2026-07-15 · **Author:** rohunagrawal · **Status:** done · **Branch:** `train_approx_top_k`

## What changed

The memory-bank top-k op (exact `jax.lax.top_k` vs TPU-fast `jax.lax.approx_max_k`) is now
selected by **model-config keys** `mem_approx_topk` / `mem_approx_recall`, not only the
`MEM_APPROX_TOPK` env var. `qwen3_mem_embed` defaults to **`mem_approx_topk: true`,
`mem_approx_recall: 0.99`**, so training/eval on that model use approx by default — no env var
needed.

## Motivation & context

The [microbenchmark](../experiments/2026-07-15-approx-topk-training.md) showed exact top-k over
M≈65,536 is ~40% of the whole 4B train step, and `approx_max_k` cuts the step **1.63× @recall 0.99**
(1.68× @0.95) at **98.7%** recall@64. Previously approx was reachable only via the `MEM_APPROX_TOPK`
env var (default off) — fine for a one-off experiment, wrong as the standing default for a training
run. Prior reference: [retrieval-modes.md](../architecture/retrieval-modes.md) `## bank_top_k`.

## Options weighed & tradeoffs

- **Config key + code fallback = exact** (chosen) vs. flipping the *code* default to approx: a
  global code-default would silently change every existing mem model + eval config that never
  opted in (repro hazard). Keeping the built-in fallback exact and setting the flag only in
  `qwen3_mem_embed.yaml` scopes the change to the model we benchmarked.
- **Precedence env > cfg > default** (chosen) vs. cfg-only: keeping the env override lets a run
  A/B exact vs approx without editing configs (`MEM_APPROX_TOPK=0`), matching the codebase's other
  eval-time env knobs (`MEM_SCORE_ACTIVATION`, `MEM_TOP_K`, …).
- **Recall 0.99** vs 0.95 default: 0.99 costs only ~15 ms/step more than 0.95 (515 vs 500) but
  lifts recall 98.2%→98.7% — cheap safety, so it's the default.

## How it was built & integrated

- `models/memory_utils.py`: new `resolve_approx_topk(cfg)` (env > cfg > exact/0.95); `bank_top_k(x,
  k, cfg=None)` calls it. Replaces the old import-time `_MEM_APPROX_TOPK` module constants.
- `cfg` threaded to every `bank_top_k` call site: `memory.py::mem_lookup`, `::mem_lookup_gqa`, and
  the `retrieval_ops.py` sharded chain (`sharded_top_k_ip` → `_sharded_top_k` → `_matmul_top_k`,
  each gains a `cfg=None` param). The cross-shard merge stays exact `top_k` (only n_shards·K
  candidates — cheap). The chunked-scan running buffer (`_scan_chunks`) is left exact (separate
  algorithm; not on the default path).
- `configs/model/qwen3_mem_embed.yaml`: added `mem_approx_topk: true`, `mem_approx_recall: 0.99`
  in the `memory:` block. Propagates into the merged main-model cfg via the existing
  `add_memory_layer` → `model_cfg.update(...)` path (same mechanism as `mem_top_k`).

New config keys: `mem_approx_topk` (bool, default via fallback = false; `qwen3_mem_embed` = true),
`mem_approx_recall` (float, default 0.95; `qwen3_mem_embed` = 0.99).

## Reference pages updated

[retrieval-modes.md](../architecture/retrieval-modes.md) `## bank_top_k` — rewritten for the
cfg-driven selection + precedence.

## Tests

- **Resolver precedence** (on `rohun-v6e-8-0`):
  `resolve_approx_topk(None)==(False,0.95)`, `({mem_approx_topk:true,recall:0.99})==(True,0.99)`,
  `({mem_approx_topk:false})==(False,0.95)`, env `MEM_APPROX_TOPK=0` overrides cfg true → exact,
  env `1`+`RECALL=0.9` overrides cfg false → `(True,0.9)`. → `RESOLVE_OK: env>cfg>default all pass`.
- **End-to-end config default drives approx** (`--mode time`, **no env var** set, so only the
  config can enable approx): median step **515.65 ms** (n=10, p10/p90 515.20/515.83) ≈ the
  approx@0.99 arm (515.1 ms), *not* the exact arm (838.7 ms) — confirms the cfg reaches
  `bank_top_k` through the real train step. (The `[BENCH]` line labels `MEM_APPROX_TOPK=0` because
  it echoes the unset env var's display default; the 515 ms proves approx ran, driven by the cfg.)

## Follow-ups & risks

- Applies to **eval** on `qwen3_mem_embed` too (approx retrieval) — intended; set
  `mem_approx_topk: false` for an exact-retrieval eval.
- Tier-2 A/B (loss parity on real data) was **not** run; if a quality regression is suspected, run
  it (needs the HF datasets pre-cached — see the experiment note).
