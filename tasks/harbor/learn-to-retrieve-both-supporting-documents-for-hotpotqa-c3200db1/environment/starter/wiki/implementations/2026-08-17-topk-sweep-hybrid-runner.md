# `sweep_topk_hybrid.sh` — RAG@K accuracy/latency sweep runner

**Date:** 2026-08-17 · **Author:** claude (session with rohunagrawal) · **Status:** done ·
**Branch:** `multihop-finetuning`.

## What changed

New `scripts/embed/sweep_topk_hybrid.sh`: loops `eval_msa_hybrid.sh` (unmodified) over a fixed
list of RAG `top_k` values with auto-K disabled (`rag.auto_k_threshold=null`), timing each
invocation end-to-end and reading back its `metrics` block, to produce an accuracy-vs-K and
latency-vs-K table for one checkpoint in one command.

## Motivation & context

rohunagrawal asked to sweep the CE-only arm's hybrid hotpotqa eval across several RAG `top_k`
values and compare both judged accuracy and end-to-end latency as a function of K. No existing
tooling parameterized `top_k` across multiple runs or measured wall-clock time — `eval_msa_hybrid.sh`
already supports overriding `rag.top_k`/`rag.auto_k_threshold` via `EXTRA_OVERRIDES` (documented
in its own header comment) for a *single* run, but nothing looped it or timed it.

## Options weighed

1. **A new eval task/evaluator that sweeps K internally in one `eval.py` process** (would let the
   corpus-encoding phase be shared once across all K, amortizing that fixed cost). Rejected: bigger
   change to `evals/gen_large_mem_rag_hybrid.py`, and CLAUDE.md's convention is standalone scripts
   over instrumenting eval/trainer internals for a one-off measurement.
2. **A thin bash loop over the existing, unmodified `eval_msa_hybrid.sh`** (chosen). Each K pays
   the full corpus-encoding + vLLM cold-start cost again — a real cost, documented as a caveat on
   every latency number this script produces — but it needed zero changes to tested eval code and
   was fast to build correctly.

## How it was built & integrated

- `scripts/embed/sweep_topk_hybrid.sh`: for each `K` in `KS` (default `5,10,25,50,100,150,200`,
  the same auto-K candidate list used elsewhere in this codebase, so the new accuracy/latency data
  lines up with each eval's own already-computed `rag_all_golds@K`/`rag_any_gold@K` coverage
  curve), calls `NAME_SUFFIX="_k${K}" EXTRA_OVERRIDES="evals.msa_hybrid.eval.rag.top_k=$K,evals.msa_hybrid.eval.rag.auto_k_threshold=null"
  bash scripts/embed/eval_msa_hybrid.sh`, wall-clock-timed with `date +%s` around the call. Reads
  the resulting GCS result JSON's `metrics` block back into one summary line per K
  (`llm_judge_accuracy`, `llm_judge_score`, `lexical_grounding`, `rag_all_golds@K`/`rag_any_gold@K`
  at that exact K, `mean_active_bank_slots`, `latency_seconds`), appended to a `.jsonl` uploaded
  beside the per-K result files.
- Idempotent per K for free: `eval_msa_hybrid.sh`'s own GCS-existence skip-check applies per
  `NAME_SUFFIX`, so a killed/resumed sweep doesn't redo completed K values.
- **No changes to `eval.py`, `evals/gen_large_mem_rag_hybrid.py`, or `eval_msa_hybrid.sh`.**

## Reference pages updated

None needed — reuses `eval_msa_hybrid.sh`'s existing, already-documented `EXTRA_OVERRIDES`/
`NAME_SUFFIX` knobs (see [evaluation/rag-eval.md](../evaluation/rag-eval.md) /
[eval-configs.md](../evaluation/eval-configs.md)) exactly as designed; nothing about how the
underlying eval works changed.

## Tests

No unit test — thin orchestration over already-tested `eval_msa_hybrid.sh`. Validated end-to-end:
first launch failed cleanly and informatively (checkpoint step 42800 had rotated off GCS between
being chosen and the sweep running, since the arm trains continuously — `FileNotFoundError`, not a
silent wrong result) on all 7 K values; relaunched against the then-current step 61200 and all 7
completed. See
[2026-08-17-hotpotqa-hybrid-topk-sweep-ce-only.md](../experiments/2026-08-17-hotpotqa-hybrid-topk-sweep-ce-only.md)
for the full result and interpretation.

## Follow-ups & risks

- **Every latency number includes the full K-independent corpus-encoding + vLLM cold-start
  cost**, repeated per K (~17-18 minutes total, all 7 K values landed within ~8% of each other) —
  this sweep cannot isolate the true marginal, K-dependent generation cost from that fixed
  overhead. A shared-encoding design (see Option 1 above) would be needed to measure that
  cleanly; not built here.
- **A checkpoint under active training is a moving target for any multi-run sweep.** This bit the
  first launch attempt. Pick a step with enough rotation headroom for the sweep's expected total
  duration (checkpoint rotation keeps a fixed *count*, not a time window, so a fresh step buys the
  most headroom), or point at a stopped/stable checkpoint if reproducibility across reruns matters
  more than freshness.
