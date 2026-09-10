# RAG@K accuracy/latency sweep: CE-only arm, hybrid hotpotqa

**Date:** 2026-08-17 · **Author:** claude (session with rohunagrawal) · **Status:** done.

## Conclusion

**There is no real accuracy/latency tradeoff in this pipeline as measured — latency is flat
across K, and accuracy is noisy and flat from K=10 onward.**

| K | latency | `llm_judge_accuracy` | `llm_judge_score` | `lexical_grounding` | `rag_all_golds@K` | `rag_any_gold@K` | bank slots |
|---|---|---|---|---|---|---|---|
| 5 | 17.8 min | 0.367 | 2.492 | 0.728 | 0.594 | 0.984 | 649 |
| 10 | 16.9 min | 0.422 | 2.578 | 0.693 | 0.742 | 0.984 | 1,365 |
| 25 | 17.2 min | 0.367 | 2.445 | 0.683 | 0.844 | 0.992 | 3,551 |
| 50 | 17.1 min | 0.414 | 2.539 | 0.724 | 0.891 | 0.992 | 7,141 |
| 100 | 17.5 min | **0.430** | 2.539 | 0.675 | 0.938 | 0.992 | 14,215 |
| 150 | 17.8 min | 0.406 | 2.555 | 0.638 | 0.945 | 0.992 | 21,249 |
| 200 | 18.2 min | 0.422 | 2.539 | 0.624 | 0.953 | 0.992 | 28,263 |

**Latency**: all 7 runs land within 16.9–18.2 minutes (≤8% spread) despite the bank growing from
649 to 28,263 slots (43×). The fixed per-run cost (encoding the full 9,811-doc corpus into the
memory bank, vLLM cold-start) dominates total wall-clock time; the K-dependent generation cost
(longer context at higher K) is a small fraction of it in this setup — see Interpretation.

**Accuracy**: bounces between 0.367 and 0.430 with no clean monotonic trend once K≥10 — best at
K=100 (0.430), but not distinguishable from K=10 (0.422) or K=200 (0.422) at n=128/point.
`rag_any_gold@K` (at least one correct doc present) is already 0.984+ by K=5; `rag_all_golds@K`
(every required doc present) keeps climbing all the way to K=200 without a correspondingly clear
gain in judged accuracy.

**Side note, not the main finding but worth flagging:** this sweep had to target step 61200, not
the step 42800 analyzed elsewhere this session (see below) — and 61200's accuracy across K
(0.37–0.43) looks meaningfully higher than 42800's corrected 0.3438, back in the range of the
source checkpoint's 0.4297. Tentative, single-sample-per-K evidence of recovery with continued
training — not confirmed.

## Hypothesis & motivation

rohunagrawal asked to sweep the CE-only arm's checkpoint
(`multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53`)
across several RAG `top_k` values on the hybrid hotpotqa eval, comparing judged accuracy and
end-to-end latency as a function of K — the RAG@K accuracy/latency tradeoff curve, as opposed to
every prior eval in this session's family, which used auto-K (the smallest K clearing a
`rag_all_golds` coverage threshold, capped at 200).

## Setup

- **Checkpoint:** same lineage/architecture as every other CE-only-arm eval this session
  (`qwen3_mem_embed`, single mem layer, `mem_top_k=64`, `mem_batched_isolation=true`). **Step
  61200**, not 42800 — see the ops note below for why.
- **Task/protocol:** `gen_large_mem_msa_hotpotqa_hybrid`, full 9,811-doc corpus, `gather_bank:
  true`, `B=8`, `max_new_tokens=512`, n=128 queries — identical to every other hybrid eval this
  session, **except** `rag.top_k` fixed per run and `rag.auto_k_threshold=null` (auto-K disabled),
  swept over `K ∈ {5, 10, 25, 50, 100, 150, 200}` — the same 7 candidates every auto-K eval in
  this codebase already picks from.
- **Judge:** `llm_judge_accuracy`/`llm_judge_score` (Qwen3-4B) + `lexical_grounding` — using the
  now-fixed `_parse_score` (see
  [2026-08-17-llm-judge-last-tag-parsing-fix.md](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md)),
  so these numbers don't need a separate correction pass.
- **New tooling:** [`scripts/embed/sweep_topk_hybrid.sh`](../../scripts/embed/sweep_topk_hybrid.sh)
  — see [implementation note](../implementations/2026-08-17-topk-sweep-hybrid-runner.md). Loops
  `eval_msa_hybrid.sh` unmodified per K, timing each run and reading back its metrics.
- **Box:** `rohun-v6e-8-0` (`us-east1-d`, `memorylayers`), kept separate from `rohun-v6e-8-1`
  (actively training this exact arm throughout).

**Ops note — the checkpoint moved under the sweep.** The CE-only arm trains continuously and
Orbax rotation keeps only the ~20 most recent checkpoints (200-step interval ≈ a 4,000-step
window). The sweep was first launched against step 42800 (the checkpoint analyzed everywhere else
this session) — by the time it ran, training had advanced to a 57400–61200 window and **all 7 K
values failed identically** with `FileNotFoundError: Checkpoint at .../42800 not found`. Relaunched
against the then-current latest step, 61200, chosen specifically to leave the most rotation
headroom for the sweep's ~2-hour total duration; all 7 completed cleanly.

## Results

Full auto-K-independent retrieval coverage curve (`rag_any_gold@K`/`rag_all_golds@K`), read off
each run's own metrics at its fixed K — consistent with every other eval's coverage curve on this
corpus (retrieval is a fixed external pre-pass, independent of the checkpoint or its generation
`top_k`):

| K | 5 | 10 | 25 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|
| `rag_any_gold@K` | 0.984 | 0.984 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 |
| `rag_all_golds@K` | 0.594 | 0.742 | 0.844 | 0.891 | 0.938 | 0.945 | 0.953 |

Bank size scales linearly with K as expected (`mean_active_bank_slots` ≈ K × ~141 per query,
consistent with the gather-bank per-query construction used throughout this eval family).

## Interpretation

**Why latency doesn't move with K.** Each run is a fresh `eval.py` process: encoding the full
9,811-doc corpus into embeddings and cold-starting the vLLM judge server both happen once per run,
independent of `top_k`. Only the generation step's prompt length depends on K (more retrieved
docs → longer context per query). At B=8, n=128, `max_new_tokens=512`, that K-dependent slice of
the total runtime is evidently small relative to the fixed corpus-encoding + cold-start cost —
consistent with all 7 runs landing within a ≤8% band regardless of a 43× range in bank size. This
means, for *this* pipeline shape, there's little latency cost to choosing a larger K — the
accuracy/latency tradeoff a RAG@K sweep is usually run to find essentially doesn't exist here,
because the bottleneck is elsewhere. (It would look different in a deployment that reuses one
encoded bank across many K choices or many queries — see the implementation note's follow-up on a
shared-encoding design that isn't built here.)

**Why accuracy doesn't clearly improve with K.** `rag_any_gold@K` — whether *a* correct document
is in the bank at all — is already 0.984 at K=5 and 0.992 by K=25; there's almost no more
single-doc recall to gain past that point. `rag_all_golds@K` (every required supporting document
present simultaneously) keeps climbing all the way to K=200, so *coverage* genuinely improves with
K — but that additional coverage doesn't translate into a clean, monotonic judged-accuracy gain in
this sample. Plausible explanations, not distinguished here: (a) real but small effect, swamped by
n=128 sampling noise (a 95%-CI half-width on a single proportion at n=128 is roughly ±0.09, larger
than most of the K-to-K deltas observed); (b) the model's answer-extraction gap (see
[2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md](2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md)'s
CoT-only follow-up) adds its own noise on top of whatever retrieval-quality signal exists,
diluting a real K-accuracy relationship if one exists underneath.

**The step-42800-vs-61200 side note is genuinely informative but not conclusive.** It's one data
point at a later step, not a replication — worth a dedicated matched-K comparison (evaluate both
steps at the same fixed K, e.g. 200) before treating "the regression recovered with more training"
as established.

## Reproducibility

```bash
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
RUN_SCRIPT_PATH=scripts/embed/sweep_topk_hybrid.sh \
RUN_ENV="DS=hotpotqa RUN_DIR=multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53 STEP=61200 KS=5,10,25,50,100,150,200" \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

- **Commit:** `b4d9541`-derived working tree on `multihop-finetuning`, plus this session's
  uncommitted `evals/metrics/llm_judge.py` fix and new `scripts/embed/sweep_topk_hybrid.sh`.
- **Checkpoint:** `gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53/qwen3_mem_embed/61200`
- **Per-K result JSONs:** `gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53/eval/step_61200/msa_hotpotqa_c10000_hybrid_autok_k{5,10,25,50,100,150,200}.json`
- **Summary:** `gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53/eval/step_61200/topk_sweep_summary.jsonl`
- **TPU:** `v6e-8` (`rohun-v6e-8-0`, `us-east1-d`, project `memorylayers`).
