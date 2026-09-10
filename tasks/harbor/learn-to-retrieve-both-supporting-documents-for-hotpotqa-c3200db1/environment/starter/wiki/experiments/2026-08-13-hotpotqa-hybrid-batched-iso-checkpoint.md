# Hybrid hotpotqa (full corpus) eval: multihop ground4layer batched-isolation warm-start, step 20200

**Date:** 2026-08-13 · **Author:** rohunagrawal (with Claude) · **Status:** done.

> **CORRECTION (2026-08-17):** `llm_judge_accuracy` below was inflated by a judge-output
> parsing bug (`_parse_score` took the first `<judgement>` tag instead of the model's actual
> final one) — see
> [2026-08-17-llm-judge-last-tag-parsing-fix.md](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md).
> Corrected (re-parsing the same stored judge output, no re-inference): **`llm_judge_accuracy =
> 0.4297` (55/128)**, down from the 0.4609 (59/128) reported below; 4 samples flipped. Kept the
> original numbers below unedited per this repo's append-only-log convention — this box is the
> current, authoritative value.

## Conclusion

**`llm_judge_accuracy = 0.4609`, `llm_judge_score = 2.58` (1–5), `lexical_grounding = 0.591`** on
the RAG→memory hybrid hotpotqa task at the full 9,811-doc corpus (auto-K capped at 200 — no
candidate cleared the 0.96 all-golds threshold). Getting this number required fixing a real
retrieval-dispatch bug first — see the implementation note linked below — so the main output of
this session is as much "the hybrid gather-bank eval now works against `mem_batched_isolation`
checkpoints" as it is the number itself.

## Hypothesis & motivation

Asked to eval
`multihop_ground4layer_s1warmstart_no_multihop_stage2only_batched_iso_topk128_bs8-2026-08-12-05-42-27`
(a 4-memory-layer, `mem_batched_isolation: true`, warm-started-from-`ground_s1_zeroinit_4layer_no_multihop`
checkpoint — see [2026-08-13-doc-access-per-query-loss-investigation.md](2026-08-13-doc-access-per-query-loss-investigation.md)
for its training lineage) on hybrid hotpotqa, full corpus, no specific hypothesis beyond getting a
judged read on where this checkpoint lands.

## Setup

- **Checkpoint:** `qwen3_mem_embed`, `mem_layers=[9,14,20,27]`, `mem_top_k=128`,
  `per_query_isolation: true`, `isolation_group_size: 1`, `mem_batched_isolation: true`,
  `mem_o_proj_zero_init: true` — step 20200 (latest available; checkpoints run 16400→20200 in
  GCS at the time of this eval).
- **Task:** `gen_large_mem_msa_hotpotqa_hybrid` (`configs/eval/tasks/gen_large_mem_msa_hotpotqa_hybrid.yaml`)
  — RAG→memory hybrid, `inject_query_gold` corpus at `target_docs=10000` (hotpotqa's actual corpus
  is 9,811 docs, so this is the **full corpus**, not a trimmed subset), per-query **gathered
  banks** (`gather_bank: true`), `B=8` data-parallel, `max_new_tokens=512`, `MEM_APPROX_TOPK=1`,
  n=128 queries. Same protocol as the [2026-07-22 MSA sweep](2026-07-22-msa-sweep-hybrid-vs-rag.md).
  **auto-K:** smallest of `{5,10,25,50,100,150,200}` clearing mean `rag_all_golds ≥ 0.96`; 200 is
  the hard cap when none clears (recorded in `metrics.rag_top_k`).
- **Judge:** `llm_judge_accuracy` + `llm_judge_score` (1–5), Qwen3-4B, `tensor_parallel_size=4`,
  plus `lexical_grounding`.
- Runner: `scripts/embed/eval_msa_hybrid.sh` (`DS=hotpotqa`).

## Results

| metric | value |
|---|---|
| `llm_judge_accuracy` | **0.4609** (59/128) |
| `llm_judge_score` (1–5) | **2.578** |
| `lexical_grounding` | 0.5906 |
| `rag_top_k` (auto-K) | 200 (**CAP** — no candidate ≥ 0.96) |
| `rag_any_gold@200` | 0.9922 |
| `rag_all_golds@200` (bank coverage) | 0.9531 |
| `rag_all_golds@10` / `@100` | 0.7422 / 0.9375 |
| `corpus_docs` | 9,811 (full corpus) |
| `bank_slots` | 2,511,616 |
| `mean_active_bank_slots` (gather-bank, per query) | 28,263 |
| `generated_count` | 128 / 128 |
| `doc_hit_rate` / `mem_pos_weight_mass` | **not available** — `aux_telemetry_oom: True`, see below |

Full auto-K coverage curve (retrieval pre-pass, corpus-wide, temp/K-independent):

| k | 5 | 10 | 25 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|
| `rag_any_gold@k` | 0.984 | 0.984 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 |
| `rag_all_golds@k` | 0.594 | 0.742 | 0.844 | 0.891 | 0.938 | 0.945 | 0.953 |

## Interpretation

- **Context, not a controlled comparison:** the [2026-07-22 sweep](2026-07-22-msa-sweep-hybrid-vs-rag.md)
  measured the *hard-neg base* checkpoint on the same task/protocol at k=200 CAP:
  `acc=0.320, score=2.36, grounding=0.481`. This checkpoint scores higher on all three
  (**+0.14 acc, +0.22 score, +0.11 grounding**), but it's a **different architecture and training
  lineage** (4 memory layers + batched isolation + a ground-then-multihop warm-start recipe vs.
  the single-layer hard-neg base) — informative as a data point, not a controlled ablation of any
  one variable.
- **`aux_telemetry_oom: True`** — the per-row aux-telemetry forward (which would have produced
  `doc_hit_rate`/`mem_pos_weight_mass` for this run) OOM'd and was disabled; judge accuracy, score,
  grounding, and the auto-K coverage curve are all unaffected (they come from the retrieval
  pre-pass and the judge, not that forward pass). Worth revisiting if per-query retrieval-mass
  diagnostics are wanted for this checkpoint specifically — the 07-22 sweep's mechanism (grounding
  collapsing on k=200-CAP arms because reading 200 semantically-close docs through memory slots
  detaches generation from evidence) is the natural hypothesis to check `mem_pos_weight_mass`
  against here too.
- **Getting a number at all required a code fix first.** The very first attempt crashed — this is
  the first `mem_batched_isolation: true` checkpoint ever run through the hybrid gather-bank eval,
  and that combination had a real, previously-unexercised dispatch gap. See
  [implementation note](../implementations/2026-08-13-batched-isolation-hybrid-eval-mask-fallback.md).
- **Infra was unusually rocky:** three straight launch attempts (`rohun-v6e-8-0`, `rohun-v6e-8-1`,
  `rohun-v6e-8-0` again) were killed by a sustained `us-east1-d` zone-wide spot-preemption wave
  (other people's `john-v6e-8-*` boxes cycling in the same window — not specific to this job); the
  successful run landed on `tn-v6e-8-0` (`europe-west4-a`), which was free at the time despite
  having hosted training earlier the same day (see
  [2026-08-13-doc-access-per-query-loss-investigation.md](2026-08-13-doc-access-per-query-loss-investigation.md)).
  First two killed attempts got as far as a clean 128/128 generation before dying in the
  judge-startup phase — good independent confirmation the retrieval fix itself works, even before
  the run that finally completed end-to-end.

## Follow-up (same day, ~21:10): `MEM_SOFTMAX_TEMP=0.5` — does the 2026-07-22 sharpening result replicate here?

The [2026-07-22 sweep](2026-07-22-msa-sweep-hybrid-vs-rag.md) found that sharpening the memory
read (`MEM_SOFTMAX_TEMP=0.5`, same corpus/queries/k=200) lifted the *hard-neg base* checkpoint's
hotpotqa hybrid accuracy 0.320→0.391 (+0.070), score 2.36→2.59, grounding 0.481→0.587. Reran the
identical env-var override against **this** checkpoint (same corpus, queries, k=200 CAP —
retrieval pre-pass is temp-independent, confirmed identical to the baseline run above) to check
whether that finding generalizes.

| | temp 1.0 (baseline, above) | temp 0.5 | Δ |
|---|---|---|---|
| `llm_judge_accuracy` | 0.4609 | 0.4531 | **−0.008** |
| `llm_judge_score` (1–5) | 2.578 | 2.375 | **−0.203** |
| `lexical_grounding` | 0.5906 | 0.6125 | +0.022 |
| `rag_top_k` (auto-K) | 200 (CAP) | 200 (CAP) | unchanged (temp-independent) |

**It does not replicate on this checkpoint.** Grounding moves the same direction as 2026-07-22
(modest improvement — the read is citing its sources slightly more), but accuracy is flat within
judge noise (±0.04 per that sweep's own estimate) rather than jumping +0.07, and the 1–5 score
actually drops by a clinically-sized 0.2. The 2026-07-22 result was measured on the single-layer
*hard-neg base* checkpoint; this one is a 4-memory-layer `mem_batched_isolation` checkpoint from a
completely different training recipe (ground-then-multihop warm-start). Sharpening's effect on
`mem_pos_weight_mass`/hop behavior plausibly depends on how peaked the *unsharpened* distribution
already is for a given architecture — not available here to check directly (`aux_telemetry_oom`
disabled it on both arms) — so this isn't evidence the original mechanism was wrong, only that
temp=0.5 isn't a checkpoint-agnostic free win and needs re-validating per architecture rather than
assumed to carry over.

Result JSON: `.../eval/step_20200/msa_hotpotqa_c10000_hybrid_autok_memtemp05.json` (same GCS
run-dir as the baseline, `_memtemp05` suffix). Same wandb run (`train_step=20200`, distinct metric
namespace `eval/msa_hotpotqa_c10000_hybrid_autok_memtemp05/*`).

```bash
MEM_SOFTMAX_TEMP=0.5 DS=hotpotqa \
RUN_DIR=multihop_ground4layer_s1warmstart_no_multihop_stage2only_batched_iso_topk128_bs8-2026-08-12-05-42-27 \
STEP=20200 NAME_SUFFIX=_memtemp05 \
TPU_NAME=tn-v6e-8-0 ZONE=europe-west4-a PROJECT_ID=memorylayers \
RUN_ENV="DS=$DS RUN_DIR=$RUN_DIR STEP=$STEP NAME_SUFFIX=$NAME_SUFFIX MEM_SOFTMAX_TEMP=$MEM_SOFTMAX_TEMP" \
RUN_SCRIPT_PATH=scripts/embed/eval_msa_hybrid.sh \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

## Reproducibility

```bash
DS=hotpotqa \
RUN_DIR=multihop_ground4layer_s1warmstart_no_multihop_stage2only_batched_iso_topk128_bs8-2026-08-12-05-42-27 \
STEP=20200 \
TPU_NAME=tn-v6e-8-0 ZONE=europe-west4-a PROJECT_ID=memorylayers \
RUN_ENV="DS=$DS RUN_DIR=$RUN_DIR STEP=$STEP" \
RUN_SCRIPT_PATH=scripts/embed/eval_msa_hybrid.sh \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

- **Commit:** `c6c21cdf5e5e66eb6dbaf71d3786bc49eab09b78` on `multihop-finetuning`, **plus the
  uncommitted `mem_lookup_batched` mask-fallback fix** (`models/memory.py`, see the implementation
  note) — required for this run to complete at all.
- **Checkpoint:** `gs://memory-layers-training/multihop_ground4layer_s1warmstart_no_multihop_stage2only_batched_iso_topk128_bs8-2026-08-12-05-42-27/qwen3_mem_embed/20200`.
- **Result JSON:** `gs://memory-layers-training/multihop_ground4layer_s1warmstart_no_multihop_stage2only_batched_iso_topk128_bs8-2026-08-12-05-42-27/eval/step_20200/msa_hotpotqa_c10000_hybrid_autok.json`.
- **wandb (eval-worker generation run):** https://wandb.ai/johnzhang2366-columbia-university/memory-layers-eval/runs/f0hbk31y
- **wandb (metrics, logged into the training run at `train_step=20200`):** https://wandb.ai/johnzhang2366-columbia-university/memory-layers/runs/multihop_ground4layer_s1warmstart_no_mul-2026-08-12-05-42-27
- **TPU:** v6e-8, `tn-v6e-8-0`, `memorylayers` project, `europe-west4-a`.
