# Hybrid hotpotqa (full corpus) eval: qa_hard_neg_think_sft4b pf32/indexed/lr_masked, step 100000

**Date:** 2026-08-13 · **Author:** rohunagrawal (with Claude) · **Status:** done.

> **CORRECTION (2026-08-17):** `llm_judge_accuracy` below was inflated by a judge-output
> parsing bug (`_parse_score` took the first `<judgement>` tag instead of the model's actual
> final one) — see
> [2026-08-17-llm-judge-last-tag-parsing-fix.md](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md).
> Corrected (re-parsing the same stored judge output, no re-inference): **`llm_judge_accuracy =
> 0.4297` (55/128)**, down from the 0.4609 (59/128) reported below; 4 samples flipped. Kept the
> original numbers below unedited per this repo's append-only-log convention — this box is the
> current, authoritative value. This is also the "source checkpoint" baseline the CE-only arm's
> regression (see
> [2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md](2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md))
> is measured against — the regression direction and rough magnitude are unchanged by this fix.

## Conclusion

**`llm_judge_accuracy = 0.4609`, `llm_judge_score = 2.81` (1–5), `lexical_grounding = 0.621`,
`doc_hit_rate = 0.992`, `mem_pos_weight_mass = 0.445`** on the RAG→memory hybrid hotpotqa task at
the full 9,811-doc corpus (auto-K capped at 200). As with the ground4layer checkpoint evaluated
earlier the same day, getting a number required a code fix first — this time a dtype crash in the
embed-conv path, unrelated to the earlier mask-dispatch bug. See the implementation note linked
below.

## Hypothesis & motivation

Asked to run the same hybrid hotpotqa protocol against
`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45`
(the checkpoint used as Arm 2's warm-start source in
[2026-08-13-doc-access-per-query-loss-investigation.md](2026-08-13-doc-access-per-query-loss-investigation.md)),
step 100000 (the latest available). No specific hypothesis beyond getting a judged read on this
checkpoint under the same protocol as the ground4layer run, for context/comparison.

## Setup

- **Checkpoint:** `qwen3_mem_embed`, **single** memory layer (`mem_layers=[14]`), `mem_size=16384`,
  `mem_top_k=64`, `mem_placement=after_attention`, `mem_o_proj_zero_init=false`, no product keys,
  no batched isolation / per-query isolation (`mem_lookup` default path). `embed_model`:
  Qwen3-Embedding-0.6B with `embed_conv: true` (kernel=1, stride=1). Trainable:
  `.*mem_.*`, `.*embed_model.*` (main 4B frozen) — this run's trainable regex includes
  `.*embed_proj_conv.*` via the `.*embed_model.*` match, so its conv weights are fp32-promoted
  (see implementation note).
- **Task / protocol:** identical to
  [the ground4layer eval](2026-08-13-hotpotqa-hybrid-batched-iso-checkpoint.md) — same
  `gen_large_mem_msa_hotpotqa_hybrid` task, full 9,811-doc corpus, `gather_bank: true`, `B=8`,
  `max_new_tokens=512`, n=128, auto-K over `{5,10,25,50,100,150,200}` at threshold 0.96.
- Runner: `scripts/embed/eval_msa_hybrid.sh` (`DS=hotpotqa`).

## Results

| metric | value |
|---|---|
| `llm_judge_accuracy` | **0.4609** (59/128) |
| `llm_judge_score` (1–5) | **2.8125** |
| `lexical_grounding` | 0.6211 |
| `doc_hit_rate` | 0.9922 |
| `mem_pos_weight_mass` | 0.4454 |
| `rag_top_k` (auto-K) | 200 (**CAP** — no candidate ≥ 0.96) |
| `rag_all_golds@200` (bank coverage) | 0.9531 |
| `corpus_docs` | 9,811 (full corpus) |
| `generated_count` | 128 / 128 |

Auto-K coverage curve is identical to the ground4layer run's (the retrieval pre-pass that builds
`rag_any_gold@k`/`rag_all_golds@k` uses a fixed external retriever over the same fixed
corpus/queries, independent of which checkpoint is being evaluated — not a coincidence).

## Interpretation

- **vs. the ground4layer batched-isolation checkpoint (same day, same protocol):** accuracy is
  identical (0.4609 both), but this checkpoint scores meaningfully higher on `llm_judge_score`
  (2.81 vs 2.58) and `lexical_grounding` (0.621 vs 0.591). **Not a controlled comparison** — different
  architecture (1 memory layer vs 4, `mem_top_k=64` vs 128, no batched/per-query isolation),
  different training recipe and lineage entirely. Useful as a second data point, not an ablation.
- **This is the first checkpoint of the day where `aux_telemetry_oom` was `False`**, so
  `doc_hit_rate`/`mem_pos_weight_mass` are actually available: `doc_hit_rate=0.992` (near-ceiling —
  retrieval finds a correct doc almost every time) against `mem_pos_weight_mass=0.445` (under half
  the softmax mass lands on a correct-doc slot) is the same "retrieval isn't the bottleneck, mass
  gets diluted across the k=200 CAP bank" pattern the 2026-07-22 sweep diagnosed on the hard-neg
  base checkpoint (there: `doc_hit_rate≈0.99` region, `mem_pos_weight_mass=0.205` at temp=1.0,
  same k=200 CAP) — this checkpoint's mass (0.445) is over 2× higher despite the same k=200 CAP,
  worth a closer look if the dilution mechanism is revisited.
- **Second, independent code fix needed the same day:** unlike the ground4layer run's mask-dispatch
  bug, this one crashed on a dtype mismatch in `apply_conv1d` (`embed_proj_conv` weights
  fp32-promoted by `promote_trainable_to_fp32`, activations bf16) — see
  [implementation note](../implementations/2026-08-13-conv1d-mixed-dtype-cast.md). Two genuinely
  different, previously-latent gaps in the hybrid gather-bank eval path surfaced by two different
  checkpoints on the same day; the eval path itself had evidently only ever been run against a
  narrow slice of the checkpoint config space before today.

## Reproducibility

```bash
DS=hotpotqa \
RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45 \
STEP=100000 \
TPU_NAME=tn-v6e-8-0 ZONE=europe-west4-a PROJECT_ID=memorylayers \
RUN_ENV="DS=$DS RUN_DIR=$RUN_DIR STEP=$STEP" \
RUN_SCRIPT_PATH=scripts/embed/eval_msa_hybrid.sh \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

- **Commit:** `c6c21cdf5e5e66eb6dbaf71d3786bc49eab09b78` on `multihop-finetuning`, **plus the
  uncommitted `mem_lookup_batched` mask-fallback fix and the `apply_conv1d` dtype-cast fix**
  (`models/memory.py`, `models/conv_utils.py`) — the latter required for this run to complete at
  all.
- **Checkpoint:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/qwen3_mem_embed/100000`.
- **Result JSON:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/eval/step_100000/msa_hotpotqa_c10000_hybrid_autok.json`.
- **wandb (eval-worker generation run):** https://wandb.ai/johnzhang2366-columbia-university/memory-layers-eval/runs/qk285ba1
- **wandb (metrics, logged into the training run at `train_step=100000`):** run
  `qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-08-09-05-49-45` in
  `johnzhang2366-columbia-university/memory-layers`.
- **TPU:** v6e-8, `tn-v6e-8-0`, `memorylayers` project, `europe-west4-a`.
