# Hybrid hotpotqa (full corpus) eval: CE-only arm (lr_masked warm-start lineage), step 42800

**Date:** 2026-08-16 · **Author:** claude (session with rohunagrawal) · **Status:** done.

> **CORRECTION (2026-08-17):** `llm_judge_accuracy` below was inflated by a judge-output
> parsing bug (`_parse_score` took the first `<judgement>` tag instead of the model's actual
> final one) — see
> [2026-08-17-llm-judge-last-tag-parsing-fix.md](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md).
> Corrected (re-parsing the same stored judge output, no re-inference): **`llm_judge_accuracy =
> 0.3438` (44/128)**, down from the 0.3828 (49/128) reported below; 5 samples flipped. The source
> checkpoint's own corrected number is 0.4297 (see
> [2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md](2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md)) —
> so the corrected regression is **0.4297→0.3438** (a slightly *larger* gap than the originally
> reported 0.4609→0.3828), same direction and conclusion. Separately, re-judging just this
> checkpoint's chain-of-thought (ignoring the extracted final answer entirely) with a larger
> judge (Qwen3-8B) gives **0.5078** — see the CoT-accuracy follow-up below; that's a distinct
> finding from this parsing bug, not a further correction to the number above.

## Conclusion

**`llm_judge_accuracy = 0.3828` (49/128), `llm_judge_score = 2.359` (1–5), `lexical_grounding =
0.619`, `doc_hit_rate = 0.992`, `mem_pos_weight_mass = 0.575`** on the RAG→memory hybrid hotpotqa
task, full 9,811-doc corpus, auto-K capped at 200 — same protocol as the two 2026-08-13 checkpoint
evals.

**This is a real regression relative to this arm's own warm-start source**, not an improvement.
The source checkpoint (`qa_hard_neg_think_sft4b_..._pf32_indexed_lr_masked` step 100000, evaluated
in
[2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md](2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md))
scored `llm_judge_accuracy=0.4609`, `score=2.81` — both higher than this checkpoint, despite 42,800
further training steps and in-training metrics (`ce_loss`, `doc_access_per_query_loss`,
`mem_pos_weight_mass`) all improving throughout, per the ongoing health-check log in
[2026-08-13-doc-access-per-query-loss-investigation.md](2026-08-13-doc-access-per-query-loss-investigation.md).
Notably, `doc_hit_rate` is identical (0.992) and `mem_pos_weight_mass` is *higher* (0.575 vs 0.445)
than the source checkpoint — so the drop in judged QA accuracy is not explained by retrieval
getting worse by these proxies. See Interpretation below.

## Hypothesis & motivation

rohunagrawal asked to run the same hybrid hotpotqa eval used for the two 2026-08-13 checkpoints
against "the 36k checkpoint" of the CE-only arm — the arm currently running and being
health-checked in the doc-access investigation doc. Step 36000 had already rotated off GCS by the
time of the request (Orbax keeps a rolling window of ~20 checkpoints per run-dir, and training had
moved past it); asked rohunagrawal whether to use the earliest still-available checkpoint (38800)
or whatever was latest at launch time — chose **latest at launch time** (42800). No specific
hypothesis beyond getting a judged read on where continued CE-only training has moved this
checkpoint, for comparison against its own source.

## Setup

- **Checkpoint:** `qwen3_mem_embed`, single memory layer (`mem_layers=[14]`), `mem_top_k=64`,
  `mem_batched_isolation=true`, `per_query_isolation=true`, `main_model` unfrozen (same lineage as
  Arm 2 in the doc-access investigation — warm-started from
  `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked` step 100000, then
  trained through the CE-only arm: CoT-derived `<think>`-prefixed CE target only, memory bank
  untouched). Run-dir
  `multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53`,
  step **42800** — the latest checkpoint on GCS at eval-launch time; training continues
  concurrently on `rohun-v6e-8-1`, so this is a moving-target snapshot, not a fixed milestone.
- **Task / protocol:** identical to the two 2026-08-13 evals —
  `gen_large_mem_msa_hotpotqa_hybrid`, full 9,811-doc corpus, `gather_bank: true`, `B=8`
  data-parallel, `max_new_tokens=512`, `MEM_APPROX_TOPK=1`, n=128 queries, auto-K over
  `{5,10,25,50,100,150,200}` at threshold 0.96 (never cleared — capped at 200, same as both prior
  evals: the auto-K coverage curve is corpus/retriever-fixed, independent of checkpoint).
- **Judge:** `llm_judge_accuracy` + `llm_judge_score` (1–5), Qwen3-4B, `tensor_parallel_size=4`,
  plus `lexical_grounding`.
- **Box:** `rohun-v6e-8-0` (`us-east1-d`, project `memorylayers`) — a separate box from the
  `rohun-v6e-8-1` box actively training this arm, per
  [wiki/evaluation/eval-boxes.md](../evaluation/eval-boxes.md) (the judge's vLLM needs the TPU
  training is holding). Verified idle/clean (`fuser -v /dev/vfio/*` empty, no tmux session, no
  orphaned `venv/bin/python`) before launching.
- **Code fixes required:** this checkpoint's architecture combines `mem_batched_isolation: true`
  with `gather_bank`'s per-row mask (needs
  [`mem_lookup_batched`'s 2D-mask fallback](../implementations/2026-08-13-batched-isolation-hybrid-eval-mask-fallback.md))
  and a trainable, fp32-promoted `embed_proj_conv` (needs
  [`apply_conv1d`'s dtype cast](../implementations/2026-08-13-conv1d-mixed-dtype-cast.md)) — the
  same two fixes the prior two evals needed. Both were already applied in the working tree
  (uncommitted at the time of this eval) and picked up automatically by the launcher's tree sync;
  no new code changes were needed for this eval.
- Runner: `scripts/embed/eval_msa_hybrid.sh` (`DS=hotpotqa`).

## Results

| metric | value |
|---|---|
| `llm_judge_accuracy` | **0.3828** (49/128) |
| `llm_judge_score` (1–5) | **2.359** |
| `lexical_grounding` | 0.619 |
| `doc_hit_rate` | 0.992 |
| `mem_pos_weight_mass` | 0.575 |
| `rag_top_k` (auto-K) | 200 (**CAP** — no candidate ≥ 0.96) |
| `rag_any_gold@200` | 0.992 |
| `rag_all_golds@200` (bank coverage) | 0.953 |
| `rag_all_golds@10` / `@100` | 0.742 / 0.938 |
| `corpus_docs` | 9,811 (full corpus) |
| `bank_slots` | 2,511,616 |
| `mean_active_bank_slots` (gather-bank, per query) | 28,263 |
| `generated_count` | 128 / 128 |

Full auto-K coverage curve (retrieval pre-pass, corpus-wide, checkpoint-independent — identical to
both 2026-08-13 evals, confirming this is a fixed external retriever over a fixed corpus/query
set):

| k | 5 | 10 | 25 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|
| `rag_any_gold@k` | 0.984 | 0.984 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 |
| `rag_all_golds@k` | 0.594 | 0.742 | 0.844 | 0.891 | 0.938 | 0.945 | 0.953 |

### Three-way comparison, same protocol

| | ground4layer batched-iso @20200 | pf32/indexed/lr_masked source @100000 | **CE-only arm @42800 (this eval)** |
|---|---|---|---|
| `llm_judge_accuracy` | 0.4609 | 0.4609 | **0.3828** |
| `llm_judge_score` | 2.578 | 2.813 | **2.359** |
| `lexical_grounding` | 0.591 | 0.621 | 0.619 |
| `doc_hit_rate` | n/a (telemetry OOM) | 0.992 | 0.992 |
| `mem_pos_weight_mass` | n/a (telemetry OOM) | 0.445 | **0.575** |

## Interpretation

The CE-only arm is worse on judged QA accuracy/score than its own un-further-trained source
checkpoint, despite 42,800 additional training steps and every in-training signal (per the
doc-access investigation's health-check log) reading as healthy and improving: `ce_loss` and
`doc_access_per_query_loss` both trending down, `mem_hit_rate` steady at 1.0,
`mem_pos_weight_mass` around 0.97–0.99 in-training. The retrieval-quality proxies available at
eval time here (`doc_hit_rate=0.992`, `mem_pos_weight_mass=0.575`) are equal-or-better than the
source checkpoint's, so the regression is not explained by retrieval getting worse — it looks more
like a change in generation behavior. Plausible, not-yet-checked explanations:

- The CE-only intervention fine-tunes the answer generation target on `<think>`-prefixed CoT text
  (see the doc-access investigation's ablation section) without changing what's retrievable — this
  may be shifting the model's generation style (e.g. always emitting reasoning-style text) in a
  way the judge penalizes for direct-answer accuracy, independent of retrieval quality.
- This is a mid-training checkpoint of an arm still actively training, not a converged end state —
  it may not be representative of where this arm settles.
- Single n=128 sample, same as both sibling evals — a 0.46→0.38 shift is ~10 examples' worth of
  difference on 128 samples; a real effect given its size, but worth another n=128 (or larger) draw
  before treating the exact magnitude as precise.

**Not yet done, worth doing next:** eval a checkpoint of Arm 2 (the CE-only arm's own source lineage,
*before* the CoT/CE-only branch) at a step count comparable to 42800's *total* additional training,
if one exists, to separate "more training in general" from "the CE-only intervention specifically"
as the cause of the regression.

## Follow-up: judging the chain-of-thought alone reveals a large answer-extraction gap

rohunagrawal observed, reading raw samples, that the model's `<think>` block frequently reasons to
the correct answer while the separately-extracted "Generated Answer" field is wrong or unrelated —
raising the question of how much of the regression above is a genuine capability drop vs. an
answer-extraction artifact. Re-judged this same eval's 128 samples with a **larger, independent
judge (Qwen3-8B)** shown **only the raw chain-of-thought** (extracted directly from the `generated`
field, not the precomputed `thinking` field — see caveat below) and **never the "Generated Answer"
field at all** — new script
[`scripts/misc/judge_cot_accuracy.py`](../../scripts/misc/judge_cot_accuracy.py).

| | accuracy |
|---|---|
| Original (Qwen3-4B, judges `generated_answer`), corrected for the parsing bug above | 0.3438 (44/128) |
| CoT-only (Qwen3-8B, judges the reasoning trace, never sees `generated_answer`) | **0.5078** (65/128) |
| CoT-only, restricted to the 113/128 samples whose `<think>` block actually closed | **0.5487** |

23 samples flipped wrong→correct under CoT-only judging; only 7 flipped correct→wrong. Clear
examples of the reasoning reaching the right answer while the extracted final answer doesn't:

- Q: "Bethpage State Parkway begins with an interchange at which highway?" GT: *Southern State
  Parkway*. CoT: "...The answer should be the Southern State Parkway." Extracted
  `generated_answer`: **"25.53 mi"** — an unrelated number from elsewhere in the context.
- Q: "Are Colocasia and Coronilla both flowering plants?" GT: *yes*. CoT: "...both are flowering
  plants... The answer is yes." Extracted `generated_answer`: **"Are both flowering plants"** —
  echoes the question, never states an answer.

**A distinct, separately-discovered bug surfaced along the way**: 15/128 samples' `<think>` block
never hit a closing `</think>` before `max_new_tokens`, and the precomputed `thinking` field comes
back **empty** for every one of them (not just partial) — silently discarding real reasoning
content. `judge_cot_accuracy.py` extracts the CoT directly from the raw `generated` text instead
(taking everything after `<think>` when unclosed, flagged `cot_truncated`) rather than trusting
the precomputed field.

**Resolved (2026-08-17): ran the identical CoT-only re-judge against the source checkpoint's own
eval JSON.** This changes the interpretation meaningfully:

| | source (step 100000) | CE-only arm (step 42800) |
|---|---|---|
| generated-answer accuracy (corrected for the [parsing bug](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md)) | 0.4297 (55/128) | **0.3438** (44/128) |
| CoT-only accuracy, all 128 samples | 0.4531 (58/128) | 0.5078 (65/128) |
| CoT-only accuracy, **complete (non-truncated) CoTs only** | **0.5392** (n=102) | **0.5487** (n=113) |
| `<think>` truncation rate (never hit `</think>`) | 26/128 (**20.3%**) | 15/128 (11.7%) |

**The complete-CoT accuracy is essentially identical between the two checkpoints (0.539 vs
0.549, indistinguishable at this sample size)** — the underlying reasoning, when it finishes,
reaches the ground truth about half the time either way. What differs sharply is the **gap
between that reasoning-quality number and the reported generated-answer accuracy**: 0.539−0.430
= **0.109** for the source vs. 0.549−0.344 = **0.205** for CE-only — the CE-only arm's final
answer diverges from its own reasoning's conclusion roughly **twice as often**. Combined with the
CE-only arm's *lower* truncation rate (it finishes thinking more reliably, if anything a point in
its favor), this points the regression specifically at **answer-extraction/formatting behavior
getting worse**, not at degraded reasoning or retrieval. This is also a more mechanistically
plausible fit for the CE-only intervention itself (training the CE target on `<think>`-prefixed
CoT text) than either the "just more training" or "wrong training-data distribution" theories —
though it doesn't rule those out as contributing, and n=128 per cell means these are suggestive,
not definitive, magnitudes.

## Reproducibility (CoT-only re-judge)

```bash
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
RUN_SCRIPT_PATH=scripts/embed/run_judge_cot_accuracy.sh \
RUN_ENV="INPUTS=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53/eval/step_42800/msa_hotpotqa_c10000_hybrid_autok.json" \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```
Sidecar result: `gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53/eval/step_42800/msa_hotpotqa_c10000_hybrid_autok_cot_accuracy.json`.

Same command against the source checkpoint (for the comparison table above):
```bash
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
RUN_SCRIPT_PATH=scripts/embed/run_judge_cot_accuracy.sh \
RUN_ENV="INPUTS=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/eval/step_100000/msa_hotpotqa_c10000_hybrid_autok.json" \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```
Sidecar result: `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/eval/step_100000/msa_hotpotqa_c10000_hybrid_autok_cot_accuracy.json`.

## Reproducibility

```bash
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
RUN_SCRIPT_PATH=scripts/embed/eval_msa_hybrid.sh \
RUN_ENV="DS=hotpotqa RUN_DIR=multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53 STEP=42800" \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

- **Commit:** `b4d9541` on `multihop-finetuning`, plus the (at-the-time) uncommitted
  `models/memory.py` / `models/conv_utils.py` fixes from the two 2026-08-13 evals — synced via the
  launcher's tree tar, not yet committed as of this eval. See the two implementation notes linked
  above for their exact diffs.
- **Checkpoint:** `gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53/qwen3_mem_embed/42800`
- **Result JSON:** `gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-23-34-53/eval/step_42800/msa_hotpotqa_c10000_hybrid_autok.json`
- **wandb:** logged into the training run at `train_step=42800` —
  `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-23-34-53`;
  standalone eval run `johnzhang2366-columbia-university/memory-layers-eval/x9303625`.
- **TPU:** `v6e-8` (`rohun-v6e-8-0`, `us-east1-d`, project `memorylayers`).
