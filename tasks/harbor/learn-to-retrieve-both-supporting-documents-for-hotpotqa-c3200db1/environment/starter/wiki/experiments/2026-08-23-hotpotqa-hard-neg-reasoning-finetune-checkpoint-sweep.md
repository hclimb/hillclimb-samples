# HotpotQA hard-neg-reasoning finetune: checkpoint sweep on the hotpotqa hybrid eval

**Date:** 2026-08-23 · **Author:** rohunagrawal (with Claude) · **Status:** done

## Conclusion

**The finetune regresses the hotpotqa RAG→memory hybrid eval almost immediately, then plateaus —
it does not keep getting worse, and it does not recover.** `llm_judge_accuracy` briefly matches
or beats the source checkpoint's own baseline at step 500 (**0.461** vs. baseline **0.430**
corrected / 0.461 as-originally-reported), then collapses to a noisy **0.22–0.36** band by step
1000–2000 and stays there through step 10000 — no further net decay, no recovery. `doc_hit_rate`
stays pinned at **0.992** at every single step (retrieval still finds a correct document almost
every time); the metric that actually moves is `mem_pos_weight_mass`, which falls from **0.387**
(step 500) to **0.14–0.19** by step 2000 and never recovers. So this isn't "the model forgot how
to retrieve" — it's "the model stopped concentrating attention on what it retrieved." The
collapse window (step 500→1000) coincides with the LR schedule reaching peak (`warmup_frac=0.1`
over `steps=10000` = a 1000-step warmup) — the next thing worth testing is a lower peak LR or a
longer warmup before concluding the dataset itself is unusable for this recipe.

## Hypothesis & motivation

Follow-up to
[2026-08-22-hotpotqa-hard-neg-reasoning-finetune-setup.md](../implementations/2026-08-22-hotpotqa-hard-neg-reasoning-finetune-setup.md):
that implementation note's finetune run showed training loss dropping smoothly from step 1 to
step 10000 with no obvious cliff, but a single hybrid-eval point at the "epoch 1" checkpoint
(step 5000) already showed a large regression against the source checkpoint. Loss curves alone
can't distinguish "gradually got worse at this task" from "broke early and then just sat there,"
and the two imply very different fixes (LR/warmup/data-mix tuning vs. an early-stopping recipe
vs. abandoning the dataset). Swept 9 more checkpoints across the run to find the actual shape.

## Setup

- **Base checkpoint (finetune source):**
  `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45`,
  step 100000. `qwen3_mem_embed`, single memory layer (`mem_layers=[14]`), `mem_size=16384`,
  `mem_top_k=64`, `embed_model` Qwen3-Embedding-0.6B with `embed_conv: true`.
- **Finetune:** `hotpotqa_hard_neg_reasoning_finetune_topk64_seq512_chunks16_bs16-2026-08-22-19-08-24`
  — warm-started (weights only, fresh optimizer, step 0) onto
  `vm2825/hotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1` (78,755 rows, 1
  `pos_doc` + 3 `neg_docs`/row, `neg_score_threshold=0.0` since every row's `neg_scores` is a
  constant `0.0` placeholder — see the implementation note for how that was confirmed).
  `trainer=midtraining_full_telemetry`: single cosine stage, `mem_*`/`embed_model`/**all** of
  `main_model` trainable, `learning_rates={mem: 1e-4, embed: 1e-5, main: 1e-5}`,
  `warmup_frac=0.1`, `steps=10000` (≈2.03 epochs @ batch 16), `checkpoint_interval=500`.
- **Eval protocol (every point, identical):** `gen_large_mem_msa_hotpotqa_hybrid` — RAG→memory
  hybrid, full 9,811-doc corpus (`gather_bank: true`), auto-K over `{5,10,25,50,100,150,200}` at
  threshold 0.96 (landed on the 200 cap at every step — same fixed external retriever/corpus
  regardless of checkpoint), `B=8` data-parallel, `max_new_tokens=512`, `n=128`, judge
  `Qwen/Qwen3-4B` (`tensor_parallel_size=4`) for `llm_judge_accuracy`/`llm_judge_score`.
- **Baseline (source checkpoint, same protocol, step 100000):** `llm_judge_accuracy=0.4297`
  (corrected — see the 2026-08-17 correction note on
  [2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md](2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md)),
  `llm_judge_score=2.8125`, `lexical_grounding=0.6211`, `doc_hit_rate=0.9922`,
  `mem_pos_weight_mass=0.4454`.
- **Swept steps:** 500, 1000, 1500, 2000, 2500, 3500, 5000, 6500, 8000, 10000 — denser early
  (where the collapse turned out to be), coarser late.

## Results

| step | epochs | `llm_judge_accuracy` | `llm_judge_score` | `lexical_grounding` | `doc_hit_rate` | `mem_pos_weight_mass` |
|---|---|---|---|---|---|---|
| baseline (100000, source ckpt) | — | 0.4297 | 2.8125 | 0.6211 | 0.9922 | 0.4454 |
| 500 | 0.10 | **0.4609** | 2.9141 | 0.6259 | 0.9922 | 0.3865 |
| 1000 | 0.20 | 0.3203 | 2.5781 | 0.5971 | 0.9922 | 0.2638 |
| 1500 | 0.30 | 0.3594 | 2.7031 | 0.5792 | 0.9922 | 0.2434 |
| 2000 | 0.41 | 0.2578 | 1.9922 | 0.5283 | 0.9922 | 0.1412 |
| 2500 | 0.51 | 0.2656 | 1.9922 | 0.5331 | 0.9922 | 0.1939 |
| 3500 | 0.71 | 0.3203 | 2.3203 | 0.5595 | 0.9922 | 0.1514 |
| 5000 | 1.02 | 0.2969 | 1.8516 | 0.5178 | 0.9922 | 0.1603 |
| 6500 | 1.32 | 0.2188 | 2.1250 | 0.5315 | 0.9922 | 0.1424 |
| 8000 | 1.63 | 0.2422 | 2.0781 | 0.5245 | 0.9922 | 0.1605 |
| 10000 | 2.03 | 0.2578 | 2.0078 | 0.5214 | 0.9922 | 0.1387 |

`rag_top_k` (auto-K) was 200 (the cap) at every single point, baseline included — the retrieval
pre-pass is checkpoint-independent, so this is expected, not a finding.

## Interpretation

- **The regression is a step, not a slope.** From step 500→1000, `llm_judge_accuracy` drops
  0.461→0.320 (a ~0.14 absolute swing, well outside this eval's sampling noise at n=128 — the
  binomial SE at p≈0.3 is ~0.04) and `mem_pos_weight_mass` drops 0.387→0.264. From step 2000
  onward, every metric oscillates within a band roughly consistent with n=128 sampling noise
  around a flat mean (accuracy 0.22–0.32, score 1.85–2.32, `mem_pos_weight_mass` 0.14–0.19) —
  no visible further decay and no recovery through step 10000 (2.03 epochs). **Training loss
  gave no warning of this** (it decreases smoothly across the whole run per the implementation
  note) — the loss curve and this eval are measuring different things.
- **`doc_hit_rate` never moves (0.9922 at every step, matching baseline exactly).** Retrieval
  finding *a* correct document in the bank is not the mechanism that breaks. `mem_pos_weight_mass`
  is: the softmax weight the read channel places on correct-document slots collapses by ~2.3–3×
  within the first ~2000 steps and stays collapsed. This is the same "retrieval isn't the
  bottleneck, mass gets diluted" pattern documented elsewhere in this repo, but here it's
  *induced by finetuning itself*, not by corpus scale.
- **Timing coincides with the LR schedule reaching peak.** `warmup_frac=0.1` over `steps=10000`
  is a 1000-step linear warmup to `learning_rates.mem=1e-4` (main/embed peak at 1e-5). The
  collapse window (500→1000) is exactly the second half of that ramp. Plausible mechanism: at
  peak LR, unfreezing the *entire* `main_model` (not just layers 13/14/15, per this finetune's
  choice — see the implementation note's rationale) combined with a dataset where every
  negative carries an identical placeholder score (no ranking signal among negatives) may let
  gradient updates overwrite the read channel's learned selectivity faster than this dataset's
  78,755 rows can re-teach it. **Not verified — this is the leading hypothesis, not a confirmed
  cause.** A lower peak LR, a longer warmup, or reverting to the layers-13/14/15-only trainable
  set (`midtraining_telemetry` instead of `midtraining_full_telemetry`) are the natural next
  probes, in roughly that order of cost.
- **Practical takeaway for this checkpoint lineage:** if the goal was "improve on hotpotqa
  hybrid via this finetune," it did not work at *any* sampled step — even the best point
  (step 500, 0.10 epochs) is a tie with baseline, not a win, and it degrades from there. An
  early-stopping recipe wouldn't help here; the useful checkpoint (if any) would have to come
  from a *different* LR/warmup/trainable-set choice, not from stopping this run sooner.

## Reproducibility

```bash
# Finetune (produced the checkpoints swept below)
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
RUN_SCRIPT_PATH=scripts/embed/train_hotpotqa_hard_neg_reasoning_finetune.sh \
  bash scripts/infrastructure/multi-vm-tpu-run.sh

# Checkpoint sweep on the hotpotqa hybrid eval
DS=hotpotqa \
RUN_DIR=hotpotqa_hard_neg_reasoning_finetune_topk64_seq512_chunks16_bs16-2026-08-22-19-08-24 \
STEPS="500_1000_1500_2000_2500_3500_6500_8000_10000" \
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
RUN_ENV="DS=$DS RUN_DIR=$RUN_DIR STEPS=$STEPS" \
RUN_SCRIPT_PATH=scripts/embed/sweep_msa_hybrid_ckpt.sh \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
# step 5000 was already scored individually with scripts/embed/eval_msa_hybrid.sh (DS=hotpotqa)
# before this sweep script existed; sweep_msa_hybrid_ckpt.sh skips already-published steps.
```

- **Commit:** _TBD_ — as of this write-up, `configs/dataset/{hotpotqa_hard_neg_reasoning_modified_finetune.yaml,sources/hotpotqa_hard_neg_reasoning_modified.yaml}`,
  `configs/trainer/midtraining.yaml` (added `learning_rates`),
  `scripts/embed/{train_hotpotqa_hard_neg_reasoning_finetune.sh,sweep_msa_hybrid_ckpt.sh}`,
  `scripts/misc/download_hotpotqa_hard_neg_reasoning_data.sh` are uncommitted on
  `multihop-finetuning`.
- **Checkpoints:** `gs://memory-layers-training/hotpotqa_hard_neg_reasoning_finetune_topk64_seq512_chunks16_bs16-2026-08-22-19-08-24/qwen3_mem_embed/{500,1000,...,10000}`.
- **Eval result JSONs:** `gs://memory-layers-training/hotpotqa_hard_neg_reasoning_finetune_topk64_seq512_chunks16_bs16-2026-08-22-19-08-24/eval/step_<N>/msa_hotpotqa_c10000_hybrid_autok.json`.
- **wandb (training):** `johnzhang2366-columbia-university/memory-layers`, run
  `hotpotqa_hard_neg_reasoning_finetune_top-2026-08-22-19-08-24` (eval metrics logged into the
  same run at each `train_step`).
- **TPU:** v6e-8, `rohun-v6e-8-0`, `memorylayers` project, `us-east1-d`.

## Follow-ups

- Test the LR/warmup hypothesis directly: rerun with a lower peak `mem` LR (e.g. 3e-5 or 1e-5)
  and/or a longer warmup, same steps, same eval protocol, same checkpoints swept.
- Try `midtraining_telemetry` (layers 13/14/15 only) instead of `midtraining_full_telemetry` on
  this same dataset, to isolate whether unfreezing the *whole* `main_model` is the culprit vs.
  the dataset/LR combination.
- `lexical_grounding` tracks `llm_judge_accuracy` closely at every step in this sweep (both drop
  together, both plateau together) — consistent, not yet independently informative.
