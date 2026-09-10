# HotpotQA hybrid eval + GRPO-readiness diagnostic: qa_hard_neg_think_sft4b, step 90000

**Date:** 2026-08-24 · **Author:** rohunagrawal (with Claude) · **Status:** done

## Conclusion

**This checkpoint is a genuinely reasonable GRPO candidate on this eval: 42.2% of questions
have real intra-group reward variance (correct AND incorrect samples in the same group of 4),
and pass@k climbs steadily from pass@1 = 0.488 to pass@4 = 0.695 (+20.7 points) — the model
frequently *can* produce a correct answer under sampling, it just doesn't do so reliably every
time, which is exactly the regime GRPO is designed to exploit.** The caveat: 30.5% of questions
never got a correct answer in any of 4 samples (temperature 0.6) — GRPO provides no direct
gradient signal for that fraction at this group size, so it would need to come from elsewhere
(more samples, better retrieval, or the SFT stage itself). Separately, on the standard hotpotqa
hybrid full-corpus eval, this checkpoint scores `llm_judge_accuracy = 0.453` (greedy, n=128,
already on record before this run) / `pass@1 = 0.488` (mean over temp-0.6 samples, this run) —
in the same range as this checkpoint lineage's other arms (0.43–0.46), not a regression.

## Hypothesis & motivation

Requested: check whether `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55`
step 90000 (mid-trained on the mixed `qa_hard_neg_think_sft4b` corpus, `qwen3_mem_embed`,
`mem_top_k=64`) could be improved with GRPO, and separately, see its performance on the standard
hotpotqa RAG→memory hybrid full-corpus eval. No GRPO/RL training infrastructure exists anywhere
in this repo (verified via full grep + reading every training doc — see
[2026-08-24-grpo-readiness-multisample-hybrid-eval](../implementations/2026-08-24-grpo-readiness-multisample-hybrid-eval.md)),
so before building an actual training loop, the cheaper question is: **at the current
checkpoint, does sampling k completions per question ever produce a mix of correct and
incorrect answers?** If most groups are uniformly all-correct or all-incorrect, GRPO's
group-normalized advantage is zero almost everywhere and there's no signal to train on.

## Setup

- **Checkpoint:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55/qwen3_mem_embed/90000`.
  `qwen3_mem_embed`, `mem_top_k=64`, trained via `scripts/embed/train_hard_neg_think.sh`
  (`trainer=staged_telemetry`, dataset `qa_hard_neg_think_sft4b`).
- **Eval:** `generation_large_mem_rag_hybrid` (RAG→memory hybrid), task
  `gen_large_mem_msa_hotpotqa_hybrid` — full 9,811-doc corpus (`gather_bank: true`), auto-K over
  `{5,10,25,50,100,150,200}` at threshold 0.96 (landed on the 200 cap, same as every other
  checkpoint in this lineage), judge `Qwen/Qwen3-4B` (`tensor_parallel_size=4`) for
  `llm_judge_accuracy`/`llm_judge_score`, `lexical_grounding`. 128 queries
  (`ragrawal36/msa-hotpotqa-qa-with-ids`), `max_new_tokens=512`.
- **Two arms, same checkpoint/corpus/judge, different decode:**
  - **Baseline** (already on record, not re-run here): `eval_msa_hybrid.sh DS=hotpotqa` — greedy
    decode (`temperature=0.0`), 1 sample/question, 128 total generations.
  - **Multi-sample** (this run): `eval_msa_hybrid_multisample.sh` — `temperature=0.6, top_k=20,
    top_p=0.95`, **k=4 independent samples/question** via the hybrid evaluator's new
    `multi_sample` mode ([implementation note](../implementations/2026-08-24-grpo-readiness-multisample-hybrid-eval.md)):
    tiled generation replicated across the box's full mesh data axis, k = mesh data-axis size.
    **k=4 because the box is a single-host v6e-4** (`tpu-v6e-4-flex`, `ct6e-standard-4t`,
    `memory-layers` project, `europe-west4-a`) — chosen specifically so k matches the box shape
    with zero wasted chips. 128 queries × 4 samples = 512 total generations, all judged.
- **GRPO-readiness metrics** (post-hoc, `scripts/analysis/grpo_readiness_metrics.py`, on the
  multi-sample arm's judged output): unbiased pass@k (Chen et al. 2021 estimator) for k=1..4,
  intra-group variance of both binary `llm_judge_accuracy` and continuous `llm_judge_score`,
  and the all-correct / all-incorrect / mixed group-composition split.

## Results

**Hotpotqa hybrid full-corpus eval — both decode arms:**

| Arm | `llm_judge_accuracy` | `llm_judge_score` | `lexical_grounding` | `doc_hit_rate` | `mem_pos_weight_mass` |
|---|---|---|---|---|---|
| Baseline (greedy, n=128, 1 sample/q) | 0.4531 | 2.8203 | 0.6263 | 0.9922 | 0.4548 |
| Multi-sample (temp=0.6, n=128×4=512) | 0.4883 | 2.9414 | 0.6314 | 0.9922 | 0.4478 |

Retrieval telemetry (`doc_hit_rate`, `mem_pos_weight_mass`) is essentially identical between
arms, as expected — retrieval doesn't depend on decode temperature. `llm_judge_accuracy` in the
multi-sample arm is the mean over all 512 samples (equivalently, `pass@1` below) — slightly
*higher* than greedy here, within the noise expected from temperature-0.6 sampling at this
sample size (binomial SE at p≈0.47, n=128 is ≈0.044; at n=512 samples it's tighter, ≈0.022, so
the 0.035 gap is at the edge of noise, not a clear effect).

**GRPO-readiness diagnostic (multi-sample arm, 128 groups of k=4):**

| k | pass@k (unbiased estimator) |
|---|---|
| 1 | 0.4883 |
| 2 | 0.6042 |
| 3 | 0.6602 |
| 4 | 0.6953 |

| Metric | Value |
|---|---|
| Mean intra-group binary variance | 0.0869 |
| Mean intra-group score variance (0-5 scale) | 1.6055 |
| Groups all-correct (4/4) | 27.3% |
| Groups all-incorrect (0/4) | 30.5% |
| **Groups mixed (1-3/4 correct) — nonzero GRPO advantage** | **42.2%** |

## Interpretation

- **42.2% of prompts carry real reward variance at k=4** — for these, GRPO's group-normalized
  advantage is nonzero and there's an actual gradient signal pushing probability mass toward the
  sampled-correct completions. This is a solid majority-adjacent fraction, not a rare edge case
  — the checkpoint is not so far along that it's already saturated (all-correct) nor so far
  behind that it's stuck (all-incorrect) on most questions.
- **pass@k's steady climb (0.488 → 0.695 over k=1..4) is the more informative signal than the
  variance number alone**: it means when the model fails on its first try, a meaningfully large
  fraction of the time a *different* sample from the same prompt succeeds. That's precisely what
  GRPO can exploit — it doesn't need to teach the model a new capability, just make the
  already-sometimes-correct behavior more reliable.
- **The 30.5% all-incorrect fraction is the real caveat.** For roughly a third of questions,
  none of 4 samples at temperature 0.6 ever landed a correct answer — GRPO gets zero signal from
  these at this group size. Worth checking whether a larger k (e.g. 8 or 16, via a bigger box)
  shrinks this fraction meaningfully before concluding GRPO's ceiling here, since some of these
  may just be low-probability-but-nonzero successes that 4 samples didn't happen to catch.
- **The full-corpus hybrid numbers (0.45-0.49 `llm_judge_accuracy`) sit comfortably in this
  checkpoint lineage's established range** (0.43-0.46 across the sibling checkpoints logged in
  [2026-08-13](2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md) and
  [2026-08-23](2026-08-23-hotpotqa-hard-neg-reasoning-finetune-checkpoint-sweep.md)) — this is a
  healthy, non-regressed checkpoint to build on, not an outlier in either direction.
- **Not tested here:** whether GRPO training actually improves the metric (this is a pre-training
  diagnostic only), and whether a different reward function (e.g. `lexical_grounding` or a
  combination) would show a different mixed-group fraction than the binary judge match used here.

## Reproducibility

```bash
# Baseline (greedy, already on record before this run — not re-run)
DS=hotpotqa RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55 \
STEP=90000 bash scripts/embed/eval_msa_hybrid.sh

# Multi-sample GRPO-readiness run (this experiment)
TPU_NAME=tpu-v6e-4-flex ZONE=europe-west4-a PROJECT_ID=memory-layers TRANSPORT=gce \
RUN_SCRIPT_PATH=scripts/embed/eval_msa_hybrid_multisample.sh \
RUN_ENV="DS=hotpotqa RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55 STEP=90000 NUM_SAMPLES=128" \
bash scripts/infrastructure/multi-vm-tpu-run.sh

# Post-hoc GRPO-readiness metrics
uv run python scripts/analysis/grpo_readiness_metrics.py \
  gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55/eval/step_90000/msa_hotpotqa_c10000_hybrid_multisample.json
```

- **Commit:** _TBD_ — as of this write-up, `evals/gen_large_mem_rag_hybrid.py` (`multi_sample`),
  `configs/eval/generation_large_mem_rag_hybrid.yaml` (new `multi_sample: false` default),
  `utils.py` (`load_inference_checkpoint` restore-args fix), `scripts/embed/eval_msa_hybrid_multisample.sh`,
  `scripts/analysis/grpo_readiness_metrics.py`, `tests/test_grpo_readiness_metrics.py` are
  uncommitted on `multihop-finetuning`.
- **Result JSONs:**
  `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55/eval/step_90000/msa_hotpotqa_c10000_hybrid_autok.json` (baseline, pre-existing),
  `.../msa_hotpotqa_c10000_hybrid_multisample.json` (multi-sample, this run).
- **wandb:** eval run
  [z029wsa1](https://wandb.ai/johnzhang2366-columbia-university/memory-layers-eval/runs/z029wsa1);
  metrics also mirrored into the training run
  [qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-08-22-17-39-55](https://wandb.ai/johnzhang2366-columbia-university/memory-layers/runs/qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-08-22-17-39-55)
  under `eval/msa_hotpotqa_c10000_hybrid_multisample/*`.
- **TPU:** single-host v6e-4 (`ct6e-standard-4t`), `tpu-v6e-4-flex`, `memory-layers` project
  (DWS flex-start), `europe-west4-a`.

## Follow-ups

- Try a larger k (needs a bigger single-host box, e.g. v6e-8 → k=8) to see whether the 30.5%
  all-incorrect fraction shrinks — distinguishes "genuinely unreachable under this policy" from
  "just unlucky at k=4."
- If GRPO training is pursued: this diagnostic doesn't exist as reusable training
  infrastructure — no GRPO trainer, loss, or rollout loop exists in this repo (confirmed via
  grep; see the implementation note). Would be new work, though the untracked local `maxtext/`
  checkout has a reference GRPO trainer worth consulting for design.
- Not checked: whether `lexical_grounding` (a different reward signal) gives a different
  mixed-group fraction than the binary judge match used here.
