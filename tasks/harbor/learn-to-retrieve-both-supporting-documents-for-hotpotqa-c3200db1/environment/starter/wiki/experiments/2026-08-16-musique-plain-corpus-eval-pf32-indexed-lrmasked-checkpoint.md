# Plain (non-hybrid) musique corpus eval: qa_hard_neg_think_sft4b pf32/indexed/lr_masked, step 100000

**Date:** 2026-08-16 · **Author:** claude (session with rohunagrawal) · **Status:** done.

> **Checked against the 2026-08-17 judge-parsing fix** (see
> [2026-08-17-llm-judge-last-tag-parsing-fix.md](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md))
> **and found unaffected**: zero samples in this eval's judge output had more than one
> `<judgement>` tag, so `llm_judge_accuracy` below is unchanged by the fix (0.1953 both ways).
> The *hybrid* eval it's compared against below **was** affected (0.3203→0.2812 corrected) —
> the "~40% relative" framing below should read as **~31% relative** (0.2812→0.1953) against
> the corrected hybrid number.

## Conclusion

**`llm_judge_accuracy = 0.1953` (25/128), `doc_hit_rate = 1.0`, `doc_access_acc = 0.0815`,
`doc_token_hit_rate = 0.653`, `mem_pos_weight_mass = 0.184`, `mem_top1_weight = 0.300`,
`mem_effective_slots = 14.4`, `mem_topk_entropy = 2.87`** on the plain, non-hybrid full-corpus
memory eval (no RAG assist — the model retrieves entirely through its own trained `mem_lookup`
over the whole doc corpus) for the same checkpoint evaluated on the RAG→memory hybrid protocol in
[2026-08-16-musique-hybrid-pf32-indexed-lrmasked-checkpoint.md](2026-08-16-musique-hybrid-pf32-indexed-lrmasked-checkpoint.md).

**Removing the RAG assist costs about 40% relative judge accuracy** (0.320 → 0.195) despite
`doc_hit_rate` staying at a perfect 1.0 in both modes — i.e. the memory layer still finds *a*
correct doc in its own top-k just as reliably as the RAG-gathered bank does, but
`mem_pos_weight_mass` drops sharply (0.371 → 0.184) and `mem_effective_slots≈14` (out of
`mem_top_k=64`) shows the softmax mass spread thin across many slots rather than concentrated on
the right ones. This is the expected shape for this codebase (RAG assist exists precisely to
narrow the candidate set the memory layer has to discriminate within) but is a useful quantified
data point on how much of this checkpoint's hybrid-mode accuracy comes from the RAG narrowing
itself vs. the memory layer's own learned retrieval.

## Hypothesis & motivation

rohunagrawal, after the hybrid musique eval, asked to run "just normal eval" on the same
checkpoint — i.e. without the RAG→memory hybrid's per-query gathered bank — as a baseline
comparison. No prior "normal"/plain-corpus eval existed for this checkpoint on musique; this is the
first data point of that kind for it.

## Setup

- **Checkpoint:** same as the hybrid eval —
  `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45`,
  step 100000 (single memory layer, `mem_layers=[14]`, `mem_top_k=64`, `embed_conv: true`).
- **Task:** `gen_large_mem_msa_musique.yaml` — plain `generation_large_mem` evaluator (overridden
  from the yaml's default `generation_large_mem_msa`, which targets the separate MSA-4B
  architecture our checkpoint doesn't have), full doc corpus (`max_docs: null`,
  `max_chunks_per_doc=2`, `chunk_size=256`), n=128, `dataset.batch_size=8`, `tp_devices=1` (the
  mem-model checkpoint loader isn't TP-aware). **No RAG**: no `gather_bank`, no
  `inject_query_gold`, no per-query bank — the model's own memory layer retrieves over the entire
  encoded corpus via its trained `mem_lookup`.
- **Metrics available:** only what the base `generation_large_mem` config computes —
  `llm_judge_accuracy` (no `llm_judge_score`/`lexical_grounding`, unlike the hybrid task config
  which wires those in explicitly) plus the evaluator's own retrieval-quality telemetry
  (`doc_access_acc`, `doc_hit_rate`, `doc_token_hit_rate`, `mem_pos_weight_mass`,
  `mem_top1_weight`, `mem_effective_slots`, `mem_topk_entropy`) — richer retrieval telemetry than
  the hybrid eval exposes, at the cost of no judge-score/grounding numbers to compare directly.
- **New tooling:** ran via the newly-added
  [`scripts/embed/eval_msa_plain.sh`](../../scripts/embed/eval_msa_plain.sh) — see
  [implementation note](../implementations/2026-08-16-msa-plain-corpus-eval-runner.md); no prior
  script targeted a one-shot checkpoint with this eval path.
- **Box:** `rohun-v6e-8-0` (`us-east1-d`, project `memorylayers`), kept separate from
  `rohun-v6e-8-1` (actively training the unrelated CE-only arm). Verified idle before launching.

## Results

| metric | value |
|---|---|
| `llm_judge_accuracy` | **0.1953** (25/128) |
| `doc_hit_rate` | 1.0 |
| `doc_access_acc` | 0.0815 |
| `doc_token_hit_rate` | 0.653 |
| `mem_pos_weight_mass` | 0.184 |
| `mem_top1_weight` | 0.300 |
| `mem_effective_slots` | 14.4 (of `mem_top_k=64`) |
| `mem_topk_entropy` | 2.867 |
| `generated_count` | 128 / 128 |

### Hybrid vs. plain, same checkpoint, same dataset (musique)

| metric | hybrid ([2026-08-16](2026-08-16-musique-hybrid-pf32-indexed-lrmasked-checkpoint.md)) | plain (this eval) |
|---|---|---|
| `llm_judge_accuracy` | 0.3203 | **0.1953** |
| `doc_hit_rate` | 1.0 | 1.0 |
| `mem_pos_weight_mass` | 0.371 | **0.184** |

## Interpretation

`doc_hit_rate=1.0` in both modes means the memory layer's own top-k always contains at least one
correct document even without RAG narrowing the candidate pool first — so the accuracy gap isn't a
"can't find the doc at all" failure. What differs sharply is how concentrated the retrieved mass
is once found: `mem_pos_weight_mass` roughly halves (0.371→0.184) and `mem_effective_slots≈14` (out
of a `mem_top_k=64` budget) shows the softmax spreading meaningfully across many candidate slots
rather than concentrating on the true positives — consistent with the "retrieval isn't the
bottleneck, mass gets diluted across a larger candidate pool" pattern this repo has seen before
(e.g. the 2026-07-22 MSA sweep, the 2026-08-13 pf32/indexed/lr_masked hotpotqa eval's
interpretation section). Without RAG's per-query pre-filtering, the memory layer has to discriminate
across the *entire* corpus's candidates rather than a pre-narrowed set, and that dilution
propagates through to a meaningfully worse final judge accuracy (0.32→0.20).

The very low `doc_access_acc=0.0815` alongside a perfect `doc_hit_rate=1.0` is the same open
nuance flagged in the CE-only arm's hybrid eval write-up (`doc_access_acc` staying low despite
near-ceiling hit rate) — not re-investigated here, but worth resolving generally rather than
per-eval, since it recurs across unrelated checkpoints/datasets.

**Caveat:** n=128, no judge-score/grounding metrics available for this task (only
`llm_judge_accuracy` is wired into the base `generation_large_mem_msa` config's `metrics:` block),
so this comparison is accuracy-only, not the fuller 3-metric comparison used elsewhere in this
family of write-ups.

## Reproducibility

```bash
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
RUN_SCRIPT_PATH=scripts/embed/eval_msa_plain.sh \
RUN_ENV="DS=musique RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45 STEP=100000" \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

- **Commit:** `b4d9541` on `multihop-finetuning`, plus the new (uncommitted at the time of this
  eval) `scripts/embed/eval_msa_plain.sh`.
- **Checkpoint:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/qwen3_mem_embed/100000`
- **Result JSON:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/eval/step_100000/msa_musique_corpus.json`
- **wandb:** logged into the training run at `train_step=100000` —
  `johnzhang2366-columbia-university/memory-layers/qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-08-09-05-49-45`;
  standalone eval run `johnzhang2366-columbia-university/memory-layers-eval/rdqouo1g`.
- **TPU:** `v6e-8` (`rohun-v6e-8-0`, `us-east1-d`, project `memorylayers`).
