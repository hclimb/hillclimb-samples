# Hybrid musique (full corpus) eval: qa_hard_neg_think_sft4b pf32/indexed/lr_masked, step 100000

**Date:** 2026-08-16 · **Author:** claude (session with rohunagrawal) · **Status:** done.

> **CORRECTION (2026-08-17):** `llm_judge_accuracy` below was inflated by a judge-output
> parsing bug (`_parse_score` took the first `<judgement>` tag instead of the model's actual
> final one) — see
> [2026-08-17-llm-judge-last-tag-parsing-fix.md](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md).
> Corrected (re-parsing the same stored judge output, no re-inference): **`llm_judge_accuracy =
> 0.2812` (36/128)**, down from the 0.3203 (41/128) reported below; 5 samples flipped. The
> hotpotqa comparison this write-up draws (0.4609 vs. 0.3203) also shifts: corrected hotpotqa is
> 0.4297 (see
> [2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md](2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md)),
> so the corrected gap is **0.4297 vs. 0.2812** — musique is still substantially harder, same
> conclusion, slightly larger gap.

## Conclusion

**`llm_judge_accuracy = 0.3203` (41/128), `llm_judge_score = 1.984` (1–5), `lexical_grounding =
0.484`, `doc_hit_rate = 1.0`, `mem_pos_weight_mass = 0.371`** on the RAG→memory hybrid musique
task, full 10,000-doc corpus, auto-K capped at 200.

**Musique is substantially harder than hotpotqa for this exact checkpoint** — same weights, same
protocol, same day:

| metric | hotpotqa ([2026-08-13](2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md)) | musique (this eval) |
|---|---|---|
| `llm_judge_accuracy` | 0.4609 | **0.3203** |
| `llm_judge_score` | 2.813 | **1.984** |
| `lexical_grounding` | 0.621 | **0.484** |
| `doc_hit_rate` | 0.992 | 1.0 |
| `mem_pos_weight_mass` | 0.445 | 0.371 |
| `rag_all_golds@200` (bank coverage) | 0.953 | **0.680** |

`doc_hit_rate` is actually *higher* on musique (1.0 vs 0.992 — the retriever always finds at least
one correct doc), but `rag_all_golds@200` (all golds simultaneously in the auto-K bank) drops to
0.68 from 0.953, and `mem_pos_weight_mass` drops too — consistent with musique's harder multi-hop
structure (more required supporting docs per question) diluting the bank's positive-mass share
even before generation, which then shows up as lower judged accuracy/grounding.

## Hypothesis & motivation

rohunagrawal asked to re-run the exact eval from
[2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md](2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md)
— same checkpoint, same protocol — swapping the dataset from hotpotqa to musique, to get a second
data point on where this checkpoint lands. No specific hypothesis beyond the comparison.

## Setup

Identical to the 2026-08-13 hotpotqa eval of this checkpoint in every respect except `DS`:

- **Checkpoint:** `qwen3_mem_embed`, single memory layer (`mem_layers=[14]`), `mem_size=16384`,
  `mem_top_k=64`, `mem_placement=after_attention`, `mem_o_proj_zero_init=false`, no product keys,
  no batched isolation / per-query isolation (`mem_lookup` default path). `embed_model`:
  Qwen3-Embedding-0.6B with `embed_conv: true` (kernel=1, stride=1).
  `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45`,
  step 100000 (unchanged from the 2026-08-13 eval — still the latest checkpoint in this run-dir).
- **Task / protocol:** `gen_large_mem_msa_musique_hybrid` (`DS=musique`), full corpus
  (`target_docs=10000` — musique's actual corpus is 10,000 docs, so this is the full corpus),
  `gather_bank: true`, `B=8` data-parallel, `max_new_tokens=512`, `MEM_APPROX_TOPK=1`, n=128
  queries, auto-K over `{5,10,25,50,100,150,200}` at threshold 0.96 (not cleared — capped at 200,
  same cap behavior as the hotpotqa eval, though the underlying coverage curve differs — see
  Results).
- **Judge:** `llm_judge_accuracy` + `llm_judge_score` (1–5), Qwen3-4B, `tensor_parallel_size=4`,
  plus `lexical_grounding`.
- **Box:** `rohun-v6e-8-0` (`us-east1-d`, project `memorylayers`) — kept separate from
  `rohun-v6e-8-1`, which is actively training the unrelated CE-only arm (see
  [2026-08-13-doc-access-per-query-loss-investigation.md](2026-08-13-doc-access-per-query-loss-investigation.md)).
  Verified idle/clean before launching. No new code fixes needed — this checkpoint's
  `apply_conv1d` dtype-cast requirement (from the 2026-08-13 hotpotqa eval of the same checkpoint)
  was already fixed and is dataset-independent.
- Runner: `scripts/embed/eval_msa_hybrid.sh` (`DS=musique`).

## Results

| metric | value |
|---|---|
| `llm_judge_accuracy` | **0.3203** (41/128) |
| `llm_judge_score` (1–5) | **1.984** |
| `lexical_grounding` | 0.484 |
| `doc_hit_rate` | 1.0 |
| `mem_pos_weight_mass` | 0.371 |
| `rag_top_k` (auto-K) | 200 (**CAP** — no candidate ≥ 0.96) |
| `rag_any_gold@200` | 1.0 |
| `rag_all_golds@200` (bank coverage) | 0.680 |
| `rag_all_golds@10` / `@100` | 0.273 / 0.609 |
| `corpus_docs` | 10,000 (full corpus) |
| `bank_slots` | 2,560,000 |
| `mean_active_bank_slots` (gather-bank, per query) | 25,855 |
| `generated_count` | 128 / 128 |

Full auto-K coverage curve (retrieval pre-pass, corpus-wide, checkpoint-independent):

| k | 5 | 10 | 25 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|
| `rag_any_gold@k` | 0.945 | 0.977 | 0.984 | 1.0 | 1.0 | 1.0 | 1.0 |
| `rag_all_golds@k` | 0.164 | 0.273 | 0.367 | 0.445 | 0.609 | 0.625 | 0.680 |

Compare to hotpotqa's curve on the same checkpoint (`rag_all_golds@10=0.742`, `@200=0.953`) — the
retriever finds *a* correct doc for musique almost as reliably (`rag_any_gold` actually reaches 1.0
by k=50, vs. hotpotqa's 0.992 plateau), but finding *every* required supporting doc simultaneously
is much harder (`rag_all_golds@200=0.680` vs. 0.953), consistent with musique's multi-hop questions
needing more distinct supporting docs per question than hotpotqa's.

## Interpretation

Musique is meaningfully harder than hotpotqa for this checkpoint across every downstream metric
(judge accuracy, judge score, lexical grounding), even though the underlying corpus is a similar
size (10,000 vs 9,811 docs) and single-doc retrieval is if anything *easier* on musique
(`rag_any_gold`/`doc_hit_rate` both ≥ hotpotqa's). The gap traces most plausibly to
`rag_all_golds@200` (0.680 vs 0.953) and `mem_pos_weight_mass` (0.371 vs 0.445): musique's
multi-hop questions need more of the 200-slot auto-K bank's mass spread correctly across *several*
required docs, not just one, so even reliable single-doc retrieval leaves a meaningful fraction of
questions without full support in the bank the generator sees. This is consistent with prior
findings in this repo that musique is the harder of the two datasets for hybrid retrieval-augmented
generation (e.g.
[2026-07-19-musique-corpus-scaling-and-throughput-pareto.md](2026-07-19-musique-corpus-scaling-and-throughput-pareto.md),
[2026-07-22-msa-sweep-hybrid-vs-rag.md](2026-07-22-msa-sweep-hybrid-vs-rag.md)) — this is a new
data point on the same checkpoint family, not a new finding.

**Caveat:** n=128, same as every sibling eval in this family — enough to see the qualitative
hotpotqa-vs-musique gap clearly (it's large: ~0.14 accuracy, ~0.83 score points) but not precise to
the third decimal.

## Reproducibility

```bash
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
RUN_SCRIPT_PATH=scripts/embed/eval_msa_hybrid.sh \
RUN_ENV="DS=musique RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45 STEP=100000" \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

- **Commit:** `b4d9541` on `multihop-finetuning`, plus the (at-the-time still uncommitted)
  `models/conv_utils.py` dtype-cast fix from the 2026-08-13 hotpotqa eval of this same checkpoint
  — synced via the launcher's tree tar.
- **Checkpoint:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/qwen3_mem_embed/100000`
- **Result JSON:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/eval/step_100000/msa_musique_c10000_hybrid_autok.json`
- **wandb:** logged into the training run at `train_step=100000` —
  `johnzhang2366-columbia-university/memory-layers/qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-08-09-05-49-45`;
  standalone eval run `johnzhang2366-columbia-university/memory-layers-eval/5aruxn88`.
- **TPU:** `v6e-8` (`rohun-v6e-8-0`, `us-east1-d`, project `memorylayers`).
