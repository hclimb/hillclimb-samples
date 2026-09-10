# MuSiQue RAG→memory hybrid: retrieval-filtered bank at c512 / c2048

**Date:** 2026-07-20 · **Author:** rohunagrawal (with Claude) · **Status: done (2026-07-21)** — main, oracle, and g4-c512 arms measured; g4-c2048 and k=100 follow-up arms deliberately not run (see Conclusion).

## Conclusion (main arms)

**At c512 the hybrid works exactly as the dilution hypothesis predicts: 0.3984 vs 0.2810
full-bank (+0.117, ~6× the noise floor) — the memory layer's first accuracy TIE with RAG@5
(0.3906/0.3984 across its two runs) — and `mem_pos_weight_mass` jumps 0.34 → 0.72. At c2048
the trade flips: 0.2734 vs 0.3125 full-bank (−0.039), because the retrieval filter's
complete-gold coverage collapses (all_golds@50: 0.852 → 0.688) — the full bank always holds
every gold; the hybrid buys focus at the price of evidence completeness, and at c2048 the
price wins.** The oracle arms (perfect retrieval + 50-doc fill) will separate how much of the
c2048 loss is recall vs anything else.

| corpus | hybrid k=50 | oracle-50 (fill) | full-bank mem @750 | RAG@5 | all_golds@50 | pos_weight_mass (hybrid / oracle / full) |
|--------|------------|------------------|--------------------|-------|--------------|-------------------------------------------|
| 512    | 0.3984     | **0.4844**       | 0.2810             | 0.3906 | 0.852       | 0.720 / 0.927 / 0.34                      |
| 2048   | 0.2734     | **0.4531**       | 0.3125             | 0.3281 | 0.688       | n/a (aux OOM) / n/a / 0.16                |

**ground4layer @1500 hybrid k=50, c512 (2026-07-21): 0.3984** — identical to hard-neg's
hybrid score with `mem_pos_weight_mass` 0.51 (vs 0.72): the 4-layer checkpoint spreads mass
thinner and lands on the same accuracy, consistent with the 07-19 "more layers change nothing
consistently" result. With the same retrieval filter, the same `all_golds@50 = 0.852` ceiling
binds both checkpoints. (Its c2048 arm and the k=100 recall-fix arm were queued but not run — session closed at user request; both are one runner command away: SIZES=2048 on the g4 checkpoint, RAG_KS=100 SIZES=2048 on hard-neg @750.)

**Oracle update (2026-07-21, both arms judged in-run):** with perfect retrieval into a 50-doc
bank, the memory layer beats RAG@5 by **+0.094 (c512) and +0.125 (c2048)** — and the oracle
barely degrades with corpus size (0.484 → 0.453; the bank is 50 docs either way). The hybrid's
entire c2048 loss is retrieval recall, not the memory layer: fix the filter (larger k, better
retriever, or multi-hop-aware retrieval) and the c2048 headroom is ~0.18 over the shipped
hybrid number.

Supporting observations:
- **Any-gold recall flatters retrieval badly on multi-hop.** any_gold@5 = 0.98 while
  all_golds@5 = 0.42 (c512) / 0.24 (c2048) — the "recall@5 0.609" style numbers quoted for the
  RAG baseline never measured complete evidence.
- **Approx top-k is fully deterministic within-run**: `row_divergence_rate = 0.0` on all 8
  tiled greedy rows, both arms (supports the always-approx policy).
- **The verbosity defect still caps everything**: 37/128 c512 generations produced no answer
  (unclosed `</think>` at 1280 tokens) — 0.3984 was scored with ~29% forced misses.
- `lexical_grounding` roughly doubles vs historical runs (0.62 / 0.50 vs 0.24–0.31) — focused
  banks make the model copy documents much more.
- The 128 evaluated queries are the pipeline's first 128 *surviving* rows, not HF rows[:128]
  (gold union 288 vs 293): the tokenizing evals and the RAG baseline have always differed by a
  small tail of queries. Measured here via the content-based query join; footnote applies to
  all prior mem-vs-RAG rows too.

## Hypothesis & motivation

The [corpus-scaling Pareto experiment](2026-07-19-musique-corpus-scaling-and-throughput-pareto.md)
showed the memory layer's accuracy decay is **attention dilution, not retrieval failure**:
`doc_hit_rate` holds at 0.94+ while `mem_pos_weight_mass` collapses 4× — the gold slots are
found and then out-voted by distractor slots. Its follow-ups ranked "fix attention dilution,
not retrieval" as the #1 verdict-changing intervention.

This experiment intervenes causally on that mechanism: a **RAG@k pre-pass** (same vanilla
Qwen3-Embedding-0.6B + settings as the classic-RAG baseline) picks the top-k docs per query,
and the memory bank is **masked to those docs' slots** during generation. Same checkpoint,
same bank, same queries — only the candidate set changes. If dilution is the binding
constraint, pruning ~90% of distractor slots should raise `mem_pos_weight_mass` and accuracy;
if accuracy stays at the full-bank level, the bottleneck is downstream (the LM half —
verbosity/truncation, per the [2026-07-18 write-up](2026-07-18-musique-midtraining-vs-rag.md)).

Reference points (same protocol, n=128, judge Qwen3-4B): c512 RAG@5 **0.3906** / mem hard-neg
0.2810; c2048 (the Pareto-plot point) RAG@5 **0.3281** / mem hard-neg **0.3125**. Noise floor
~0.008–0.02 — differences under ~0.03 are ties.

Not a throughput experiment: the hybrid's decode is transformer-bound at small banks
(~0.505 q/s ceiling vs RAG's flat 0.579 at 256-tok docs); the serving payoff, if any, is on
long-document tasks.

## Setup

- **Evaluator:** `generation_large_mem_rag_hybrid`
  ([implementation note](../implementations/2026-07-20-rag-hybrid-evaluator.md)) — per-query
  `mem_mask` = token-validity AND top-k docs' slots; masking happens before top-k selection so
  it is equivalent to shrinking the bank. Approx top-k ON (`MEM_APPROX_TOPK=1`, recall 0.99 —
  the standing [retrieval-modes](../architecture/retrieval-modes.md) policy; the Pareto
  reference rows were also measured with approx on), fixed 512-token prompt pad,
  `rag.top_k=50`.
- **Checkpoint:** hard-neg midtrain @750 —
  `gs://memory-layers-training/musique_sft_midtrain_topk64_seq1024_chunks20_bs32-2026-07-18-17-35-52/qwen3_mem_embed/750`
  (EUROPE-WEST4 copy; original in `-usc1`). 1 memory layer, `mem_top_k=64`
  (checkpoint-authoritative).
- **Corpora:** c512 and c2048 via `inject_query_gold` (identical haystacks to the Pareto
  accuracy sweeps), same first 128 queries, `max_new_tokens=1280`.
- **TPU:** v6e-8 flex slice, 2 × `ct6e-standard-4t` (`tpu-v6e-slice-mig`, europe-west4-a) —
  first eval run on a multi-host slice (runbook §2.3 lists the slice-specific behavior).

## Repro

```bash
RUN_ENV="RUN_DIR=musique_sft_midtrain_topk64_seq1024_chunks20_bs32-2026-07-18-17-35-52 \
STEP=750 CKPT_BUCKET=memory-layers-training PYTHONUNBUFFERED=1" \
TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers \
bash scripts/infrastructure/multi-tpu-box-run.sh \
  tpu-v6e-slice-mig-1wjb=scripts/embed/eval_musique_hybrid.sh \
  tpu-v6e-slice-mig-1z9d=scripts/embed/eval_musique_hybrid.sh
# defaults: SIZES=512_2048 RAG_KS=50 NUM_SAMPLES=128 MAX_NEW_TOKENS=1280
# oracle ceiling arm: prepend ORACLE=1 [ORACLE_FILL_TO=50] to RUN_ENV
```

Commit SHA: pending PR (branch not yet cut at launch time; smoke + arms ran from the synced
working tree). Result JSONs land at
`gs://memory-layers-training/<run-dir>/eval/step_750/musique_c{512,2048}_hybrid_k50.json`
and in the training run's wandb
([musique_sft_midtrain…-17-35-52](https://wandb.ai/memory-layers/memory-layers/runs/musique_sft_midtrain_topk64_seq1024_chun-2026-07-18-17-35-52)).

## Results

Main-arm results and reading: see the Conclusion table above. Full coverage curves and
per-sample data are in the result JSONs (below); judge annotations were applied by a
standalone pass (`judge_one.py`) after a manifest-handling bug silently skipped the in-run
judge — fixed in `evals/shared.py` in this same change.

**Oracle c512 (perfect retrieval, fill to 50): 0.4844** — beats every system measured on this
corpus (hybrid 0.3984, RAG@5 0.3906/0.3984, full-bank 0.2810), with `mem_pos_weight_mass`
0.927 and `lexical_grounding` 0.77. The c512 chain is monotone in weight mass
(0.34 → 0.72 → 0.93 for full → RAG@50 → oracle), and the 0.398 → 0.484 gap says **retrieval
completeness is now the binding constraint** on the hybrid (~matches the 15% of queries whose
gold set is incomplete at k=50). Oracle c2048 running — it bounds how much of the c2048
regression is recall: if it clears 0.3125, the lever is a better/larger-k filter
(all_golds@100 = 0.773 suggests k=100 recovers most of it).

## Threats to validity

- n=128, noise floor ~0.02: read <0.03 differences as ties.
- Mild OOD: midtraining banks were ~256–320 docs; a 50-doc mask is smaller, and top-k 64 over
  ~12.8k candidate slots is a ~10× higher selection fraction than training saw. The oracle
  arm partially controls for this.
- The verbosity/truncation defect from the 2026-07-18 write-up caps what any bank
  intervention can show (~40% of generations previously failed to close `</think>` at 1280).
