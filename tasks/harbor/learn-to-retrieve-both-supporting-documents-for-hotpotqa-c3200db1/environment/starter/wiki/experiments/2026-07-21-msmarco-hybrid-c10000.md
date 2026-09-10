# MS-MARCO RAG→memory hybrid at c10000: single-hop removes the recall ceiling

**Date:** 2026-07-21 · **Author:** rohunagrawal (with Claude) · **Status: arm 1 done** —
hybrid k=50 measured; RAG@5 / full-bank / oracle baselines at this corpus **deliberately not
run yet** (see Follow-ups).

## Conclusion

**The hybrid scores `llm_judge_accuracy` 0.609 (78/128) on MS-MARCO at a 10,000-doc corpus
(2.56M-slot bank) with `rag_all_golds@50 = 1.000`** — on single-hop, the retrieval filter is
complete for every query, so the recall collapse that sank the MuSiQue hybrid at c2048
(all_golds@50 0.688) is structurally absent, exactly as hypothesized. 0.609 is the highest
memory-layer accuracy recorded in this repo at any corpus size (MuSiQue hybrid peaked at
0.398 @ c512; full-bank at a comparable corpus, c11656, scored 0.234 — different dataset, so
indicative, not a comparison). **No same-protocol baseline at c10000 exists yet**, so
whether 0.609 beats RAG@5 or the full bank on *this* task is an open cell, not a claim.
Noise floor at n=128, p≈0.6 is ~0.043 — read differences under ~0.09 as ties.

| arm (c10000, n=128) | llm_judge_accuracy | status |
|---|---|---|
| **hybrid k=50 (this run)** | **0.609** | done |
| hybrid k=10 | 0.5625 | done (2026-07-21) — see paired analysis below |
| hybrid k=5 (gather-bank) | 0.5469 | done (2026-07-21) — first arm with full telemetry; see below |
| **oracle fill-10 (gather-bank)** | **0.8125** | done (2026-07-21) — **+0.25 over hybrid k=10 at the SAME bank size; only ~4 points of it is recall. See below — this changes the headline reading.** |
| **RAG@10 (docs in prompt)** | **0.7578** | done (2026-07-22) — same encoder, same eval-matched corpus, same 128 queries; see below |
| full-bank memory | — | not run |

**k=10 arm (same queries, same corpus): 0.5625 — a statistical tie with k=50, with the
direction fully consistent with pure recall loss.** Paired per-query cross-tab (queries
aligned by prompt): both correct 61, k50-only 17, k10-only 11, neither 39 — 28 discordant
pairs split 17/11 (McNemar p≈0.26, not significant). Of the 5 queries whose gold fell
outside the top-10 (`all_golds@10` 0.961), 3 were correct at k50 and only 1 at k10 — the
coverage loss accounts for ~2 of the 6 net flips; the rest is bank-composition/generation
noise. **No evidence the 5× sharper mask helps beyond k=50** — dilution appears already
fully rescued at 50 docs on this task — and no evidence it hurts beyond its recall cost.
Note the 39 both-wrong queries all had complete gold coverage at k=50: the residual failure
mode is downstream of retrieval. (`msmarco_c10000_hybrid_k10.json`, same GCS dir;
lexical_grounding 0.444, 5 empty answers.)

**k=5 arm (gather-bank mode, [implementation note](../implementations/2026-07-21-hybrid-eval-gather-bank.md)):
0.5469, and the telemetry that was OOM-lost at this corpus size is back.** The k-curve
0.547 / 0.5625 / 0.609 (k=5/10/50) tracks complete-gold coverage 0.766 / 0.961 / 1.000
monotonically. Three findings the telemetry adds:

- **Within-bank retrieval is perfect:** `doc_hit_rate` 0.8203 = `rag_any_gold@5` exactly —
  every query whose bank contains a gold attends to it in the top-64; the entire deficit is
  the 23 gold-less banks. Retrieval inside the memory layer is not a failure mode at any k
  measured.
- **~43% parametric answering on gold-less banks:** 10 of the 23 queries with NO gold in
  the bank were still judged correct — Qwen3-4B answers a chunk of MS-MARCO from its
  weights. This floor props up every arm (and will equally prop up a RAG baseline); absolute
  numbers on this task overstate what retrieval contributes.
- **First per-sample dilution×correctness read at c10000:** `mem_pos_weight_mass` mean
  0.314; **0.492 on judged-correct vs 0.238 on judged-wrong** — correct answers put ~2× the
  attention mass on gold slots (within-model, correlational).

Paired vs k=50: both 62, k50-only 16, k5-only 8, neither 42 — net −8, of which the 30
uncovered queries contribute −4 (13 correct at k5 vs 17 at k50); the rest is the usual
bank-composition jitter. Wall clock: gather mode ran the whole arm in ~13 min (generation
~1.6 min, ~0.54 s/query — the full-bank scan removed; cumulative 55 → 7.2 → 1.6 min across
the three eval-efficiency steps). (`msmarco_c10000_hybrid_k5.json`; lexical_grounding 0.468.)

**RAG@10 baseline (2026-07-22): 0.7578 — RAG beats every retrieved hybrid arm; only the
oracle bank beats RAG.** Measured on an **eval-matched** corpus and query set built from the
hybrid run's own record (`datagen/msmarco/build_msmarco_rag_corpus_from_eval.py` →
`ragrawal36/msmarco-c10000-rag-corpus-evalmatched` + `…-eval-queries`;
runner `scripts/embed/eval_msmarco_rag.sh`) — the MuSiQue rows' "first 128 *surviving* vs
HF rows[:128]" mismatch does not apply here (verified: HF row 0 is pipeline-dropped, so the
old rows[:N] recipe *would* have shifted the whole query set). The three-way at a matched
10-doc candidate set: **hybrid k=10 0.5625/0.578 < RAG@10 0.758 < oracle-mem-10 0.8125.**
Reading the same retrieved docs in-context beats reading them through memory slots by
~0.18 — the in-context LM handles semantic-neighbor distractors nearly as well as the
memory layer handles benign ones, while the memory read loses ~0.2 to them. The
architecture's ceiling is not the problem (the oracle bank *beats* in-context RAG);
distractor-robustness of the memory read is. Caveats: the pipeline's own
`rag_recall@k` values (0.164@1 / 0.477@5) are implausible against the same
encoder+corpus's directly-computed coverage (`any_gold@5` 0.82) — the retrieval-metric id
mapping is suspect (echoes the historical `recall@5 = 0.006` artifact) and those numbers
should not be quoted; `rag_accuracy` is judged on generated text and unaffected. The ~43%
parametric floor props up RAG exactly as it props up the memory arms.
Per-sample record: `msmarco_c10000_rag_top10_samples.json` (same GCS dir) — hybrid-schema
samples with `rag_docs` + per-sample judge verdicts (`scripts/misc/rag_to_viewer_json.py`
re-runs the real `llm_judge_accuracy`; re-judge mean 0.7500 vs the pipeline's 0.7578 — one
flip, within the judge's known repeat noise). Two more honest details it surfaced:
RAG's in-prompt gold coverage was **119/128 (0.930)** vs the hybrid bank's 123/128 (0.961) —
the corpus repo carries original doc text while the bank embeds the 256-token round-trip, so
the rankings differ at the margin — meaning RAG's +0.18 was earned with slightly *worse*
coverage (the verdict is conservative). Drop the samples JSON into
`scripts/analysis/qa_compare_viewer.html` beside `…_k10_docs.json` for side-by-side answers
and retrieved docs.

**Oracle fill-10 (2026-07-21, gather-bank): 0.8125 — and it reframes the k-sweep.** Same
10-doc bank size as hybrid k=10 (0.5625), coverage forced to 1.0, distractors drawn in
corpus order instead of the retriever's top-10. Paired per-query: both 67, oracle-only 37,
k10-only 5, neither 19 — and of k=10's 5 uncovered queries the oracle fixes 4, so **recall
explains only ~4 of the 25-point gap. The other ~21 points are distractor company**: the
retrieval filter hands the memory layer the query's nearest semantic neighbors — precisely
the hardest confounders — while the oracle's corpus-order fill is benign. The telemetry
agrees emphatically: `mem_pos_weight_mass` **0.992** with benign distractors (identical on
correct and wrong answers — attention is never the problem) vs **0.19–0.31** measured on the
retrieved arms; `lexical_grounding` jumps to 0.63 (vs ~0.45 retrieved). So the earlier
"dilution is already rescued at k=50" reading was too kind: at matched bank size the
retrieved arms lose ~0.2 accuracy to **semantic-neighbor confusion**, a failure the k-sweep
could not see because every k shares it. The memory layer at this checkpoint is *capable* of
0.81 on this task; the sharp question is now distractor-robustness (hard-negative training
at eval-like ratios, or a diversity-aware filter à la MMR), not bank size. The 19
never-correct queries (with mass 0.992!) are the residual non-retrieval failure floor.

Supporting observations:

- **Coverage curve:** all_golds@5 = 0.766, @10 = 0.961, @25 = 1.0 (~1.1 golds/query; 143
  gold docs over 128 queries). A future RAG@5 baseline will operate at 77% complete-gold
  coverage — worth remembering when reading that comparison.
- **The verbosity defect barely bites here:** 3/128 empty answers (vs 37/128 on MuSiQue at
  1280 tokens). Median answer 39 words. `max_new_tokens: 512` is comfortable for MS-MARCO.
- `lexical_grounding` 0.456; `mean_active_bank_slots` 4,384 of 2.56M (top-50 docs' valid
  tokens; MS-MARCO passages average ~77 valid tokens of the 256-token slot budget).
- `doc_hit_rate` / `mem_pos_weight_mass` unavailable: the aux telemetry forward OOMs at
  this bank size and self-disables (known behavior since c2048).
- **First run of the data-parallel eval mode** ([implementation
  note](../implementations/2026-07-21-hybrid-eval-data-parallel-batch.md)): B=8 distinct
  queries per forward via per-example `[B, M]` mem_mask — 128 generations in **~7.2 min**
  (~3.3 s/query steady state) vs ~55 min projected for the legacy tiled mode. Whole arm
  ~17 min end to end on the v6e-8 flex slice.

## Hypothesis & motivation

The [MuSiQue hybrid](2026-07-20-musique-rag-hybrid.md) showed masking the bank to the RAG
top-50 rescues attention dilution (+0.117 at c512) but is **recall-bound on multi-hop**: at
c2048 the filter's complete-gold coverage collapsed and the trade flipped negative, while
the oracle arms proved the memory layer itself beats RAG@5 under perfect retrieval.
MS-MARCO is single-hop (~1.1 golds/query) — retrieval completeness shouldn't degrade, so a
10k-doc corpus isolates the dilution-rescue effect at a scale where full banks decay badly.
Checkpoint: **base hard-neg think @100000** — the only candidate with MS-MARCO in its
training mix (msmarco-triplets is 1 of 5 hard-neg SFT sources), and the checkpoint the
MuSiQue midtrain warm-started from.

## Setup

- **Evaluator:** `generation_large_mem_rag_hybrid`, task
  `gen_large_mem_msmarco_hybrid` ([implementation note](../implementations/2026-07-21-msmarco-hybrid-task.md)):
  `msa_msmarco_v1` docs/QA (75,574-doc source corpus; `inject_query_gold` → 143 gold +
  9,857 distractors = 10,000), `rag.top_k=50`, `max_new_tokens=512` (no EOS early-stop —
  every query pays the full budget), `prompt_pad_len=512`, `MEM_APPROX_TOPK=1` (standing
  policy), **`dataset.batch_size=8`** (data-parallel mode).
- **Checkpoint:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09/qwen3_mem_embed/100000`
  (EUROPE-WEST4 copy made for this run; original in `-usc1`). 1 memory layer, top-k 64.
- **TPU:** v6e-8 flex slice, 2 × `ct6e-standard-4t` (`tpu-v6e-slice-mig`, europe-west4-a).
- **Judge:** Qwen3-4B, in-run.

## Repro

```bash
RUN_ENV="RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09 \
STEP=100000 CKPT_BUCKET=memory-layers-training PYTHONUNBUFFERED=1" \
TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers \
bash scripts/infrastructure/multi-tpu-box-run.sh \
  tpu-v6e-slice-mig-1wjb=scripts/embed/eval_msmarco_hybrid.sh \
  tpu-v6e-slice-mig-1z9d=scripts/embed/eval_msmarco_hybrid.sh
# defaults: SIZES=10000 RAG_KS=50 NUM_SAMPLES=128 MAX_NEW_TOKENS=512, batch_size 8 from the task config
# oracle arm: prepend ORACLE=1 ORACLE_FILL_TO=50 to RUN_ENV
# full-bank baseline: eval.py with gen_large_mem_msmarco_c512 task + target_docs=10000
```

Commit SHA: branch `msa-hybrid-sweep` (arms ran from the synced working tree).
Result JSON:
`gs://memory-layers-training/qa_hard_neg_think_…-02-32-09/eval/step_100000/msmarco_c10000_hybrid_k50.json`,
logged to the training run's wandb (`eval/msmarco_c10000_hybrid_k50/llm_judge_accuracy`).
Smokes (n=2 tiled, n=8 batched) at `…_smoke.json` / `…_smokeb8.json` in the same directory.
`…_k10_docs.json` is a k=10 re-run whose samples additionally carry `rag_docs` (the 10
retrieved docs' text per query) for inspection in
`scripts/analysis/qa_compare_viewer.html`; it also backfills the k=10 telemetry
(gather-bank mode).

## Threats to validity

- **No same-protocol baseline yet** — the headline is an absolute number; every comparison
  above is cross-task or cross-protocol and labeled as such.
- n=128, noise ~0.043 at p≈0.6.
- Dilution telemetry (`mem_pos_weight_mass`) unavailable at this bank size (aux OOM), so
  the mechanism story rests on the accuracy deltas once baselines exist.
- The B=1-vs-B=8 end-to-end equality check was skipped at user request; kernel-level
  equivalence is unit-tested (`tests/test_rag_hybrid_mask.py::per_example_mask_batched_equals_single`)
  and B=8 reproduced the earlier B=1 smoke's answers on the 2 overlapping queries.

## Follow-ups (each ≤ ~20 min of box time at B=8)

1. **Full-bank memory @ c10000** — the dilution baseline; one `eval.py` command.
2. ~~RAG @ c10000~~ **done at k=10** (0.7578; eval-matched corpus/queries — see above).
3. **Oracle fill-50** — `ORACLE=1 ORACLE_FILL_TO=50`; separates "semantic-neighbor
   distractors" from "corpus-order distractors" now that coverage is already 1.0.
4. ~~k-sweep~~ **k=10 done** (0.5625 — tie with k=50 minus recall; see above). Remaining
   k values look uninformative on this task: coverage saturates by k=25 and sharpening
   below 10 only loses recall.
5. **Union-bank gather** (efficiency, not accuracy): gather the batch's top-k docs into a
   small shared bank + per-example mask over it — removes the full-bank scan (~2–2.5×
   decode), lifts the 4M-slot corpus ceiling (bank stays on host), and restores the aux
   telemetry. Design discussed 2026-07-21; not yet built.
