# MSA sweep: hybrid@autoK vs RAG@10 across all 9 MSA benchmarks (overnight)

**Date:** 2026-07-22 · **Author:** rohunagrawal (with Claude, autonomous overnight run) ·
**Status: DONE (06:00)** — 7 of 8 MSA datasets + QASPER measured end-to-end (hybrid@autoK +
RAG@10 + per-sample judge accuracy, 1–5 score, retrieved docs, and the new
llm_judge_grounding_accuracy); triviaqa_10m and narrativeqa are blocked on a data defect
(below). msmarco_v1 from 2026-07-21/22 ([its page](2026-07-21-msmarco-hybrid-c10000.md)).

## Conclusion

**RAG@10 beats the hybrid on every usable dataset except QASPER, where they tie at the
floor (0.19 vs 0.18); the decisive variable is not corpus, bank size, or K — it is whether
the memory read stays grounded in its retrieved evidence.** The new grounding metric makes
the mechanism legible: at small auto-K (msmarco k=10, NQ k=5, dureader k=10) the hybrid's
answers are grounded 0.68–0.98 — at msmarco k=10 grounding is at RAG parity (0.984 vs
0.992) — but on every k=200-CAP arm it collapses to 0.39–0.48 while RAG holds 0.70–1.00.
Same checkpoint, same corpora, same retrieved candidates: reading 200 semantically-close
docs through memory slots detaches the generation from the evidence, exactly the
semantic-neighbor confusion the msmarco oracle isolated. Auto-K itself worked as designed
(k=5 → 200 across datasets, CAP recorded when 0.96 all-golds is unreachable), and the caps
are diagnostic: multi-hop + entity-dense corpora never reach 96% complete-evidence coverage
even at 200 docs. QASPER's tie is the flip side — when retrieval fails BOTH systems and no
parametric floor exists, the reads converge (grounding 0.89/0.95 there is copying, not
correctness). Caveats: single checkpoint (hard-neg base — musique's 0.148 reflects its known
pre-midtrain floor, not the architecture), n=128, judge noise ±0.04, grounding is
support-not-correctness (popqa RAG grounds 1.00 at 0.77 accuracy).

## Protocol (fixed across datasets — the msmarco_v1 methodology, unchanged)

- **Hybrid arm:** `gen_large_mem_msa_<ds>_hybrid` tasks — `inject_query_gold` corpus at
  `target_docs=10000` (full corpus when smaller), per-query **gathered banks**
  (`gather_bank: true`), B=8 data-parallel, `max_new_tokens=512`, `MEM_APPROX_TOPK=1`,
  n=128 queries, checkpoint **hard-neg base @100000**
  (`qa_hard_neg_think_…-02-32-09`). **auto-K**: `rag.top_k` = smallest of
  {5,10,25,50,100,150,200} with mean `rag_all_golds ≥ 0.96`; the largest candidate (200) is
  a hard cap when none clears (recorded in `metrics.rag_top_k`).
- **RAG arm:** k=10 docs in prompt, on the **eval-matched** corpus + queries built from the
  hybrid arm's own JSON (`build_msmarco_rag_corpus_from_eval.py` — dataset-agnostic despite
  the name); `rag_only.py` retrieve→generate→judge; per-sample record via
  `rag_to_viewer_json.py`.
- **Metrics on both arms:** `llm_judge_accuracy` + **`llm_judge_score` (1–5)** (Qwen3-4B
  judge, tp=4) + `lexical_grounding`; every sample carries `rag_docs` (retrieved texts, rank
  order) for `scripts/analysis/qa_compare_viewer.html`.
- **Orchestration:** `scripts/embed/msa_sweep_overnight.sh` on both slice workers
  (`tpu-v6e-slice-mig-{1wjb,1z9d}`) — per dataset: hybrid on both hosts, then RAG on 1z9d
  while 1wjb polls GCS; all artifacts idempotent, driver resumable.
- Corpus sizes (docs repo totals; ≥10k are trimmed to 10,000): 2wikimultihopqa 6,119 ·
  dureader 1,456 · hotpotqa 9,811 · musique 11,656 · narrativeqa 4,082 ·
  natural_questions 10,001 · popqa 8,676 · triviaqa_10m 12,589. QA rows: triviaqa_10m has
  exactly 128; narrativeqa 293.

## Results

| dataset | auto-K | hybrid acc | hybrid 1–5 | hybrid grounding | RAG@10 acc | RAG grounding | notes |
|---|---|---|---|---|---|---|---|
| msmarco_v1 (07-21) | k=10 manual | 0.578 | — | 0.984 | 0.758 | 0.992 | grounding from `_k10_docs`; oracle-mem-10 = 0.8125 |
| natural_questions | k=5 | 0.586 | 3.21 | 0.803 | 0.797 | 0.930 | single-hop template |
| triviaqa_10m | — | — | — | — | — | — | **UNUSABLE n=8**: QA pipeline drops 120/128 rows (giant packed pos_doc); needs data fix |
| dureader | k=10 | 0.414 | 2.67 | 0.681 | 0.828 | 0.922 | Chinese |
| hotpotqa | k=200 CAP | 0.320 | 2.36 | 0.481 | 0.734 | 0.883 | multi-hop |
| 2wikimultihopqa | k=200 CAP | 0.211 | 2.55 | 0.395 | 0.383 | 0.773 | hardest multi-hop; score>acc partial credit |
| popqa | k=200 CAP | 0.273 | 1.98 | 0.392 | 0.773 | 1.000 | rare entities: no parametric floor; largest acc gap (0.5) |
| musique | k=200 CAP | 0.148 | 1.77 | 0.467 | 0.344 | 0.703 | base ckpt's known pre-midtrain floor; not comparable to 07-20 midtrain rows |
| narrativeqa | — | — | — | — | — | — | **UNUSABLE**: 0/293 rows survive (novel-scale pos_doc); same defect as triviaqa |
| **qasper (new)** | k=200 CAP | 0.180 | 1.98 | 0.889 | 0.188 | 0.945 | **tie at the floor**; grounding here = technical copying, not correctness |

RAG re-judge values (converter pass) sit within judge noise of the pipeline numbers
throughout; per-sample records for every arm (`rag_docs`, judge outputs, 1–5 scores) are in
the `*_samples.json` / hybrid JSONs, and grounding sidecars are `*_grounding.json`
(top-5-by-overlap docs per judge call, one call per sample — support ≠ correctness).

## Repro

```bash
RUN_ENV="RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09 \
STEP=100000 CKPT_BUCKET=memory-layers-training PYTHONUNBUFFERED=1" \
TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers \
bash scripts/infrastructure/multi-tpu-box-run.sh \
  tpu-v6e-slice-mig-1wjb=scripts/embed/msa_sweep_overnight.sh \
  tpu-v6e-slice-mig-1z9d=scripts/embed/msa_sweep_overnight.sh
```

Results: `gs://memory-layers-training/qa_hard_neg_think_…-02-32-09/eval/step_100000/`
`msa_<ds>_c10000_hybrid_autok.json` · `msa_<ds>_c10000_rag_top10.json` ·
`msa_<ds>_c10000_rag_top10_samples.json`. Commit SHA: branch `msa-hybrid-sweep` (synced working tree).
TPU: v6e-8 flex slice (2× ct6e-standard-4t).

## Threats / notes-in-advance

- Auto-K on multi-hop sets (musique/2wiki/hotpotqa) may hit the 200 cap without clearing
  0.96 — the cap case is recorded and the coverage curve is in each JSON.
- dureader is Chinese; the Qwen3-4B judge handles zh, but scores are not comparable
  cross-language.
- The ~43% parametric-floor caveat from msmarco applies to every dataset here, both arms.
- `rag_recall@k` from the RAG pipeline remains do-not-quote (row-index id bug); use the
  hybrid pre-pass coverage curves instead.
- **llm_judge_grounding_accuracy** (ran 05:40–06:00): `scripts/misc/judge_grounding.py` +
  `scripts/embed/run_judge_grounding.sh` — one judge call per sample over the top-5
  rag_docs by content-word overlap with the answer (k=200 arms cannot fit all docs in the
  judge window; the cap is recorded per sidecar). Values in the Results table; sidecars
  `*_grounding.json` beside each source JSON in GCS.

## Follow-up (2026-07-22 ~12:45): memory-softmax temperature 0.5 on hotpotqa

Motivated by the grounding/mixing diagnosis above (per-token slot attention mixes
near-duplicate values; the answer-bearing component scales with gold weight share), one arm
re-ran with the memory layer's post-top-k softmax sharpened via the existing
`MEM_SOFTMAX_TEMP=0.5` env override (`models/memory.py::_mem_score_opts`) — same corpus,
same queries, same auto-K CAP k=200 banks (retrieval pre-pass is temp-independent),
generation still greedy.

| hotpotqa hybrid | temp 1.0, k=200 (baseline) | temp 0.5, k=200 | temp 0.25, k=100 |
|---|---|---|---|
| llm_judge_accuracy | 0.3203 | **0.3906** (+0.070) | 0.3906 (plateau) |
| llm_judge_score | 2.36 | **2.59** | 2.42 |
| mem_pos_weight_mass | 0.205 | 0.286 | 0.307 (still rising) |
| llm_judge_grounding_accuracy | 0.481 | 0.587 | 0.621 (still rising) |
| doc_hit_rate | 0.9922 | 0.9922 | 0.9922 (all unchanged) |
| all_golds@k (bank coverage) | ~0.95 | ~0.95 | 0.9375 (8 queries lose a gold) |

**Reading:** mass and grounding improve monotonically with sharpening — the read hops less
and less — but accuracy stops converting past temp 0.5 and the 1–5 score dips. The
grounding metric only requires support from ≥1 doc; hotpotqa answers need TWO. The residual
failure at temp 0.25 looks like grounded-but-incomplete: the near-argmax read locks onto one
supporting doc faithfully and under-serves the second hop (plus the k=100 coverage cost).
Sweet spot on this dataset ≈ temp 0.5 at full k; the two-knob cell (temp+k changed together)
leaves temp-0.25-alone unseparated. Sharpening cures hopping; multi-hop needs a controlled
multi-doc read, not maximal sharpness — which is the argument for commit-to-docs decoding or
trained-in binding over ever-lower temperatures.

Paired: 20 queries flip correct-ward vs 11 wrong-ward (net +9; suggestive rather than
conclusive on its own — McNemar p≈0.15 — but the mass movement is the predicted mechanism
signature and accuracy/score moved with it). Result JSON
`msa_hotpotqa_c10000_hybrid_autok_memtemp05.json` (+ `_grounding` sidecar). Obvious next
step: the small temp × MEM_TOP_K grid on one high-coverage arm and one CAP arm.

## Follow-up 2 (2026-07-22 ~16:45): cross-checkpoint probe — ground4layer @38k, hotpotqa k=100

`ground_s1_zeroinit_4layer-2026-07-04-09-42-55` @38000 (stage-1 grounding recipe: 4 memory
layers at [9,14,20,27], mem_top_k=128, zero-init `mem_o_proj`, main model frozen) on the
same 128 hotpotqa queries, fixed k=100, temp 1.0:
**acc 0.3516 · score 2.26 · mem_pos_weight_mass 0.157 · doc_hit_rate 0.992** (coverage
0.9375, identical to the hard-neg k=100 arm — encoder is checkpoint-independent).
Reading: edges the hard-neg default-temp arm (0.320 @ k=200) within noise, sits ~one noise
floor under both sharpened hard-neg arms (0.391); lowest per-slot gold mass of the four
hotpotqa cells — the 07-19 "more layers spread mass thinner without ranking architectures"
pattern again. Untested: sharpening this checkpoint (its low mass predicts a
MEM_SOFTMAX_TEMP=0.5 gain at least as large as hard-neg's). Result:
`ground_s1_zeroinit_4layer-…/eval/step_38000/msa_hotpotqa_c10000_hybrid_autok_k100.json`.
