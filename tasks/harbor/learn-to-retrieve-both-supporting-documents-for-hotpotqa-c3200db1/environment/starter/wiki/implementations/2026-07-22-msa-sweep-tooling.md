# MSA sweep tooling: auto-K, per-dataset task configs, two-host overnight driver

**Date:** 2026-07-22 · **Author:** rohunagrawal (with Claude) · **Status:** done ·
**Branch:** `msa-hybrid-sweep`

**What changed**, to run the [overnight MSA sweep](../experiments/2026-07-22-msa-sweep-hybrid-vs-rag.md):

1. **auto-K in the hybrid evaluator** (`evals/gen_large_mem_rag_hybrid.py`): with
   `rag.auto_k_threshold` set (base config default `null` = off), `rag.top_k` is replaced
   after the retrieval pre-pass by the smallest `rag.auto_k_candidates` entry
   ([5,10,25,50,100,150,200]) whose mean `rag_all_golds` clears the threshold; the largest
   candidate is a hard cap when none does (choice logged + recorded as `metrics.rag_top_k`).
   Free: the pre-pass already ranks the full corpus; the full rankings are just cut after
   the choice instead of before.
2. **8 generated task configs** `gen_large_mem_msa_<ds>_hybrid.yaml` — msmarco-hybrid
   protocol (target 10000, gather_bank, B=8, max_new 512) with per-dataset sources and
   `auto_k_threshold: 0.96`; metrics now include **`llm_judge_score` (1–5)** — the metric
   existed in `evals/metrics/llm_judge.py` and was registered, just never wired into a task.
3. **Runners** `scripts/embed/eval_msa_hybrid.sh` / `eval_msa_rag.sh` — DS-parameterized
   versions of the msmarco pair; the RAG runner folds in the corpus build (from the hybrid
   arm's own JSON) and the viewer-JSON conversion (now also annotating `llm_judge_score`
   per sample — `scripts/misc/rag_to_viewer_json.py`).
4. **Driver** `scripts/embed/msa_sweep_overnight.sh`, launched on BOTH slice workers: per
   dataset, hybrid runs on both hosts (JAX self-syncs at TPU init), then RAG runs on
   `RAG_HOST` (1z9d, the JAX rank-0/results host) while the peer polls GCS (cap 90 min) so
   both hosts enter the next hybrid together. Every artifact has a GCS skip-check → the
   driver is kill-and-relaunch resumable at any phase.

## Reference pages updated
None — evaluator-types' hybrid section already describes gather/batching; auto-K is a task
knob documented in the base config; the sweep protocol lives in the experiment page.

## Tests
`py_compile`/`bash -n` on all files; end-to-end validation via the hotpotqa n=8 smoke
(auto-K choice print, both judge metrics, gather banks on a multi-hop dataset) — output
recorded in the experiment page as the sweep's smoke record.

## Follow-ups & risks
- The two-host phase-B sync relies on GCS polling with a 90-min cap; a wedged RAG phase
  idles the peer for that long before the sweep continues (accepted for overnight).
- `build_msmarco_rag_corpus_from_eval.py` is dataset-agnostic but keeps its msmarco name.
