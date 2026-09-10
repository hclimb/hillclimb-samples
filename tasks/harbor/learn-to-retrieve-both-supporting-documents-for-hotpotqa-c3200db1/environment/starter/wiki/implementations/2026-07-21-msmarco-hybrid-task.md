# MS-MARCO c10000 hybrid eval task (config + runner; evaluator unchanged)

**Date:** 2026-07-21 · **Author:** rohunagrawal (with Claude) · **Status:** done ·
**Branch:** `msa-hybrid-sweep`

**What changed:** the RAG→memory hybrid eval
([2026-07-20 evaluator note](2026-07-20-rag-hybrid-evaluator.md)) can now run on MS-MARCO —
a new task config `configs/eval/tasks/gen_large_mem_msmarco_hybrid.yaml` and runner
`scripts/embed/eval_msmarco_hybrid.sh`, defaulting to a **10,000-doc corpus (2.56M slots)**
and **`max_new_tokens: 512`**. `evals/gen_large_mem_rag_hybrid.py` needed **zero changes**:
its QA loading is config-driven (`rag.query_dataset` / `query_column`, standard
`pos_doc_ids`), and the c10000 bank sits under the 4M-slot replicated-bank guard.

## Motivation & context

The MuSiQue hybrid result was recall-bound at c2048: `all_golds@50` collapsed 0.852 → 0.688
on multi-hop gold sets, and the oracle arms showed the *memory layer itself* beats RAG@5 once
retrieval is perfect ([2026-07-20 experiment](../experiments/2026-07-20-musique-rag-hybrid.md)).
MS-MARCO is **single-hop (~1 gold/query)**, so the retrieval-completeness failure mode is
structurally absent — it isolates the dilution-rescue effect at a corpus size (10k docs) where
the full bank decays badly on MuSiQue. Checkpoint: the **base hard-neg think** run
(`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09` @100000 — the exact
checkpoint the MuSiQue midtrain warm-started from), the only candidate with MS-MARCO in its
training mix (msmarco-triplets is 1 of 5 hard-neg SFT sources).

## Decisions & tradeoffs

- **`max_new_tokens: 512`, not the MuSiQue 1280.** `inference.py::_generate_tokens` is a
  `fori_loop` with **no EOS early-stop**, so every query pays the full budget; MS-MARCO
  answers are short. Cuts decode cost ~2.5× at the price of comparability with the MuSiQue
  rows (which needed 1280 for think-chains).
- **`target_docs: 10000` default** (source corpus `msa-msmarco-v1-docs-with-ids` has 75,574
  docs; corpus still built in-run by the evaluator's `inject_query_gold` scan). Bank =
  10,000 × 256 = 2.56M slots; with `mem_k_dim = mem_v_dim = 1024` that is ~10.5 GB bf16
  replicated per chip + ~8 GB model on 31 GB v6e HBM — measured fine. (The guard comment's
  "~2048-ish" dim is stale; real dim is 1024.)
- **`NAME_SUFFIX` runner knob** (new vs the MuSiQue runner): appended to the GCS result name
  so smokes (`_smoke`) don't trip the idempotency skip-check for the real arm.
- **Checkpoint staged region-local:** the @100000 checkpoint existed only in `-usc1`; copied
  (19.6 GiB + `.hydra/`) to `gs://memory-layers-training` under the same run-dir path, same
  pattern as the midtrain @750 copy.
- Everything else inherited deliberately: `MEM_APPROX_TOPK=1` exported in the runner (standing
  [retrieval-modes](../architecture/retrieval-modes.md) policy), `batch_size: 1`,
  `prompt_pad_len: 512`, k=50, Qwen3-4B judge, multi-host slice conventions.

## How it was built & integrated

Both files are line-for-line mirrors of their MuSiQue counterparts
(`gen_large_mem_musique_hybrid.yaml`, `eval_musique_hybrid.sh`) with the dataset sources
swapped to `msa_msmarco_v1_qa` / `msa_msmarco_v1_docs`,
`rag.query_dataset: ${HF_USERNAME}/msa-msmarco-v1-qa-with-ids` (verified: `question` +
`pos_doc_ids` columns, 9,345 rows), and the defaults above. Eval key `msmarco_hybrid`;
results land at `gs://<bucket>/<run-dir>/eval/step_<N>/msmarco_c<size>_hybrid_k<k>.json`.
Oracle arms via `ORACLE=1 [ORACLE_FILL_TO=50]`, unchanged.

## Reference pages updated

None needed: [evaluator-types](../evaluation/evaluator-types.md) describes the evaluator
(unchanged), and task configs are enumerated by directory, not by page. This note + the
experiment write-up carry the new-task record.

## Test record

End-to-end smoke on the 2×`ct6e-standard-4t` v6e slice (`tpu-v6e-slice-mig-{1wjb,1z9d}`),
n=2, c10000, k=50, both workers via `multi-tpu-box-run.sh`; both hosts `__RUN_EXIT__=0`:

```
2 queries matched against ragrawal36/msa-msmarco-v1-qa-with-ids (9345 rows), 2 gold doc ids
corpus (scanned 75574): 2 gold + 9998 distractor chunks = 10000 (target=10000)
retrieval pre-pass done in 86.5s   rag_any_gold@5=1.0  rag_all_golds@5=1.0  (… @100=1.0)
bank replicated: mem_k (2560000, 1024), mem_v (2560000, 1024), eff_doc_len=256, rows/query=8
WARNING: aux telemetry forward OOM'd — doc_hit_rate/mem_pos_weight_mass disabled  [expected, as at c2048]
metrics: llm_judge_accuracy 1.0 · row_divergence_rate 0.0 · mean_active_bank_slots 3873
         lexical_grounding 0.452 · bank_slots 2560000 · corpus_docs 10000
```

Sanity readings: `mean_active_bank_slots` 3,873 (≤ 50×256 = 12,800 — the validity-AND is
correct; MS-MARCO passages average ~77 valid tokens), both generations well-formed and judged
correct. Timing: query 1 = 69.5 s (compile), steady state **~26 s/query** (512 tok ≈ 51 ms/tok
at the 2.56M bank) — ~1¼ h projected for the n=128 arm.

## Follow-ups & risks

- `doc_hit_rate` / `mem_pos_weight_mass` are unavailable at this bank size (aux forward OOMs,
  self-disables) — the dilution telemetry story rests on the accuracy deltas + coverage curve.
- The smoke's coverage=1.0 is n=2; the n=128 arm's `rag_all_golds@50` is the number that
  tests the single-hop-recall premise.
- `judge`/`gen` sample-matching caveat from the MuSiQue write-up (first 128 *surviving* rows)
  applies here identically.
