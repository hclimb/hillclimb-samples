# Evaluator Types

`evals/__init__.py :: get_evaluator(eval_cfg, key)` dispatches on `eval.type`. All subclass
`evals/base.py::Evaluator` (`evaluate(model, dataset, step, **kwargs) → inference_metrics`, plus
`_get_output_path` and `_detect_model_type`). Raw samples are written to the output file; scoring
happens later ([metrics.md](metrics.md)).

| `type` | Class | What it does |
|--------|-------|--------------|
| `nll` | `NLLEvaluator` | Forward pass → per-token cross-entropy → mean loss + perplexity. |
| `generation` | `GenerationEvaluator` | Autoregressive generate from **text** prompts, compare to references. |
| `generation_embed` | `GenerationEmbedEvaluator` | **Embed docs per batch** → inject `mem_k/mem_v/mem_mask` into `main_model` weights → JIT decode (no re-embed). Needs `provide_docs`. |
| `generation_large_mem` | `GenLargeMemEvaluator` | **Two-phase static memory** (below). Needs a `doc_dataset`. |
| `generation_large_mem_msa` | `GenLargeMemMSAEvaluator` | Large-mem variant for the MSA model. |
| `swap_logit_delta` | `SwapLogitDeltaEvaluator` | Retrieval-quality diagnostic: two teacher-forced forwards per batch, one with original docs, one with each query's positive doc chunks swapped for a hard-neg from the same row. Reports `mean(logp_A − logp_C)` over answer tokens. Needs `provide_docs` + a qa dataset carrying `neg_doc` (see below). |
| `ruler` | `RULEREvaluator` | RULER long-context suite ([ruler.md](ruler.md)). |
| `generation_large_mem_rag_hybrid` | `GenLargeMemRagHybridEvaluator` | RAG→memory hybrid: dense-retrieve top-k docs per query, restrict the bank to their slots (below). |
| `generation_base` | `gen_base_model.run_generation` | Baseline generation with **no memory**, via vLLM — **deferred to the parent process** (skips JAX). |

## `generation_large_mem` — two phases
1. **Embed all docs** from `doc_dataset` with only the embed model, gathered to CPU numpy; embed
   weights freed.
2. **Shard `mem_k`/`mem_v` across all devices** on the `data` axis, load the main model, generate
   with **chunked sharded retrieval** (`shard_axis='data'`). See
   [../architecture/sharded-retrieval.md](../architecture/sharded-retrieval.md).
- `inject_query_gold` guarantees each query's gold docs are in the corpus (needs `pos_doc_ids`),
  filling the rest with distractors up to `max_docs`.

## `swap_logit_delta` — how it's constructed
- Per batch, the memory bank is the concatenation of every row's chunks (`B × num_chunks_per_doc` slots — same as nll eval).
- **Pass A:** standard teacher-forced forward with the batch's own docs → per-token `log P(gt_answer_token | prompt, docs)`.
- **Pass C:** for each row `i`, replace the positive doc chunks (where `pos_doc_mask[i]==1`) with the row's own first non-positive chunk (from `pack_docs`'s neg pool), then re-run the forward → per-token `log P(gt | prompt, docs_swapped)`. All `B` swaps happen in one bank rebuild → one extra forward per batch.
- **Metric:** per-example `mean(logp_A − logp_C)` over loss-masked answer positions; batch mean → eval mean.
- **NaN safety:** the per-sample reduction masks non-finite `logp_gt` at token level (`isfinite & loss_mask`) and uses the valid-count as the denominator — a rare NaN at a padded position (e.g. all-endoftext prompt tail) can't poison a whole example's or batch's mean.

### Bias / speed tradeoff — why *batched* swap
The swap is done in a single tensor mutation (one bank rebuild → one Pass-C forward per batch) rather than per-query (which would need `B` separate forwards). Consequence: in Pass C, query `i`'s pool also loses other queries' positives, not only its own. In practice the bias is tiny — other queries' positives are on unrelated topics, so top-K retrieval for query `i` already routes almost no mass to them — and the metric shows this: on `combined_hard_neg_sft4b` at the SFT-4B checkpoint, the number is stable at `~+0.45 nats` across pool sizes 64, 256, 512 (batch × num_chunks_per_doc). A fully surgical per-query swap would cost `~B×` more forwards for essentially no metric change at these pool sizes. Pool sizes ≥ 1024 OOM on v6e-8 at the score-matrix step (`(B, H, T, M)` bf16 ≈ 68 GB at pool=1024) — pending more HBM (bigger TPU) before we can check whether the metric decays at that scale.

### Dataset requirement
The swap picks the row's first slot where `pos_doc_mask == 0`. Datasets with real `neg_doc` fields (e.g. `combined_hard_neg_sft4b` — `min_neg_docs=2`) → that slot holds a real hard-negative chunk, so the swap is a proper `pos → hard-neg` **replacement**. Datasets without a `neg_doc` field (e.g. plain `msa_hotpotqa_qa`) → the first non-pos slot is padding (`docs_mask=0`), so the swap effectively **masks the positive out of the bank** — still a valid signal, but a different semantics; prefer hard-neg datasets.
## `generation_large_mem_rag_hybrid` — retrieval-filtered bank

Same corpus construction as `generation_large_mem` (`inject_query_gold`), then per query:
dense-retrieve top-`rag.top_k` docs with the vanilla Qwen3-Embedding tower (parity with the
classic-RAG baseline) and rewrite `main_model.mem_mask` to
(token-validity AND top-k docs' slots) — mem_mask is applied *before* top-k selection, so this
is equivalent to shrinking the bank, with no index remapping and no recompile. Prompts are
left-padded to a fixed `prompt_pad_len` (one compile per run), bank **replicated** (guarded to
≤4M slots). Two batching modes via `dataset.batch_size` (per-example masks since 2026-07-21):
**1** = legacy tiled mode — the query is tiled to the data-axis size with a shared `[M]` mask,
and `row_divergence_rate` (tiled greedy rows must agree) is the within-run determinism
monitor; **a multiple of the mesh data-axis size** (e.g. 8) = data-parallel mode — that many
DISTINCT queries per forward with a per-example `[B, M]` mask (chunked/replicated lookup
only), ~B× query throughput since decode is bandwidth-bound, no divergence metric. Reports
`rag_any_gold@k` / `rag_all_golds@k` (multi-hop questions have 2–4 golds — all-golds is the
recall that binds); approx top-k stays on per
[retrieval-modes](../architecture/retrieval-modes.md) policy.

`gather_bank: true` (2026-07-21) replaces full-bank replication with **per-query small
banks**: the corpus bank stays in host numpy and each batch ships only its rows' top-k docs'
slots (one fixed block per row + per-example mask). Judge-equivalent to the mask (candidate
sets identical; text can drift via bf16 top-k tie-breaking), the per-token scan shrinks to
B×k docs, the ≤4M-slot HBM guard no longer applies, and the aux telemetry
(`doc_hit_rate`/`mem_pos_weight_mass`) fits again at large corpora. Requires
`max_chunks_per_doc: 1`. `rag.oracle: true` masks to the query's own `pos_doc_ids`
instead (+ `rag.oracle_fill_to`). Multi-host safe; on a slice only the JAX rank-0 host writes
results/manifest and runs the judge. Design + bring-up:
[implementation note](../implementations/2026-07-20-rag-hybrid-evaluator.md).

**`multi_sample: true`** (2026-08-24, GRPO-readiness diagnostic) — repurposes tiled mode
(`dataset.batch_size: 1`) for **k independent samples of one query** instead of a determinism
check. Tiled mode already runs one query replicated across every mesh data-axis row
(`n_rows`, e.g. 4 on a single-host v6e-4); at `temperature: 0.0` those rows decode identically
by construction (used to compute `row_divergence_rate`), but at `temperature > 0` they are
genuinely independent draws (`jax.random.categorical` assigns independent noise per batch row
even from one PRNG key) — **for free**, no extra generation compute over the standard
single-sample eval. With `multi_sample: true`, all `n_rows` completions are kept (instead of
collapsing to row 0) and written as separate `results` entries sharing a `group_id` (the query
index) and a `sample_idx` (0..n_rows-1); retrieval-telemetry (`doc_hit_rate`/
`mem_pos_weight_mass`) is computed per-sample too, since it depends on the generated answer
span even though the retrieved doc set is identical across the group. `file_metrics.group_size`
records `n_rows`. Raises if `multi_sample: true` without `dataset.batch_size: 1` — the group
*is* the tiled replica set, there's no other axis to draw samples from. The
`row_divergence_rate`/"non-deterministic" warning is suppressed in this mode (divergence is
the point, not a bug). Post-hoc pass@k / intra-group-variance metrics are computed by a
separate standalone script from the resulting samples, not by the evaluator itself — see
[implementation note](../implementations/2026-08-24-grpo-readiness-multisample-hybrid-eval.md).

## Common knobs
`num_samples` · `max_new_tokens` · `temperature`/`top_k`/`top_p` · `output_file` ·
`lookup_chunk_size` (temporarily sets `mem_lookup_chunk_size` for chunked retrieval) ·
`doc_dataset` (large-mem) · `embed_batch_size`/`max_docs` · `answer_slot_diag` /
`doc_access_acc` diagnostics (embed path). Env knobs from
[../architecture/memory-layer.md](../architecture/memory-layer.md) (`MEM_TOP_K`,
`MEM_SCORE_ACTIVATION`, …) apply at eval time.
