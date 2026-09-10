# RAG→memory hybrid evaluator (`generation_large_mem_rag_hybrid`) + multi-host eval fixes

**Date:** 2026-07-20 · **Author:** rohunagrawal (with Claude) · **Status:** done ·
**Branch:** pending PR (uncommitted at time of writing)

**What changed:** a new evaluator, `evals/gen_large_mem_rag_hybrid.py`, that runs a dense
retrieval pre-pass per query (same vanilla Qwen3-Embedding-0.6B + settings as the classic-RAG
baseline) and restricts the shared memory bank to the top-k retrieved docs' slots via a
per-query `mem_mask` during generation. Shipping it also required making the **entire eval
path multi-host safe** — no eval had ever run on a multi-host mesh, and checkpoint load,
document embedding, the RAG encoder, and the vLLM judge all broke on the 2×`ct6e-standard-4t`
v6e slice ([runbook §2.3](../infrastructure/experiment-launch-instructions.md)).

## Motivation

[2026-07-19 corpus-scaling](../experiments/2026-07-19-musique-corpus-scaling-and-throughput-pareto.md)
found the memory layer's accuracy decay is **attention dilution, not retrieval failure**
(`doc_hit_rate` 0.94+ while `mem_pos_weight_mass` collapses 4×), and named "fix dilution, not
retrieval" the #1 verdict-changing follow-up. The hybrid is a causal intervention on exactly
that: prune ~90% of distractor slots before the memory layer scores them — same checkpoint,
same bank, only the candidate set changes.

## Options weighed

- **Mask vs. gather vs. per-example mask op.** Chosen: rewrite the global `[M]` `mem_mask`
  per query. `retrieval_ops` applies mem_mask as `float32.min` *before* top-k
  (`models/retrieval_ops.py::_matmul_top_k` / `_scan_chunks`), so masking ≡ physically
  shrinking the bank, with no flat-index remapping and no recompile (shape/dtype constant).
  Gathering a small per-query bank would remap indices and complicate telemetry; a `(B, M)`
  per-example mask would touch every retrieval path incl. training. Cost of the mask approach:
  one query per step (`dataset.batch_size: 1` enforced).
- **Inline retrieval vs. retrieval-file plumbing.** Chosen: inline —
  `build_encoder`/`encode_texts` are imported from
  `evals/rag/single_embedding_retrieval.py`, so encoder parity with the RAG baseline
  (`scripts/misc/rag_only.py`: 256/128 max lengths, no instruct prefix) is by construction and
  there is no cross-run artifact to keep in sync. An earlier plan with a top-k results file +
  normalizer script was dropped as over-engineered.
- **Replicated vs. sharded bank.** Chosen: replicated (the parent's `MEM_REPLICATE_BANK`
  branch): no data-axis sharding, no CPU `mem_v` callback, no pre-norm. Guarded at 4M slots —
  past that, use `GenLargeMemEvaluator`'s sharded path.
- **B=1 on an 8-device data axis.** The query is tiled to `mesh.shape['data']` rows; greedy
  rows must agree, so cross-row divergence is counted and reported (`row_divergence_rate`) as
  a free determinism check. The smoke ran exact (`MEM_APPROX_TOPK=0`) and measured
  `row_divergence_rate: 0.0`; production arms run **approx** per the standing policy
  ([retrieval-modes](../architecture/retrieval-modes.md), set 2026-07-20: approx always
  preferred for speed), with `row_divergence_rate` kept as the monitor that approx stays
  benign over a masked bank.
- **Fixed prompt padding.** Prompts are left-padded to `prompt_pad_len` (default 512) so the
  whole run compiles once; per-query max lengths would re-JIT the generation loop every query.
  Measured: query 1 = 87 s (compile), query 2 = 1.5 s.

## How it was built & integrated

- `evals/gen_large_mem_rag_hybrid.py::GenLargeMemRagHybridEvaluator` — corpus selection is a
  copy of the parent's `inject_query_gold` scan (haystack bit-identical to
  `gen_large_mem_musique_c512`); QA rows are loaded straight from the HF repo (question text +
  `pos_doc_ids`), joined by index and verified per query by a question⊆prompt containment
  assert. Telemetry (`doc_hit_rate`, `mem_pos_weight_mass` on the answer span) reuses the
  parent's numpy helpers against the *restricted* mask. Output JSON matches the parent's
  `{metrics, samples}` contract, so `llm_judge_accuracy` / `lexical_grounding` run unchanged;
  samples additionally carry `rag_doc_ids`, `rag_all_golds_covered`, `n_gold_docs`,
  `rows_diverged`.
- New metric: **`rag_all_golds@k`** (plus `rag_any_gold@k`) over the full similarity ranking.
  MuSiQue questions carry 2–4 golds; all-golds coverage is the recall that bounds a multi-hop
  answer, and the existing `recall@k` (any-gold) hides it.
- Oracle mode (`rag.oracle: true`, `rag.oracle_fill_to: N`): mask = the query's own
  `pos_doc_ids` (+ distractor fill in corpus order), no encoder — the perfect-retrieval
  ceiling separating recall misses from residual dilution.
- Dispatch: `evals/__init__.py` (lazy import), type base
  `configs/eval/generation_large_mem_rag_hybrid.yaml` (the `rag:` block), task
  `configs/eval/tasks/gen_large_mem_musique_hybrid.yaml` (mirrors `…musique_c512`,
  `batch_size: 1`, corpus size = `eval.doc_dataset.target_docs`), runner
  `scripts/embed/eval_musique_hybrid.sh` (`RUN_DIR`/`STEP`/`SIZES`/`RAG_KS`/`ORACLE` env,
  GCS skip-check, upload gated on local results existing — self-selects the rank-0 host).

### Multi-host eval fixes (first eval ever run on a multi-host mesh)

- `utils.py::load_inference_checkpoint` — bare `jax.device_put(weights)` raised on orbax's
  non-fully-addressable restored arrays; now skips them (same guard idiom as
  `save_checkpoint::unshard`).
- `evals/utils.py::global_device_put` (new) — host-replicated numpy → global array via
  `jax.make_array_from_callback`; `jax.device_put(jnp.array(x), P(...))` breaks multi-host
  because the ambient-mesh `jnp.array` materializes a mesh-wide array that device_put then
  refuses to reshard. `embed_documents_tokenized` takes an optional `mesh=` and keeps the old
  path when absent.
- `evals/rag/single_embedding_retrieval.py` — jitted `embed_fn` captured the encoder weights
  as a closure ("Closing over jax.Array that spans non-addressable devices"); weights are now
  a jit argument. Warmup + `encode_texts` placements use a local `_put_batch(mesh=…)`;
  batch-sharded encoder outputs are gathered via `process_allgather` (a plain `np.array` on
  them throws).
- `inference.py::_generate_tokens` — unchanged (the internal `device_put` is a sharding
  constraint under jit and was always multi-host safe; a guard added during bring-up was
  reverted after it interrogated a tracer).
- `evals/shared.py::run_eval_worker` — only the JAX rank-0 worker writes a manifest, and rank
  order ≠ host order on slices; non-rank-0 parents now get an empty manifest instead of
  `FileNotFoundError` (their metrics pipeline was already a no-op via the file-existence gate).
- `evals/vllm.py::start_server` — with `VLLM_TPU_LOCAL_ONLY=1` (set by the slice runner only)
  the judge subprocess gets `TPU_SKIP_MDS_QUERY=1`, `TPU_PROCESS_BOUNDS=1,1,1`,
  `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1`: without it libtpu reads the slice topology from
  instance metadata and the engine proc dies waiting for its peer host. Single-host boxes are
  untouched.

## Reference pages updated

[evaluator-types](../evaluation/evaluator-types.md) (new evaluator section);
[experiment-launch-instructions §2.3](../infrastructure/experiment-launch-instructions.md)
(multi-host eval behavior + ops notes from bring-up). The
[2026-07-19 Pareto page](../experiments/2026-07-19-musique-corpus-scaling-and-throughput-pareto.md)
got a dated correction to its stale hard-neg checkpoint pointer.

## Tests

CPU unit test (pure helpers + the load-bearing pre-top-k masking assumption against the real
`retrieval_ops._matmul_top_k`), run on `tpu-v6e-slice-mig-1wjb` via
`scripts/embed/test_rag_hybrid_mask.sh` (`JAX_PLATFORMS=cpu`, safe on one slice worker):

```
$ uv run python tests/test_rag_hybrid_mask.py
PASS rank_unique_doc_ids
PASS build_doc_slot_mask
PASS gold_coverage
PASS masked_slots_never_retrieved
ALL PASS
```

End-to-end smoke (n=2, c512, k=50, both slice workers, standard launcher): full pipeline
incl. vLLM judge and GCS/wandb upload, exit 0 on both hosts. Sanity numbers:
`mean_active_bank_slots` 7,157/131,072 (top-50 docs' valid tokens only — the validity-AND is
correct), `doc_hit_rate` 1.0, `row_divergence_rate` 0.0, and `rag_all_golds@5 = 0.0` vs
`@25 = 1.0` on the two queries — the multi-hop recall gap the metric exists to expose.

## Follow-ups & risks

- Mild OOD: midtraining banks were ~256–320 docs; a top-50 mask is smaller, and top-k 64 over
  ~12.8k candidate slots is a ~10× higher selection fraction than training saw. The oracle
  arms partially control for this.
- The eval's wall clock is not a throughput measurement (the scan still covers the full bank);
  hybrid serving cost composes from the measured primitives in the 2026-07-19 page.
- `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1` hardcodes the ct6e-standard-4t host shape.
- The `pkill`/`fuser` preamble in the runner is inherited from the v5p sweep scripts; it was
  exonerated for the bring-up failures but has not been re-examined on slices.
