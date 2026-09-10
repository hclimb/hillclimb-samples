# GRPO-readiness diagnostic: multi-sample generation in the RAG→memory hybrid evaluator

**Date:** 2026-08-24 · **Author:** rohunagrawal (with Claude) · **Status:** done · **Commit:** _TBD_ (uncommitted on `multihop-finetuning`)

## What changed

`GenLargeMemRagHybridEvaluator` (`evals/gen_large_mem_rag_hybrid.py`) gets a new opt-in
`eval.multi_sample` flag: with `dataset.batch_size: 1` (tiled mode) and `temperature > 0`, it
now keeps **every** tiled replica's completion as an independent sample of the same query
(tagged with `group_id`/`sample_idx`) instead of collapsing to one representative row. This
turns the existing hotpotqa hybrid full-corpus eval into a GRPO-readiness diagnostic: sample
k completions/question, judge each, then compute pass@k and intra-group reward variance
post-hoc from the same JSON the eval already writes — no separate pipeline, no new dataset, no
change to any other caller of this evaluator.

A new standalone script, `scripts/analysis/grpo_readiness_metrics.py`, does the post-hoc
computation: unbiased pass@k (Chen et al. 2021 estimator) and intra-group variance of both
binary judge correctness and the continuous 0-5 judge score, plus the fraction of groups that
are all-correct / all-incorrect / mixed (mixed = the only groups where GRPO's group-normalized
advantage is nonzero).

## Motivation & context

The ask: check whether the current mid-trained checkpoint
(`gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55/qwen3_mem_embed/90000`)
could be improved with GRPO, and separately, see its performance on the standard hotpotqa
hybrid full-corpus eval. No sampling-groups, pass@k, or intra-group-variance machinery existed
anywhere in the repo prior to this (verified: every evaluator generates exactly one completion
per question; `num_samples` everywhere means *number of distinct eval questions*, never
samples-per-question) — see [evaluator-types](../evaluation/evaluator-types.md) and
[metrics](../evaluation/metrics.md) for the prior state. There is also no GRPO/PPO/RL
scaffolding anywhere in the repo proper (confirmed via a full grep + reading every training
doc/loss registry entry); this diagnostic is a prerequisite check before investing in an actual
training loop, not a training implementation itself.

## Options weighed

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Which eval to adapt | Extend the existing `generation_large_mem_rag_hybrid` evaluator/task (`gen_large_mem_msa_hotpotqa_hybrid`, the same one `eval_msa_hybrid.sh DS=hotpotqa` runs — full 9,811-doc corpus) | A brand-new closed-book evaluator keyed to `configs/dataset/sources/hotpotqa_hard_neg_reasoning_modified.yaml` (a separate HotpotQA hard-neg SFT source this checkpoint never trained on) — considered first, but the user redirected: reuse the pipeline that already answers "how does this checkpoint do on the standard hotpotqa hybrid eval," and get the GRPO diagnostic from the SAME run instead of building a second one |
| How to get k samples/query | Repurpose the evaluator's existing **tiled mode** (`dataset.batch_size: 1`, one query replicated across the mesh's whole data axis) — at `temperature > 0` those replicas are independent draws for free (`jax.random.categorical` assigns independent noise per batch row from one PRNG key), so k = the mesh data-axis size falls out of the SAME forward pass tiled mode already runs for its `row_divergence_rate` determinism check | A new sampling loop that calls `inference.generate()` k times per query with distinct keys — would work, but re-derives a mechanism the evaluator already has, and costs the same generation compute without reusing any existing structure |
| Group size (k) | k = 4, via a single-host **v6e-4** (`ct6e-standard-4t`) box — group size is a box choice (mesh data-axis size), not a script parameter | k = 8 on a v6e-8 — more samples/group for a tighter variance estimate, but the user chose v6e-4/k=4 for this pass |
| Where pass@k/variance are computed | A separate standalone post-hoc script (`scripts/analysis/grpo_readiness_metrics.py`) reading the eval's already-written, already-judged JSON | Computing them inside the evaluator itself — would require the evaluator to know about pass@k/GRPO semantics, coupling a generic generation+telemetry evaluator to one specific downstream analysis; the post-hoc split matches CLAUDE.md's "standalone scripts reuse existing code, don't instrument the core eval path" convention |
| Retrieval telemetry (`doc_hit_rate`/`mem_pos_weight_mass`) per sample | Extended to be genuinely per-sample (the aux forward now runs over the real n_rows distinct completions instead of n_rows tiled copies of row 0) | Leaving it query-level (computed once, copied across the group) — retrieval (the doc *set*) is identical across a group, but the metric is measured on the *generated answer span*, which differs per sample; leaving it query-level would silently mix a per-query quantity into what's presented as a per-sample table |

## How it was built & integrated

`evals/gen_large_mem_rag_hybrid.py::GenLargeMemRagHybridEvaluator.evaluate`:
- New `multi_sample = bool(self.cfg.get("multi_sample", False))`. Raises if `multi_sample=True`
  and the batch isn't tiled (`dataset.batch_size != 1`) — the group *is* the tiled replica set.
- `gen_rows` keeps all `n_rows` tiled completions instead of `gen_np[:1]` when `multi_sample`;
  a new `n_group = gen_rows.shape[0]` unifies indexing across all three cases (plain tiled: 1,
  multi-sample tiled: n_rows, data-parallel: B) — every per-row loop was rewritten from
  `for r in range(B)` to `for s in range(n_group)`, with `qidx = 0 if tiled_mode else s`
  indexing into the *real* per-query arrays (`pos_sets`, `masks_np`, `prompt_texts`, …, still
  sized B) and `rr = s` indexing into the gathered aux-telemetry arrays (unifies the old
  `rr = 0 if tiled_mode else r` — `keep=1`/`n_group`/`B` in the three cases makes `rr=s` correct
  in all of them without a branch).
- The aux-telemetry forward feeds `gen_rows` directly (the real n_rows distinct completions)
  instead of tiling row 0, and `keep = n_group if (tiled_mode and multi_sample) else (1 if
  tiled_mode else B)` so `doc_hit_rate`/`mem_pos_weight_mass` are computed per-sample.
- Each result dict gains `group_id` (the query index) and `sample_idx` (0..n_rows-1 under
  multi_sample, else 0) — the only new fields; every existing field is unchanged.
- `file_metrics` gains `multi_sample` (bool) and, when true, `group_size` (= n_rows). The
  `row_divergence_rate` "non-deterministic" warning is suppressed under `multi_sample` (rows
  diverging is the point, not a bug).
- Non-multi_sample paths (every existing caller) are unchanged: verified by hand that `qidx`,
  `rr`, `keep`, and `n_group` all reduce to their exact prior values when `multi_sample=False`.
- **Bug found and fixed during smoke-testing:** the pre-existing tail-overshoot trim
  (`if len(results) > num_samples: results = results[:num_samples]`, meant for the case where
  `num_samples` isn't a multiple of `dataset.batch_size`) compared a ROW count against
  `num_samples`, which always meant QUERY count. Under `multi_sample`, one query yields
  `group_size` rows, so a 4-query run with `group_size=4` produced 16 rows and got silently
  truncated to the first 4 — i.e. just query 0's own group, discarding 3 of the 4 requested
  queries entirely. Fixed by scaling the trim target: `num_samples * n_rows if multi_sample else
  num_samples`. Every non-multi_sample case is untouched (`n_rows` doesn't enter the formula).

New run script `scripts/embed/eval_msa_hybrid_multisample.sh` — sibling to `eval_msa_hybrid.sh`,
same env-var contract (`DS`/`RUN_DIR`/`STEP`/`NUM_SAMPLES`), adds `TEMPERATURE`/`TOP_K`/`TOP_P`
(default `0.6`/`20`/`0.95`, matching the `force_thinking` generation defaults used elsewhere in
the repo) and always passes `evals.msa_hybrid.eval.multi_sample=true
evals.msa_hybrid.dataset.batch_size=1`. Writes to a distinct output name
(`msa_<ds>_c10000_hybrid_multisample<suffix>.json`) so it can never collide with a plain
`eval_msa_hybrid.sh` run's results for the same run-dir/step.

New standalone script `scripts/analysis/grpo_readiness_metrics.py` — takes a local path or
`gs://` URI (via `gsutil cat`, same account convention as `scripts/misc/read_eval_samples.py`),
groups `samples` by `group_id`, and for each group computes `pass_at_k(n, c, k)` (the
unbiased Codex-paper estimator) for every `k` up to the smallest observed group size, plus
`np.var` of the binary `llm_judge_accuracy` list and the continuous `llm_judge_score` list.
Aggregates: mean pass@k per k, mean intra-group variance (binary and score), and the
all-correct / all-incorrect / mixed group fractions. No TPU/JAX import — pure Python + numpy,
runs anywhere.

## Reference pages updated

- [evaluation/evaluator-types.md](../evaluation/evaluator-types.md) — new `multi_sample: true`
  subsection under `generation_large_mem_rag_hybrid` documenting the mechanism, the new
  `group_id`/`sample_idx` fields, and the `dataset.batch_size: 1` requirement.

## Tests

`uv run python tests/test_grpo_readiness_metrics.py` (pure Python, no TPU — covers the
estimator and grouping logic used by `scripts/analysis/grpo_readiness_metrics.py`):

```
PASS pass_at_k_all_correct
PASS pass_at_k_all_incorrect
PASS pass_at_k_monotonic_in_k
PASS pass_at_1_equals_mean_accuracy
PASS group_samples
ALL PASS
```

The evaluator change itself has no CPU-only unit test (unlike `test_rag_hybrid_mask.py`'s pure
helpers, the new logic is inside the JAX-dependent generation loop) — verified instead with a
real smoke run on TPU before the full run, per CLAUDE.md's "smoke-test anyway" convention.

**First smoke run** (`NUM_SAMPLES=4`) hit the checkpoint-restore crash fixed in
[2026-08-24-checkpoint-restore-cross-topology-sharding](2026-08-24-checkpoint-restore-cross-topology-sharding.md)
— unrelated to this change (fails before the evaluator ever runs), documented there.

**Second smoke run**, same command, after that fix: succeeded, but exposed the tail-overshoot
trim bug above — `NUM_SAMPLES=4` produced `generated_count: 4`, all four rows `group_id: 0`
(one query's full group, the other 3 requested queries silently dropped).

**Third smoke run**, after the trim-target fix, `TPU_NAME=tpu-v6e-4-flex ZONE=europe-west4-a
PROJECT_ID=memory-layers TRANSPORT=gce RUN_SCRIPT_PATH=scripts/embed/eval_msa_hybrid_multisample.sh
RUN_ENV="DS=hotpotqa RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55
STEP=90000 NUM_SAMPLES=4" bash scripts/infrastructure/multi-vm-tpu-run.sh` — exit 0, correct
shape this time:

```
metrics.generated_count: 16
metrics.multi_sample: true
metrics.group_size: 4
groups: {0: [0,1,2,3], 1: [0,1,2,3], 2: [0,1,2,3], 3: [0,1,2,3]}   # 4 distinct queries, full groups
```

Samples within a group are genuinely distinct generations (verified by inspection — 4 different
answer texts per group, not 4 copies), and every sample carries `llm_judge_accuracy`/
`llm_judge_score`/`doc_hit_rate` populated per-sample. Ran the real post-hoc script against this
output (`scripts/analysis/grpo_readiness_metrics.py`) and got a sane, non-degenerate result on
real (if tiny, n=4-group) data:

```json
{
  "n_groups": 4,
  "pass_at_k": {"pass@1": 0.125, "pass@2": 0.25, "pass@3": 0.375, "pass@4": 0.5},
  "mean_intra_group_binary_variance": 0.09375,
  "mean_intra_group_score_variance": 1.546875,
  "group_reward_composition": {"all_correct_frac": 0.0, "all_incorrect_frac": 0.5, "mixed_frac": 0.5}
}
```
(4 groups is far too small to draw any real conclusion from — this is a pipeline-correctness
check, not the actual diagnostic result. The full `NUM_SAMPLES=128` run is the real deliverable;
see the companion experiment write-up once it lands.)

## Follow-ups & risks

- `multi_sample` requires tiled mode, so group size is fixed to whatever box you run on (the
  mesh's full data-axis size) — there's no way to get e.g. k=4 groups on a v6e-8 box without
  wasting 4 of the 8 chips, or k=16 groups on a v6e-4 box without two tiled rounds per query
  (not implemented). Fine for this diagnostic; would need revisiting for anything wanting a
  group size decoupled from the box shape.
- Retrieval telemetry per-sample means the aux forward's HBM cost is unchanged per call (still
  one call per query, same `n_rows`-wide batch it was before) — no new OOM risk introduced.
- The post-hoc script assumes every sample in a group has both `llm_judge_accuracy` and
  `llm_judge_score` already computed (i.e. both metrics were enabled on the task, as they are
  on `gen_large_mem_msa_hotpotqa_hybrid.yaml`); a group with a missing judge field is silently
  excluded from `pass_at_k`/variance aggregates and counted in `groups_missing_judge_score`.
