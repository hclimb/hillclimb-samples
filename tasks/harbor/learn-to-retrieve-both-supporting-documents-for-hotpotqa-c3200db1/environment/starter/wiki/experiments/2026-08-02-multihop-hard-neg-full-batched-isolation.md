# multihop_hard_neg_full: batched per-row retrieval enables the 200-hard-negative recipe

**Date:** 2026-08-02 **Author:** rohunagrawal (with Claude) **Status:** DONE (infra) — the
retrieval-efficiency result (batched isolation) is fully confirmed; two further bugs were found
and fixed (a `trainer.py` stage-transition OOM, and `doc_access_loss` silently no-op-ing under
`mem_batched_isolation`, which meant the entire session up to that point trained on exactly zero
gradient); training has been restarted from step 0 with both fixes and **confirmed producing real,
fluctuating loss values on real hardware**. Accuracy/eval numbers still pending — needs a
dedicated eval box, not stood up this session.

## Conclusion, first

The colleague's `multihop_hard_neg_full` recipe (200 hard negatives/question, ~204 docs in the
memory bank per query) could not train at a usable batch size on our v6e-8 slice with the default
cross-batch memory retrieval — it OOMs at `batch_size=16` (32.03G vs 31.25G cap) and the recipe's
own config comment already knew `batch_size=8` was the ceiling. A new **true per-row retrieval
path** (`mem_lookup_batched` / `mem_batched_isolation`, `models/memory.py`) removes the cross-batch
join entirely (query `b` only ever scores against its own docs, via a batched einsum on a reshaped
`[B, m, H]` bank instead of masking a `[B, N, T, B·m]` matrix) — verified numerically identical to
the existing masked-isolation path, and **~1.5x faster at matched batch size** (65.3ms vs 97.2ms
@B=8, retrieval kernel only) with **flat peak memory in B** (23.68G @B=8 → 23.71G @B=16, vs the
default mode which can't reach B=16 at all). Training ran into a completely separate, unrelated
bug — a stage-transition `opt_state` rebuild in `trainer.py` that transiently doubled optimizer
memory at exactly the point headroom was thinnest, causing a 100%-reproducible OOM at the Stage
0→1 boundary (step 5000). That's now fixed (verified byte-identical to the old behavior in
`tests/test_stage_transition_opt_state.py`, then confirmed past the boundary on real hardware),
and training is running at `batch_size=8` with the new retrieval path. **`batch_size=16`
end-to-end still doesn't fit** despite the retrieval kernel supporting it in isolation — the
embed model's own memory cost scales with B and wasn't part of that benchmark; this is a separate,
still-open follow-up.

## Hypothesis & motivation

Rohun: with 200 dataset hard negatives already available per query, does the memory layer's
default behavior — every query in the batch attends to *every* row's docs, not just its own — buy
anything, or is it purely wasted compute/memory at this scale? If not needed, restricting each
query to its own docs *structurally* (not via a post-hoc mask) should let training run faster
and/or at a larger batch than the recipe's config comments assumed.

## Setup

- **Model:** `qwen3_mem_embed` — main `Qwen/Qwen3-4B` (frozen) + `Qwen3-Embedding-0.6B` embed
  model (trainable), single memory layer at layer 14, `mem_top_k=128`, `mem_num_heads=4`,
  `mem_k_dim=mem_v_dim=1024`.
- **Data:** `multihop_hard_neg_full` — `mihir-1999/multihop_qa_sft-hard-neg-train` (1,341,045
  rows) with doc-id-resolved hard negatives against `multihop_doc_corpus.arrow` (1,289,524 unique
  docs); `num_chunks_per_doc=256`, `doc_chunk_seq_len=256` → `m_per_query = 65,536` doc-token bank
  slots per query (~204 docs/query: ~4 positives + up to 200 hardest negatives).
- **Independent variable:** retrieval mode — `full_masked` (existing `per_query_isolation` mask on
  the full cross-batch matrix, the recipe's only option before this change) vs **`batched`** (new
  `mem_batched_isolation` — true per-row retrieval) vs the unmasked cross-batch `chunked`/
  `two_pass` paths (representative of "what if isolation weren't applied at all," i.e. the
  O(B)-wasted-compute baseline this whole exercise is about). Held fixed: geometry above,
  `mem_top_k=128`, `mem_approx_topk=true` @ recall 0.99 (repo policy).
- **Metric:** retrieval-kernel-isolated step time (ms) + peak HBM (`jax.local_devices()[0].
  memory_stats()`), swept over batch size `{4, 8, 16, 32}` on the real 8-chip v6e-8 slice (both
  hosts, full mesh) — `benchmarks/bench_hard_neg_full_retrieval.py`, random query/bank (no Qwen),
  matching the pattern of `benchmarks/bench_sharded_retrieval.py` /
  `tests/benchmark_wall_clock.py`. Then a real end-to-end training launch to confirm the winning
  config actually works outside the isolated benchmark.

## Reproducibility

- Benchmark command (sweep): `TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers bash scripts/infrastructure/multi-tpu-box-run.sh tpu-v6e-slice-mig-lhxn=scripts/misc/sweep_hard_neg_full_retrieval.sh tpu-v6e-slice-mig-qvlq=scripts/misc/sweep_hard_neg_full_retrieval.sh`
- Training command: `TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers bash scripts/infrastructure/multi-tpu-box-run.sh tpu-v6e-slice-mig-lhxn=scripts/embed/train_multihop_hard_neg_full.sh tpu-v6e-slice-mig-qvlq=scripts/embed/train_multihop_hard_neg_full.sh` — Hydra overrides: `model=qwen3_mem_embed model.main_model.model_id=Qwen/Qwen3-4B model.memory.mem_top_k=128 +model.memory.per_query_isolation=true +model.memory.isolation_group_size=1 +model.memory.mem_batched_isolation=true dataset=multihop_hard_neg_full trainer=staged_telemetry trainer.checkpoint_interval=200 trainer.max_to_keep=20`
- Commit: `48a9041617aa914fae63ff5ae2b247ceca8f02b5` (`datagen` branch) **+ uncommitted working-tree
  changes** — `mem_lookup_batched` in `models/memory.py`, `configs/dataset/multihop_hard_neg_full.
  yaml` (batch_size, isolation notes), the stage-transition `opt_state` fix in `trainer/trainer.py`,
  new benchmark/test/launch scripts. See
  [implementation note](../implementations/2026-08-02-hard-neg-full-efficient-retrieval.md) for
  the full diff description; not committed (only commit when explicitly asked).
- Checkpoint: `gs://memory-layers-training/multihop_hard_neg_full_batched_iso_topk128_bs8-2026-08-02-03-39-56/qwen3_mem_embed/200/` (first checkpoint; more land every ~7 min at steady state) through
  `gs://memory-layers-training/multihop_hard_neg_full_batched_iso_topk128_bs8-2026-08-02-16-02-39/qwen3_mem_embed/5328/` (first checkpoint confirmed past the Stage 0→1 boundary, post-fix).
- wandb: run name `multihop_hard_neg_full_batched_iso_topk128_bs8`, `wandb_run_id=auto` — resolve
  the exact URL from the run name in the `memory-layers` wandb project.
- TPU: v6e-8 flex-start 2x4 slice, `tpu-v6e-slice-mig-{lhxn,qvlq}`, project `memory-layers`,
  `europe-west4-a`.

## Results, in full

### Retrieval-kernel benchmark (isolated, random q/bank — `N=4, T=512, D=Dv=1024, mem_top_k=128, m_per_query=65,536`)

| Mode | B=8 | B=16 | B=32 |
|------|-----|------|------|
| **batched** (new) | **65.3ms**, 23.68G | **132.9ms**, 23.71G | OOM (needs 16.0G more, 15.1G free) |
| full_masked (existing `per_query_isolation`) | 97.2ms, 23.67G | **OOM** (32.03G used, 31.25G cap) | OOM (wants 68.7G) |
| chunked (no isolation, cross-batch scan) | 910.7ms, 23.68G | 6,096ms, 19.42G | OOM (wants 49.0G) |
| two_pass (no isolation, cross-batch) | 773.4ms, 23.68G | 5,497ms, 32.30G | 21,925ms (slow but survives) |

`B=4` fails for every mode — expected: the mesh is `(data=8, model=1)`, so `B` must be a multiple
of 8 regardless of retrieval mode.

### End-to-end training launch (real Qwen3-4B + Qwen3-Embedding-0.6B, batched isolation)

| Attempt | batch_size | trainer | ckpt_interval | Outcome |
|---------|-----------|---------|---------------|---------|
| 1 | 16 | staged_telemetry | 200 | **OOM on first train step**: `Used 33.66G of 31.25G hbm. Exceeded by 2.42G` — the isolated retrieval benchmark's B=16 result (23.71G) didn't capture the embed model's own memory cost, which scales with B |
| 2 | 8 | staged_telemetry | 200 | Step 1: 280.5s (JIT compile) → steady state ~2.0-2.4s/step. Checkpoints saved cleanly 200→5000 (all verified in GCS). **Crashed at step 5000** (~3h20m in): `Attempting to allocate 5.94M. There are 4.20M free.` |
| 3 (resume) | 8 | staged_telemetry | 200 | Confirmed: crash is **reproducible, not a leak** — resumed process, fresh checkpoint manager, crashed on its very first save, identical numbers. This is `staged.yaml`'s Stage 0→1 boundary (`max_step:5000`): the whole embed model becomes trainable, needs much more gradient memory |
| 4 (resume, fix attempt 1) | 8 | **staged** (dropped telemetry) | 200 | Survived ~15 min of active Stage-1 recompilation (vs instant failure before) — then **crashed anyway**: `5.94M / 4.02M free`. Telemetry removal shifted the margin by ~0.18MB but didn't close it |
| 5 (resume, fix attempt 2) | 8 | staged | **333** (decoupled from stage boundaries) | **Crashed with the identical `4.02M free`** as attempt 4 — the checkpoint-interval change measured **zero** effect. Cleanly falsifies the checkpoint/transition-coincidence theory |
| 6 (resume, **actual fix**) | 8 | staged | 333 | **Crossed the boundary.** Root-caused by reading `trainer.py`'s stage-transition code directly (not guessing config) — see Interpretation below. Reached step 375+ into Stage 1, checkpoint saved cleanly at step **5328**, steady-state ~2s/step, no errors on either host |

**Precise, 100%-reproducible repro (attempts 2-5, pre-fix):** resume `qwen3_mem_embed` +
`multihop_hard_neg_full` + `mem_batched_isolation` from any checkpoint at step 5000,
`trainer=staged`, `batch_size=8`, `mem_top_k=128` → crashed on the very next step with
`RESOURCE_EXHAUSTED: Attempting to allocate 5.94M. There are ~4.0-4.2M free`, every time. Fixed as
of attempt 6.

## Interpretation

**The retrieval-efficiency result is solid and unaffected by any of this.** The retrieval-kernel
benchmark correctly predicted the ranking (batched > full_masked > chunked ≈ two_pass) and the
qualitative finding (batched's memory is flat in B) — it only overstated what batch size would
work *end-to-end*, because it didn't include the embed model's own B-scaling memory cost.
`batch_size=8` with `mem_batched_isolation` is a real, measured, 1.5x-faster-at-matched-B win over
the existing masked path — that conclusion doesn't depend on anything below.

**The training-run blocker turned out to be unrelated to the memory layer entirely.** Two
independent, evidence-based hypotheses (telemetry overhead; checkpoint/stage-transition timing
coincidence) were tried and **both cleanly falsified** — the free-memory number didn't move at
all between the two checkpoint-interval settings. At that point, rather than keep guessing at
config knobs, the user directly challenged an assumption behind both hypotheses (that Stage 1
needs backward through more of the model than Stage 0) by pointing at the actual per-layer
`jax.remat` wrapping in `qwen3.py`/`qwen3_mem.py`. Reading the real code from there settled it:

- `trainer.py`'s `stop_grad_frozen` defaults to `false` and is never overridden by this recipe, so
  `jax.value_and_grad(loss_fn)(weights)` computes gradients over the **entire** weights pytree —
  main 4B model included — in **every** stage. The backward graph never changes shape across
  stages; freeze/masking only gates which computed gradients get *applied*.
- The actual stage-dependent cost was in `utils.py::setup_optimizer_for_stage` +
  `trainer.py`'s transition block: at every stage transition (and forced on *every resume*, via a
  `-1` sentinel that guarantees one re-transition regardless of where the resumed step actually
  falls), the code built a **full second copy** of Adam's optimizer state
  (`self.optimizer.init(...)`, full `mu`/`nu` for every param) while the old one was still
  referenced, then discarded the fresh copy in favor of the old one (a "momentum transfer" step).
  That transient ~2x optimizer-state footprint, landing exactly where headroom was already
  thinnest, is what OOM'd.
- Fix (verified in `tests/test_stage_transition_opt_state.py` before touching `trainer.py`, then
  applied): reuse `opt_state` in place across a transition and reset only the count-bearing state
  nodes the LR schedule depends on — no second allocation, ever. The test caught a real bug in an
  earlier draft of the fix (resetting only the Adam-moments node's counter, missing a *separate*
  schedule-counter node) before it reached production code.
- **Confirmed on real hardware**: resumed from the step-5000 checkpoint with the fix applied,
  crossed the boundary cleanly, reached step 375+ into Stage 1 with a checkpoint at step 5328.

Full diagnosis chain, all falsified/confirmed hypotheses, and the fix's code are in the
[implementation note](../implementations/2026-08-02-hard-neg-full-efficient-retrieval.md).

No accuracy signal yet — training is past Stage 0 now but hasn't run long enough for a meaningful
checkpoint to evaluate. The eval side (judged accuracy vs `qa_hard_neg_think_sft4b` / RAG
baselines) needs a dedicated eval box (`wiki/evaluation/eval-boxes.md`), not stood up this
session — the natural next step once training has run further.

**Still open, unrelated to tonight's fix:** `batch_size=16` end-to-end OOMs for a different reason
(embed-model memory cost scaling with B) — not attempted; and `_train_step`'s `forward` static
jit argument is rebuilt fresh every stage transition, forcing an unnecessary multi-minute
recompile at every transition and resume (a real performance cost, found while fixing the OOM,
not itself fixed).

## A second, larger bug found after the Stage-transition fix: zero gradient the entire session

Every progress line, every attempt, the whole night, printed `Loss: 0.0000` — `total_loss`, the
actual differentiated quantity, not a diagnostic. Missed because `CE` (shown alongside it)
fluctuated normally and Stages 0-1 deliberately zero `ce_weight`, making a static `CE`
contribution *expected* — masking that `Loss` itself never moved. Caught only when asked directly
whether `mem_batched_isolation` is compatible with `doc_access_loss` (`trainer=staged`'s only
other nonzero-weight loss in Stages 0-1).

**It wasn't.** `losses/doc_access_loss.py` needs `aux_data["mem_scores"]` as the full
*cross-batch* `[B,T,N,B·m_per_query]` grid (an explicit in-batch-negatives design — its own
docstring: *"Queries compete against all docs in the batch"*). `mem_lookup_batched` never
populates that key, so the loss's own guard (`if not mem_scores_list: return 0.0`) fired every
step. With `ce_weight=0.0` in Stages 0-1 and this the only other nonzero-weight loss,
`total_loss` was a literal constant `0.0` — for the entire session up to that point, across every
attempt.

**Fix:** a new loss, `doc_access_per_query_loss` — the exact same `log_z - log_pos` objective,
scoped to a query's own `m_per_query` slots instead of the batch-global grid (no `block_eye`
cross-query construction needed at all: a query's own slots can't contain another query's docs
under per-row isolation, so the thing `block_eye` exists to prevent is already structurally
impossible). Fed by a *new*, opt-in capability in `mem_lookup_batched`
(`model.memory.mem_collect_full_scores=true`) exposing the full **per-row** grid,
`[B,T,N,m_per_query]` — cheap here specifically because it's the same size class the whole
`batched` mode already runs at, unlike the cross-batch grid `doc_access_loss` needs. Found and
fixed alongside it: `mem_lookup_batched` was also promoting its score tensor from bf16 to fp32 via
`/ jnp.array(H, dtype=float32)`, and mislabeling `aux_data["mem_top_k_logits"]` with
post-activation values instead of the raw logits every sibling function stores there — both real
bugs, independent of this one, caught while checking dtype for this fix. Verified in
`tests/test_doc_access_per_query_loss.py` against an independent numpy reference (not
`doc_access_loss` itself, so a shared bug in both couldn't cancel out) — full detail and test
output in the [implementation note](../implementations/2026-08-02-hard-neg-full-efficient-retrieval.md).

**Restarted from step 0** (new trainer config `staged_batched_isolation`, new run name) rather
than resuming — every checkpoint produced before this fix was trained on a constant-zero-gradient
function, so there was nothing to preserve.

**CONFIRMED on real hardware:** `Loss` now fluctuates genuinely step to step (~0.26-0.39 sampled
over the first 350 steps: `0.2930, 0.3585, 0.3334, 0.3433, ...`), not a stuck constant — real
gradient signal, for the first time this session. Checkpoints landing cleanly on both hosts
(step 333 confirmed on both).
