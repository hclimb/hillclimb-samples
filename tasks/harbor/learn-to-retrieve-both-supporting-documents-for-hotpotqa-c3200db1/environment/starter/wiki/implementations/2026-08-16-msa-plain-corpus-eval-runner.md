# `eval_msa_plain.sh` — one-shot runner for the plain (non-hybrid) MSA corpus eval

**Date:** 2026-08-16 · **Author:** claude (session with rohunagrawal) · **Status:** done ·
**Branch:** `multihop-finetuning`.

## What changed

New `scripts/embed/eval_msa_plain.sh`: a `DS=<dataset> RUN_DIR=<run-dir> STEP=<n>`-parameterized
runner for the **plain, non-hybrid** full-corpus memory eval (`gen_large_mem_msa_<DS>.yaml`, no RAG
assist — the model builds its own memory bank over the whole doc corpus and retrieves via its
trained `mem_lookup`), targeting an arbitrary one-shot checkpoint. Mirrors
[`eval_msa_hybrid.sh`](../../scripts/embed/eval_msa_hybrid.sh)'s shape (skip-if-done via a GCS
`dst` check, judge-server cleanup, `log_eval_to_wandb.py` for GCS + training-run wandb logging).

## Motivation & context

rohunagrawal had just gotten a hybrid (RAG→memory) hotpotqa/musique eval running against several
checkpoints (see the 2026-08-13/2026-08-16 hotpotqa/musique write-ups) and asked for a "normal"
(i.e. non-hybrid) eval of the same checkpoint on musique, for comparison. No existing script
targeted a one-shot checkpoint with the plain corpus-mode eval: `eval_msa_evals.sh` only evaluates
the pretrained MSA-4B baseline (no `checkpoint_dir`), and `scripts/misc/ground_eval_box.py` has the
right recipe but only as a polling eval-box loop over tracked runs, not something you can point at
one checkpoint on demand.

## Options weighed

1. **One-off `uv run eval.py ...` command, not a script.** Rejected: CLAUDE.md's rule for
   benchmarks/experiments is standalone, reusable scripts under `scripts/…`, not ad hoc commands —
   and this exact need (one-shot plain corpus eval against an arbitrary checkpoint) is generic
   enough to recur.
2. **Extend `ground_eval_box.py` to take a one-shot mode.** Rejected: that script's whole shape is
   a polling loop over `DATASET_TASKS`/tracked run-dirs; bolting a one-shot single-task mode onto
   it is a bigger, riskier change than a small new sibling script, for a script already flagged in
   `wiki/evaluation/eval-boxes.md` as needing consolidation, not more forking.
3. **New sibling script mirroring `eval_msa_hybrid.sh`** (chosen). Reuses the established
   `RUN_DIR`/`STEP`/GCS-skip-check/`log_eval_to_wandb.py` pattern already proven by the hybrid and
   RAG runners; the actual eval recipe (task config + `eval.type` override) is lifted directly from
   `ground_eval_box.py`'s already-working `_corpus` task path, not reinvented.

## How it was built & integrated

- `scripts/embed/eval_msa_plain.sh`: composes `+eval/tasks@evals.msa_plain=gen_large_mem_msa_${DS}`
  then overrides `evals.msa_plain.eval.type=generation_large_mem` — the task yaml's own default
  type is `generation_large_mem_msa` (the MSA-4B architecture's evaluator, which wants an `msa` cfg
  block our `qwen3_mem_embed` checkpoints don't have), so this override is load-bearing, exactly as
  in `ground_eval_box.py`. Also carries that script's `dataset.batch_size=8` (halved from the task
  yaml's implicit default) and `tp_devices=1` (the mem-model checkpoint loader isn't TP-aware).
  Result name `msa_${DS}_corpus[NAME_SUFFIX]`, distinct from the hybrid runner's
  `msa_${DS}_c10000_hybrid_autok` so both can coexist under the same run-dir's `eval/step_<N>/`.
- No changes to `eval.py`, `evals/`, or any task config — purely a new orchestration script over
  existing, already-tested machinery.

## Reference pages updated

None — the underlying `generation_large_mem` eval type, the `gen_large_mem_msa_<ds>.yaml` task
family, and the `eval.type` override needed for our architecture are all pre-existing and already
covered by `wiki/evaluation/eval-configs.md` / `evaluator-types.md`; this just gives that recipe a
reusable one-shot entry point, same category as [the MSA sweep tooling
note](2026-07-22-msa-sweep-tooling.md) that added the hybrid/RAG sibling runners.

## Tests

No unit test — this is a thin orchestration wrapper over already-tested `eval.py` machinery
(the `generation_large_mem` eval type and the `_corpus` task recipe are exercised continuously by
`ground_eval_box.py`). Validated by a real-hardware run instead: launched against
`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked` step 100000 on
musique, `rohun-v6e-8-0` — completed clean on the first attempt (`rc=0`, no tracebacks), correct
GCS result upload and training-run wandb log. See
[2026-08-16-musique-plain-corpus-eval-pf32-indexed-lrmasked-checkpoint.md](../experiments/2026-08-16-musique-plain-corpus-eval-pf32-indexed-lrmasked-checkpoint.md)
for the eval result.

## Follow-ups & risks

- Only wired for the MSA family (`gen_large_mem_msa_<ds>.yaml`); a non-MSA plain corpus eval would
  need a different task-config prefix.
- Same consolidation TODO flagged in `wiki/evaluation/eval-boxes.md` applies here in spirit: this
  is now a *third* near-identical eval runner shape (hybrid / RAG / plain) — worth folding into one
  parameterized script if a fourth variant shows up.
