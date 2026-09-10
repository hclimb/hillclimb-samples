# Eval Boxes (evaluating a run while it trains)

**The rule: judged accuracy cannot come from the training loop. It comes from a separate box.**

## Why in-loop eval can't produce judge/grounding metrics

`llm_judge_accuracy` and `lexical_grounding` are computed by `run_metrics()`, which is only ever
called from `evals/shared.py::run_metrics_pipeline` — in **`eval.py`'s parent process**, after the
JAX worker exits and frees the TPU for the vLLM judge ([two-process-design.md](two-process-design.md)).

The trainer has no such split. `trainer/trainer.py::_run_evals` calls `evaluator.evaluate(...)` and
keeps only its returned `inference_metrics`; it never calls `run_metrics`. So:

> **A task's `metrics:` block is INERT during training.** Pointing `trainer.evals` at a task set
> that lists `llm_judge_accuracy` yields no judge numbers — silently, with no error.

What in-loop eval *can* produce: `nll` (+ `doc_access_acc`, + `mem_pos_weight_mass` when the aux
config enables it) from `NLLEvaluator`, which self-logs `eval/<key>/…`. Note that
`generation_embed` / `gen_large_mem` do **not** self-log scalars — they log only a wandb
*artifact* — and `trainer.py`'s `val/*` logging is gated on `not cfg.trainer.evals`, so when
`trainer.evals` is set their `inference_metrics` reach neither wandb nor the console.

The second reason to stay out of the loop: the judge's vLLM wants the TPU that JAX is holding, and
in-loop generation eval costs training throughput. Hence `staged_ground.yaml`'s `eval_interval:
1000000000` and `eval_set: none`.

## The pattern

A box (`scripts/misc/*_eval_box.py`) runs **on its own `rohun-*` v6e-8**, polls GCS for new
checkpoints of a tracked run, evaluates each milestone with `eval.py`, uploads results to
`gs://$GCS_BUCKET/<prefix>/<run>/step<N>/<alias>.json`, and logs the metrics to wandb.

- **Idempotent via GCS**: a step/alias whose result blob exists is skipped, so several boxes can
  shard the same runs by dataset with no coordination (`*_EVAL_DATASETS` per box).
- **Breadth-first**: one checkpoint per run per pass, so several runs' curves fill together
  instead of one run draining its history first.
- **Step-done sentinel** = the *last* task in the list, so a step counts as done only once every
  task landed; a slow/interrupted task is retried, not skipped.

## ⚠ Target a RUN-DIR, not a run_name

**`run_name` is not a run identity.** Every launch mints its own
`<run_name>-<YYYY-MM-DD>-<HH-MM-SS>` dir (`utils.py::run_dir_name`), and re-using a run_name
months later re-uses the prefix. A box that scans `prefix=f"{run_name}-"` therefore sees **every
launch that ever used that name** and takes the max step across all of them.

This is not hypothetical. `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16` had a complete
**April** run at steps 40k–100k. A fresh launch of the same name made a name-scanning box report
`latest_ckpt = 100000`, evaluate the **April model**, and log it into the **new** wandb run at
`train_step=100000` — a plausible-looking curve from the wrong model, with no error anywhere.

`hard_neg_eval_box.py` therefore takes the **run-dir basename** and scans only that dir; a bare
run_name raises. `train.py` prints the dir at startup next to the wandb id. Results are keyed by
run-dir too, so two launches sharing a name keep separate result trees.

**Getting the dir before training exists.** Copying it from `train.py`'s startup line forces a
serial launch. To start both boxes at once, pin the timestamp: `multi-tpu-box-run.sh` mints one
`RUN_START_TIME` and forwards it to every box, so training sets `trainer.run_start_time` and the
box composes the same `RUN_DIR` from `RUN_NAME` + `RUN_START_TIME`. Both compute one identity with
no round-trip:

```bash
bash scripts/infrastructure/multi-tpu-box-run.sh \
    rohun-v6e-8-0=scripts/embed/train_hard_neg_think.sh \
    rohun-v6e-8-1=scripts/embed/hard_neg_eval_box_run.sh
# [multi] RUN_START_TIME=2026-07-16-18-00-00  (one identity for all boxes)
```

The box starting before the dir exists is fine — it lists zero steps and polls, exactly as it
already does while waiting for the first milestone.

`ground_eval_box.py` still scans by name and has this flaw — safe only while its run names are
unique.
- **`use_wandb=false`** is passed to `eval.py` — otherwise it opens its own run in
  `memory-layers-eval` (`init_wandb`). The box does the logging itself.
- **Free the TPU first**: the previous eval's vLLM judge lingers on `/dev/vfio`;
  `VLLMInference.free_tpu_devices()` clears every holder.
- **`tp_devices=1`**: the mem-model checkpoint loader isn't TP-aware; `tp>1` breaks weight loading.

**Milestone vs. rotation.** Orbax keeps only `max_to_keep=4` checkpoints (`utils.py:344`), so the
window in which a checkpoint is evaluable is `4 × checkpoint_interval` steps. Keep the box's
milestone ≥ the checkpoint interval, and keep a cycle's total eval time inside that window — else
checkpoints rotate away unevaluated and the curve gets holes. Lower `num_samples` or shard across
boxes to fit.

## Getting eval metrics into the TRAINING wandb run

Two shapes, both keyed on a **custom `train_step` x-axis** rather than wandb's monotonic step:

```python
run.define_metric("train_step")
run.define_metric("eval/*", step_metric="train_step")
run.log({f"eval/{alias}/{k}": v, "train_step": step})   # NOTE: no step= argument
```

This is required, not stylistic: every task logs at the *same* checkpoint step, and
`wandb.log(step=step)` keeps only the first task per step and **drops the rest** (this is what
silently lost musique from the ground box's curves).

| Shape | How | When |
|-------|-----|------|
| **Companion run** (`ground_eval_box.py`) | `wandb.init(id=f"{run}_geval", resume="allow")` — a *separate* run named `<run>_eval` | Training run has no deterministic id; no coupling to the training process |
| **Same run** (`hard_neg_eval_box.py`) | Training sets `trainer.wandb_run_id=auto`; the box derives the same id from the **run-dir** via `utils.wandb_run_id_from_run_dir` and attaches | You want `eval/*` and `train/*` on one run |

**The id is keyed on the run-dir, not the run name** — the same string that names the GCS folder,
so wandb run ↔ GCS folder is 1:1 and a box pointed at a folder can compute the id without being
told. Keying on the name instead made *every* launch of a name share one id: a relaunch silently
appended to the earlier attempt's curve. It is not byte-equal to the dir — wandb ids are capped
(≤64) and a real dir is longer (`<51-char name>-<19-char stamp>` = 71) — so the name is slugged
and truncated to 40 while the **timestamp is kept in full**, since that is what makes it unique.

**Same-run requires wandb shared mode** (wandb ≥ ~0.19; the boxes run 0.28). Two processes writing
one run is only safe with `mode="shared"`:

- training (`train.py`) = **primary**: `x_primary=True, x_label="train"`
- the box = **secondary**: `x_primary=False, x_label="eval", x_update_finish_state=False`

`x_update_finish_state=False` is load-bearing: the box calls `run.finish()` after each point, and
without it that would mark the still-running **training** run as finished.

**Preemption-resume forks the wandb run.** A resume mints a *new* run-dir → new id → a second
wandb run, with the curve split. That is the price of per-launch uniqueness; the two properties
come from the same mechanism and you cannot have both automatically. To stay on one curve, pass
the **original** id explicitly (`trainer.wandb_run_id=<id>` — any literal is used verbatim), and
point a box at each dir. `wandb_run_id` defaults to `null` (legacy: fresh random id per launch).

## Boxes in-repo

| Box | Runner | Tasks | wandb |
|-----|--------|-------|-------|
| [`ground_eval_box.py`](../../scripts/misc/ground_eval_box.py) | `scripts/misc/ground_box_run.sh` | oracle + corpus × msmarco/hotpotqa/musique | companion `<run>_eval` |
| [`hard_neg_eval_box.py`](../../scripts/misc/hard_neg_eval_box.py) | `RUN_DIR=… scripts/embed/hard_neg_eval_box_run.sh` | `eval_set=hard_neg_think_c512` — msmarco/hotpotqa/musique @512 docs + scienceQA NLL, n=128 | **into the training run** (targets one run-dir) |

## TODO — abstract the box (the two above are ~80% copy-paste)

`hard_neg_eval_box.py` was adapted from `ground_eval_box.py`, and `sim_eval_box.py` before that.
Each new experiment forks the whole file, so **every fork re-inherits the same subtle machinery**
(checkpoint→run-dir resolution, the sentinel, the `train_step` axis) and any fix has to be applied
N times. The `wandb.log(step=…)` bug that dropped musique is exactly the kind of thing a fork
silently carries forward.

Nearly all of it is generic and should move to one module (e.g. `evals/eval_box.py`):

- GCS discovery — `_gcs_exists`, `_steps_by_dir`, `latest_ckpt`, `ckpt_dir_for_step`
- the scan loop — breadth-first round-robin, milestone walk, sentinel idempotency, `SCAN_S` sleep
- per-checkpoint work — `free_tpu_devices()`, the `eval.py` subprocess + overrides, result glob,
  GCS upload
- wandb — `train_step` axis + the shared-mode/companion choice

What actually differs per experiment, i.e. what should be the config surface:

| Knob | ground | hard_neg |
|------|--------|----------|
| task list (alias → task cfg) | 6 oracle/corpus tasks | 4 c512 tasks |
| result prefix | `ground_eval` | `hard_neg_eval` |
| env var prefix | `SIM_EVAL_*` / `GROUND_EVAL_*` | `HARD_NEG_EVAL_*` |
| wandb target | companion `<run>_geval` | training run via `wandb_run_id_from_name` |
| per-task cmd extras | `eval.type` override, bs=1 for `BS1_RUNS` | `num_samples` for non-NLL tasks |
| metric extraction | `_find` heuristic (judge acc only) | all scalars from `metrics`/`stats` |

Note the last row is a real behavioural difference, not just shape: the ground box logs **only**
`llm_judge_accuracy`, so consolidating should take `hard_neg`'s version and let ground gain the
rest. Do this before forking a *third* box.
