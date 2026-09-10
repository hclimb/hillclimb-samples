# The Training Loop

`trainer/trainer.py::Trainer`. Owns the step, stage transitions, logging, eval, and
checkpointing.

## `__init__`
Parses stages (`parse_training_stages`), `opt_state = optimizer.init(weights)`, freezes the
aux-loss config for JIT (stage-0 overrides applied), and builds the in-loop evaluators (each
with its own dataset/doc_dataset).

## `_train_step` — JIT-compiled, one gradient step
`@jax.jit(static_argnames=(forward, optimizer, aux_loss_config, trainable_patterns),
donate_argnums=_DONATE)`.

> **Buffer donation is OFF by default** (`_DONATE = ()`), and that is deliberate. Donating
> `weights`/`opt_state` makes stage 3 (`main_model` trainable) emit **NaN gradients from a finite
> loss**, which deadlocks the run. Measured on one box at ckpt 16000 / LR=1e-12 / identical
> batches: `(2,3)` → 5/20 steps NaN, `(3,)` (opt_state only) → 5/20, `()` → **0/20**. Donation is a
> *trigger*, not the root cause — `opt_state` is never read by the backward, so what donation does
> is change XLA's buffer assignment and hence the fusion of a backward that sits near a numerical
> cliff (same weights+batch give losses differing in the 5th decimal across modes). Re-enable only
> for repro, via `MEM_DONATE=both|opt`. See
> [2026-07-17-stage3-grad-nan-donation](../implementations/2026-07-17-stage3-grad-nan-donation.md).
- `loss_fn(w)`: `forward(inputs, w, pad_mask, collect_aux)` → `CE = softmax_cross_entropy(logits,
  onehot(targets))`, masked by `loss_masks` and **per-row `ce_enable`** (rows with
  `ce_enable=0` — retrieval-only similarity rows — feed `doc_access_loss` but not CE, so they
  train retrieval without teaching the LM to reproduce the doc). Then `compute_aux_losses(...)`;
  `total = main_loss·ce_weight + aux.total`.
- `value_and_grad` → global grad norm + per-group norms `grad_norm_{mem,embed,value,main}`
  (weight-0 telemetry; freeze zeroes *updates* not grads, so `grad_norm_main` can be nonzero
  even with main frozen — read alongside the stage `trainable_params` log).
- **NaN/Inf guard:** `lax.cond` skips the optimizer update when `total_loss` or `grad_norm` is
  non-finite (counts surfaced as `loss_nan_count` / `grad_nan_count`). `grad_norm` is
  `optax.global_norm(grads)` over **all** params, taken from `value_and_grad` **before**
  `optimizer.update` — so it sees raw grads, and `optax.freeze` cannot mask a NaN from it.
- **Deadlock tripwire:** `skip_fn` returns weights **and** `opt_state` unchanged, so a persistent
  NaN freezes the weights, Adam's `count` never advances, and the LR stays pinned at the
  schedule's `init_value` — the run looks alive while making zero progress (a stage-3 NaN burned
  ~3900 steps this way on 2026-07-16). `trainer.nan_abort_after_samples` (default 50, `0`
  disables) aborts after that many *consecutive* non-finite samples. It is checked at the
  `log_interval` cadence, not per step, because reading the flags forces a device sync.
  Corollary: `loss_nan_count`/`grad_nan_count` are **sampled at `log_interval`**, so they
  under-report by ~`log_interval`× — treat them as a signal, not a step count.

## `train()` — the loop
1. `load_checkpoint` → `(step, opt_state, current_stage_idx)`.
2. **Resume sentinel:** if resuming mid-run (`step>0`), set `current_stage_idx=-1` to force one
   stage transition (else stage-0 settings silently persist). Full resume (`resume_step is None`)
   also restores dataloader position.
3. For each `(tokens, masks)` in `data.generator()`:
   - Stage transition when `get_current_stage_idx(step) != current` (see
     [multi-stage-training.md](multi-stage-training.md)).
   - `process_train_pairs(tokens, masks)` → `inputs, targets, input_masks, loss_masks, ce_enable`.
   - `_train_step(...)`; update pbar.
   - **W&B log** (proc 0): `total_loss`, `ce_loss`, `ce_weight`, `grad_norm`, `lr`, nan counts,
     and every `train/<aux_name>`. **Cadence = `trainer.log_interval`** (default 1): the host pulls
     (`float(ce_loss)`/`int(nan)`/`wandb.log`) run only every N steps, since each pull forces a
     device sync that serializes the pipeline (~10% of step time at N=1). N>1 lets JAX dispatch
     ahead — identical training math, coarser logging; the NaN-skip guard in `_train_step` still
     runs every step. See [../experiments/2026-07-16-train-speed-axes.md](../experiments/2026-07-16-train-speed-axes.md).
   - **Eval** every `eval_interval` (`_run_evals`); **checkpoint** every `checkpoint_interval`
     (`save_checkpoint` + `_save_loader_state`).
4. Final save + `checkpoint_manager.wait_until_finished()`.

## Other
- `_current_lr` reads the Adam `count` off `opt_state` and evaluates the schedule.
- **Dataloader state** (`_save_loader_state`/`_restore_loader_state`): `dataloader_state.json`
  written next to each orbax step via fsspec (warn-only); enables byte-exact stream resume.
- **`GradAccumTrainer`** (`trainer/grad_accum_trainer.py`): used when `grad_accum_steps>1`;
  splits each batch into mini-steps and accumulates grads, passing `mini_batch_offset` so
  positive-doc slot indices stay globally correct.
