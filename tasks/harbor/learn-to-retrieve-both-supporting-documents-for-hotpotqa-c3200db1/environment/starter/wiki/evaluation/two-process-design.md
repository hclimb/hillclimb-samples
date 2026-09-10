# Two-Process Design

`eval.py` runs in **two processes** so JAX and the vLLM judge never hold the TPU at once:
a JAX subprocess does all model inference then **exits, releasing the TPU**; the parent (no
JAX) then runs judge/metrics. Orchestrated in `evals/shared.py`.

## Parent (`eval.py` → `evals/shared.py`)
0. **`setup_gcs_credentials()`** — MUST run before step 1. It points `GOOGLE_APPLICATION_CREDENTIALS`
   at the user's `adc.json` (from `GCS_USER_EMAIL`). Without it, the `gs://` config read in step 1
   falls back to the box's compute service account, which **403s** on `memory-layers-training`.
   Both `eval.py` and `rag_eval.py` call it first.
1. **`resolve_train_cfg(cfg)`** — from `checkpoint_dir` (`<run_dir>/<model>/<step>`), loads
   `<run_dir>/.hydra/config.yaml` (via `gcsfs` for `gs://`) to reconstruct the exact training
   model architecture. (No `checkpoint_dir` → use `cfg` as-is.)
1b. **`apply_checkpoint_model_cfg(cfg, train_cfg, cli_overrides)`** — makes the checkpoint's saved
   `model` config **authoritative** when a checkpoint is loaded, so the eval's default `model`
   (`configs/eval.yaml` → `qwen3_mem_embed`) cannot silently clobber the trained architecture (e.g.
   rebuild `mem_layers=[9,14,20,27]` as `[14]`). Explicit `model.<path>=…` CLI overrides are layered
   on top; an explicit `model=<group>` swap is respected as-is. See
   [2026-07-18-eval-checkpoint-config-authoritative](../implementations/2026-07-18-eval-checkpoint-config-authoritative.md).
2. **`init_wandb`**.
3. **`run_eval_worker`** — dumps eval+train cfg to JSON, spawns
   `python -m evals.eval_worker --eval-cfg … --train-cfg … --output-dir … --manifest-out …
   [--checkpoint-dir --step]`, waits, reads back the **manifest**.
4. **`run_metrics_pipeline`** — see below.

## JAX worker (`evals/eval_worker.py`)
- **Pre-JAX:** `_collect_deferred_evals` pulls out any `generation_base` evals (they use vLLM,
  not JAX) as deferred manifest entries. If *all* evals are deferred, it skips JAX init entirely.
- `jax.distributed.initialize()`; `effective_model_cfg = merge(train_cfg.model, cfg.model)`. This
  merge is now a **no-op safety net**: the parent already made `cfg.model` checkpoint-authoritative
  in step 1b, so `cfg.model ⊇ train_cfg.model`. (Historically this merge is where the eval default
  silently overrode the trained architecture — see step 1b.)
- **Sorts evals by `tp_devices` (desc)** so all evals at one TP degree run together — the model
  reloads at most once per distinct `tp_devices`.
- `get_model` + `load_inference_checkpoint` (Orbax `CheckpointManager` + `PyTreeCheckpointer`).
- **aux_loss config:** taken from `train_cfg.trainer.aux_losses`, merged with any `cfg.aux_losses`
  overrides (so eval can enable `doc_access_acc` etc.).
- For each eval: reload model if its `tp_devices` differs; load `dataset` (+ optional
  `doc_dataset`); `evaluator.evaluate(model, dataset, step, doc_dataset, aux_loss_config)` →
  `inference_metrics`; record `{output_file, metrics_cfg, inference_metrics}` in the manifest.
- Writes the manifest (proc 0 only).

## Metrics parent (`run_metrics_pipeline`)
For each manifest entry: run any deferred `generation_base` (`gen_base_model.run_generation`,
vLLM); collect `inference_metrics`; if `output_file` + `metrics_cfg` exist, load the samples,
`run_metrics` (LLM judge / grounding — see [metrics.md](metrics.md)), write annotated samples +
scores back. Returns `{eval_key/metric: value}`.

## Manifest entry
`{output_file, metrics_cfg, inference_metrics, [deferred_type, deferred_eval_cfg,
deferred_dataset_cfg]}` — the hand-off contract between the two processes. Raw samples land under
`<out>/eval_results/step_<step>/<eval_key>/<output_file>`.
