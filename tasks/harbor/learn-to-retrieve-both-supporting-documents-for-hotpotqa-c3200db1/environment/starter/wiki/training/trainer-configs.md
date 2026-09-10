# Trainer Configs

`configs/trainer/*.yaml` — hyperparameters + the training regime. Selected by `trainer=<name>`.
`configs/train.yaml` composes `model=qwen3_mem_embed`, `dataset=pretraining_cot`,
`trainer=staged` by default.

## `standard.yaml` — base fields
| Field | Meaning |
|-------|---------|
| `steps` | total gradient steps (100000) |
| `learning_rate` / `weight_decay` / `clip_grad_norm` | adamw + clip (1e-4 / 0.1 / 1.0) |
| `ce_weight` | CE weight (0.1; usually overridden per-stage) |
| `checkpoint_interval` / `eval_interval` | save / in-loop eval cadence |
| `tp_devices` | tensor-parallel degree ([../architecture/conventions.md](../architecture/conventions.md)) |
| `resume_from` | run dir to resume from (null = fresh) |
| `use_wandb` / `wandb_project` | logging |
| `run_start_time` | `null` (default) = the run-dir's timestamp comes from `hydra.run.dir` at launch. Set to `YYYY-MM-DD-HH-MM-SS` to **pin** it, making the run-dir (and wandb id) knowable *before* training starts — how [`multi-tpu-box-run.sh`](../infrastructure/experiment-launch-instructions.md) gives a training box and an eval box one identity in parallel. Malformed values raise at startup. Reusing a value with the same `run_name` re-creates the collision run-dirs prevent. |
| `wandb_run_id` | `null` (default) = fresh random id per launch. `"auto"` = deterministic id from this launch's **run-dir** (`utils.py::run_dir_name`, `{run_name}-{date}-{time}`), making the wandb run and the GCS folder 1:1 so an [eval box](../evaluation/eval-boxes.md) pointed at the folder can log `eval/*` into this run. Unique per launch, so re-using a `run_name` can't append to an older run. A preemption-resume mints a new dir ⇒ a **second** wandb run; pass a literal id here to stay on one curve. |
| `nan_abort_after_samples` | `50`. Abort after this many **consecutive** non-finite samples (sampled at `log_interval`, so 50 ≈ 5000 steps at the default 100). A persistent NaN grad freezes weights *and* `opt_state`, so Adam's `count` never advances and the LR stays pinned at the schedule's `init_value` — the run looks alive but makes zero progress. `0` disables. See [training-loop.md](training-loop.md). |
| `aux_losses` | base per-loss `{enabled, weight, …}` ([auxiliary-losses.md](auxiliary-losses.md)) |

No `training_stages` → single-phase. Adding `training_stages` switches on staging
([multi-stage-training.md](multi-stage-training.md)).

## Variants
| Config | Regime |
|--------|--------|
| `standard` | single-phase, all-trainable-by-default |
| `staged` | **default.** 4-stage `qwen3_mem_embed`: warmup mem+conv → +embed → +CE → unfreeze main (cosine) |
| `staged_sim` | extended 150k horizon (QA + text-pair similarity); stage-3 peak LR 1e-5 to protect the 4B |
| `staged_ground` | grounding recipe: **2 stages, main frozen throughout**, in-loop eval off (dedicated eval boxes), full weight-0 telemetry block, zero-init `mem_o_proj` |
| `standard_ground` | `standard` + the weight-0 telemetry block (full-FT msmarco isolation runs) |
| `staged_telemetry` | `staged` + the weight-0 telemetry block. NOT a `staged_ground` variant — that name is a different *recipe* (frozen main), not "staged + telemetry" |
| `staged_distill` | `staged` + `distillation_loss` (teacher→student) |
| `staged_msa` | MSA two-phase schedule (paper §3.3.1): `msa_route_loss`; disables the mem_embed losses |
| `midtraining` | single-stage fine-tune from a checkpoint: unfreeze mem + embed + main layers 13/14/15, cosine. Sets per-group `learning_rates: {mem: 1e-4, embed: 1e-5, main: 1e-5}` (mirrors `staged`'s own group rates; `setup_optimizer_for_stage`'s opt-in per-group path) — inherited by `midtraining_telemetry`/`midtraining_frozen_telemetry`/`midtraining_full_telemetry`. A plain `trainer.learning_rate=` override no longer has any effect once this is set. |

## Per-stage overrides
`trainable_params`, `max_step`, `ce_weight`, `aux_losses`, `lr_schedule`, `learning_rate`,
`warmup_frac`, `min_lr` — see [multi-stage-training.md](multi-stage-training.md). The Hydra
mechanics of the wider config tree belong to the (future) configuration section; dataset
configs to the data section.
