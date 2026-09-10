# Training

How a training run is wired and how to change its behavior. Entry point: **`train.py`**
(`uv run train.py`). To actually launch on TPUs, see
[../infrastructure/experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md).

## Flow (`train.py`)
Hydra loads `configs/train.yaml` → `jax.distributed.initialize()` →
`get_model(cfg.model, tp_devices)` → `setup_optimizer(cfg, model)` →
`get_dataset(cfg.dataset, model)` → `setup_checkpointing(cfg)` → `wandb.init` (proc 0) →
`Trainer(...).train()`. Picks `GradAccumTrainer` when `trainer.grad_accum_steps > 1`, else
`Trainer`.

## Total loss
```
total = ce_weight · CE  +  Σ_i weight_i · aux_i
```
CE is per-token softmax cross-entropy, masked to answer tokens and per-row gated by
`ce_enable`. Weight-0 aux entries are stop-gradient (pure telemetry). Both `ce_weight` and the
aux weights are **per-stage**.

## Pages
- [training-loop.md](training-loop.md) — the `Trainer` step & loop: NaN guard, logging, eval, checkpoint cadence, dataloader resume
- [multi-stage-training.md](multi-stage-training.md) — staged trainable-params / loss-weight / LR-schedule + stage transitions
- [optimizer.md](optimizer.md) — the optax chain, freeze mask, LR schedules, LoRA activation
- [auxiliary-losses.md](auxiliary-losses.md) — the loss registry and every registered loss
- [trainer-configs.md](trainer-configs.md) — `configs/trainer/*.yaml`

## Quick start
```bash
uv run train.py                                   # default: qwen3_mem_embed + pretraining_cot + staged
uv run train.py model=qwen3 dataset=qa trainer=standard
uv run train.py trainer.steps=50000 trainer.learning_rate=5e-5
uv run train.py trainer.aux_losses.doc_access_loss.weight=0.5   # dotted Hydra overrides
```
Config tree composed in `configs/train.yaml` (`model` / `dataset` / `trainer`).
