# Optimizer & Freezing

`utils.py::setup_optimizer` → `setup_optimizer_for_stage(cfg, model, stage, all_stages)`.
Builds the optax chain, the trainable-param mask, and the LR schedule for a stage.

## The optax chain
```
optax.chain(
  clip_by_global_norm(trainer.clip_grad_norm),
  adamw(learning_rate=lr, weight_decay=trainer.weight_decay),
  transforms.freeze(mask),          # omitted when trainable_params == ["all"]
)
```

## Trainable-param mask
`trainable_params` is a list of regex. `tree_map_with_path` builds a boolean mask (True =
**freeze**) over the weight tree: a param is trainable iff its key matches any pattern.
`optax.transforms.freeze` zeroes the *updates* for masked params (gradients are still computed
— hence `grad_norm_main` can be nonzero while main is "frozen"). `["all"]` → no freeze,
everything trains.

## LR schedule (per stage)
Peak LR = stage `learning_rate` or `trainer.learning_rate`. Stage duration is derived from
stage boundaries.
- `lr_schedule: cosine` + `warmup_frac>0` → `warmup_cosine_decay_schedule` (init 1e-8 → peak →
  `min_lr`).
- `cosine` alone → `cosine_decay_schedule` (peak → `min_lr`).
- `warmup_frac>0` only → linear warmup then constant peak.
- neither → constant peak LR.

## LoRA activation
While scanning `trainable_params`, if a matched key contains `a_proj` (a LoRA adapter), the
relevant sub-model's `use_lora` flag is set (`main_model` / `embed_model` / top-level), the
forward `partial` is rebuilt with the new cfg, so the LoRA branch in `qwen3.mlp` goes live.
This is how you train only adapters while the base stays frozen. See LoRA in
[../architecture/conventions.md](../architecture/conventions.md).

## HBM cost of unfreezing the main model — check the chip, not the box

⚠️ **Moments are allocated for FROZEN params too.** `optax.adamw` creates `mu`/`nu` for every leaf
it is handed, and the `freeze(mask)` downstream of it only zeroes updates — it does not prevent
allocation. So **every stage pays ~2× all params (~16 GB/chip here) no matter how little is
trainable**, and LoRA saves no optimizer memory by default. Opt in to
`MEM_MASKED_OPTIMIZER=1` to allocate moments for trainable leaves only — but it **cannot resume
existing checkpoints** (different `opt_state` pytree). See
[the implementation note](../implementations/2026-07-20-optimizer-moment-allocation.md).

**Freezing is what makes the 4B affordable, and the cost of unfreezing is per-chip, not per-box.**
Beyond the always-allocated moments above, a stage whose `trainable_params` matches `main_model`
adds gradients for the whole model. For a 4B:

| term | size |
|---|---|
| params (bf16) | ~8 GB |
| gradients | ~8 GB |
| Adam `mu` + `nu` | ~16 GB |
| **total** | **~32 GB** |

**Trainable weights are promoted to fp32 before the optimizer sees them.** `train.py` calls
`utils.py::promote_trainable_to_fp32(model.weights, cfg.trainer.training_stages)` right after
model init and BEFORE `setup_optimizer` — every leaf matched by the union of all stages'
`trainable_params` regexes is upcast bf16→fp32 (frozen leaves stay bf16, no memory cost there).
This exists because `optax.adamw` with a bf16 param would give bf16 `mu`/`nu` too, and
`w_bf16 + Δw_bf16` rounds the update away whenever `|Δw| < ULP(|w|)` — for weights at
unit-ish magnitude (RMSNorm scales) or even ~0.02 (fresh projection inits) under LR~1e-4, that's
EVERY step, silently. See
[wiki/experiments/2026-08-07-bf16-ulp-freeze-empirical-confirmation.md](../experiments/2026-08-07-bf16-ulp-freeze-empirical-confirmation.md)
for the empirical confirmation (bf16 arm: 6 watched norms bit-exact frozen across 9,218 steps).

⚠️ **This promotion is only as good as what actually reaches the optimizer.** `load_checkpoint`'s
warm-start/resume paths restore `model.weights` from a checkpoint AFTER this promotion has already
run — and until 2026-08-13 the restore helper only re-applied sharding, not dtype, so a leaf saved
as bf16 by an older/different-lineage checkpoint came back bf16 and silently re-trapped exactly the
leaves the promotion exists to protect. Fixed in `utils.py::load_checkpoint`'s
`restore_and_reshard` (casts the restored value to the target leaf's dtype). See
[the implementation note](../implementations/2026-08-13-warmstart-restore-dtype-fix.md). Any
run that warm-started or resumed from a bf16-saved checkpoint before that fix should be treated as
having trained its promoted leaves under the bf16 trap regardless of what `promote_fp32`'s
startup log claimed.

**`mu` is deliberately kept bf16 even for fp32-promoted leaves** — `utils.py` calls
`optax.adamw(..., mu_dtype=jnp.bfloat16)` — while `nu` inherits the (now fp32, for promoted
leaves) param dtype automatically, since only `mu_dtype` is overridden. This asymmetry is
intentional, not an oversight: `mu`'s update `m ← m + 0.1·(g − m)` is unbiased two-sided noise
under bf16 truncation (no systematic drift), but `nu`'s update `v ← v + 0.001·(g² − v)` can only
ratchet UP under bf16 (`g² ≥ 0`, so the delta is one-signed) — a one-way drift that decays the
effective LR `1/√v` with no recovery. `nu` must stay fp32; `mu` in bf16 saves ~1.15 GB/chip with
no such risk. See the inline comment at `utils.py::setup_optimizer_for_stage` (search
`mu in bf16 saves`) for the full derivation.

At the default **`tp_devices: 1` this is REPLICATED on every chip** — nothing is sharded — so the
binding constraint is a single chip's HBM:

| | HBM/chip | unfrozen 4B (~32 GB) |
|---|---|---|
| v6e | ~31 GB | **just over — does not fit** |
| v5p | ~95 GB | fits with room (ran at batch 16 on a v5p-4) |

The v6e margin is *thin*, and the observed failure matches: it died short by ~47 MB, not by
gigabytes. That makes `tp_devices=2` a genuine fix rather than a heavy hammer — sharding two ways
puts the per-chip cost near ~16 GB, well inside 31 GB, while an 8-chip slice still gives 4-way
data parallelism.

**Batch size does not rescue it.** Measured 2026-07-20 on a v6e-8 slice, stage C of
`staged_ground_mainunfreeze`: batch 16 → `RESOURCE_EXHAUSTED: allocate 47.50M, 36.65M free`;
batch 8 → the same allocation with **37.26M free**. Halving the batch bought 0.6 MB, because batch
scales *activations*, not the replicated parameter/optimizer state. A frozen-main stage carries no
optimizer state at all, which is why stages A/B of the same run fit comfortably.

Ways to fit an unfrozen main model, best first: shard with **`tp_devices > 1`** (2 is enough here);
use a **v5p** (~95 GB/chip, no config change); or use **LoRA** (`## LoRA activation` above —
optimizer state only for the adapters). Adding chips at `tp_devices: 1` does **not** help, and
neither does bf16 optimizer state (already the default, see above).

## Notes
- Momentum is carried across stage boundaries by the trainer, not here — see
  [multi-stage-training.md](multi-stage-training.md).
- `weight_decay` and `clip_grad_norm` are global (`configs/trainer/standard.yaml`).
