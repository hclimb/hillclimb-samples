"""Optimizer-update cost (lever F bound) — device-bound step decomposition, stage 3.

Splits the ~480 ms device-bound step (stage 3, all-trainable — the 85%-of-steps regime where the Adam
update touches all 4.6B params) into compute vs optimizer:

  grad_only  jax.grad(loss_fn) only, forced by global_norm (no optimizer update) -> fwd+bwd floor.
  full       jax.grad + optax.adamw update + apply (donated, like _train_step)   -> full step.

  optimizer-update cost = full - grad_only

Bounds every optimizer-side lever (bf16/8-bit Adam state, fused update): if the update is a small
slice, F isn't worth the quality A/B; if large, it's worth pursuing. Synthetic batch (shape-driven);
both arms pipelined (dispatch N, sync once) so they're device-bound. Uses the model's real (sharded)
weights, so opt_state is FSDP-sharded and fits — same as real stage-3 training.
"""
import argparse
import functools
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import optax

from profile_train_step import _compose_cfg, _one_batch, _stage_loss_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg = _compose_cfg()
    from models import get_model
    from utils import setup_optimizer_for_stage, freeze_dict
    from losses import compute_aux_losses

    stages = list(cfg.trainer.get("training_stages") or [])
    stage = stages[args.stage]
    aux_cfg, ce_weight = _stage_loss_cfg(cfg, args.stage)
    if args.check:
        print(f"[CHECK] ok stage={args.stage} ce={ce_weight} trainable={list(stage['trainable_params'])}", flush=True)
        return

    from utils import init_jax_distributed; init_jax_distributed()
    model = get_model(cfg.model, cfg.trainer.tp_devices)
    optimizer, model, _ = setup_optimizer_for_stage(cfg, model, stage_config=stage, all_stages=stages)
    forward = model.forward
    inputs, targets, input_masks, loss_masks, ce_enable = _one_batch(cfg)
    aux_cfg_d = dict(aux_cfg)
    ce_w = jnp.array(float(ce_weight))

    def loss_fn(w):
        pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
        out = forward(inputs, w, pad_mask=pad_mask, collect_aux=True)
        one_hot = jax.nn.one_hot(targets, out.logits.shape[-1])
        ce_loss = optax.softmax_cross_entropy(out.logits, one_hot)
        ce_mask = loss_masks * ce_enable[:, None]
        main_loss = (ce_loss * ce_mask).sum() / (ce_mask.sum() + 1e-9)
        aux = compute_aux_losses(out.aux, loss_masks, input_masks, inputs, aux_cfg_d)
        return main_loss * ce_w + aux["total"]

    # grad_only: output is the grad global-norm, so XLA can't DCE the backward. w constant each iter.
    @jax.jit
    def grad_only(w):
        return optax.global_norm(jax.grad(loss_fn)(w))

    # full: grad + adamw update + apply, donated like Trainer._train_step (constant HBM footprint).
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def full(w, opt_state):
        g = jax.grad(loss_fn)(w)
        updates, new_s = optimizer.update(g, opt_state, w)
        return optax.apply_updates(w, updates), new_s

    def time_grad(iters, warmup):
        w = model.weights
        for _ in range(warmup):
            gn = grad_only(w)
        jax.block_until_ready(gn)
        t0 = time.perf_counter()
        for _ in range(iters):
            gn = grad_only(w)
        jax.block_until_ready(gn)
        return (time.perf_counter() - t0) / iters * 1e3

    def time_full(iters, warmup):
        w, opt_state = model.weights, optimizer.init(model.weights)
        for _ in range(warmup):
            w, opt_state = full(w, opt_state)
        jax.block_until_ready((w, opt_state))
        t0 = time.perf_counter()
        for _ in range(iters):
            w, opt_state = full(w, opt_state)
        jax.block_until_ready((w, opt_state))
        return (time.perf_counter() - t0) / iters * 1e3

    g_ms = time_grad(args.iters, args.warmup)
    f_ms = time_full(args.iters, args.warmup)
    optc = f_ms - g_ms
    print(f"\n\n################ OPTIMIZER-UPDATE COST (stage {args.stage}, device-bound ms/step) ################", flush=True)
    print(f"  grad_only (fwd+bwd compute floor) : {g_ms:8.2f} ms/step", flush=True)
    print(f"  full (grad + adamw update + apply): {f_ms:8.2f} ms/step", flush=True)
    print(f"  -> optimizer-update cost          : {optc:8.2f} ms/step  ({optc/f_ms*100:.1f}% of step)", flush=True)
    print(f"     (bounds lever F: bf16/8-bit Adam state can reclaim at most a fraction of this)", flush=True)
    print("###################################################################################\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("!!!! bench_optcost FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(0)
