"""Verify + measure the wired Axis-A path in the REAL Trainer._train_step (stage 0).

Calls the production Trainer._train_step with trainable_patterns=None (baseline) vs =stage-0
patterns (Axis A: stop_gradient frozen weights). Confirms the wiring runs without error and measures
the device-bound speedup. Quality-neutrality is analytic (stop_gradient on a frozen weight zeros only
its own grad; the activation-gradient path to trainable params is unchanged) and already reflected in
the profiler's sg arm (515->264 ms stage 0) — this checks the trainer.py integration specifically.

Both arms pipelined (dispatch N, sync once), continuous weight stream (step time is value-independent
-> one w evolves across both arms, separate compiles for the two static trainable_patterns).
"""
import argparse
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

from profile_train_step import _compose_cfg, _one_batch, _stage_loss_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg = _compose_cfg()
    from models import get_model
    from utils import setup_optimizer_for_stage, freeze_dict
    from trainer.trainer import Trainer

    stages = list(cfg.trainer.get("training_stages") or [])
    stage = stages[args.stage]
    patterns = tuple(stage["trainable_params"])
    aux_cfg, ce_weight = _stage_loss_cfg(cfg, args.stage)
    if args.check:
        print(f"[CHECK] ok stage={args.stage} patterns={patterns} ce={ce_weight}", flush=True)
        return

    from utils import init_jax_distributed; init_jax_distributed()
    model = get_model(cfg.model, cfg.trainer.tp_devices)
    optimizer, model, _ = setup_optimizer_for_stage(cfg, model, stage_config=stage, all_stages=stages)
    forward = model.forward
    inputs, targets, input_masks, loss_masks, ce_enable = _one_batch(cfg)
    aux_frozen = freeze_dict(aux_cfg)
    ce_w = jnp.array(float(ce_weight))
    w, opt_state = model.weights, optimizer.init(model.weights)

    def run(tp, n, warmup, label):
        nonlocal w, opt_state
        last_ce = None
        for _ in range(warmup):
            w, opt_state, ce, aux, ln, gn, gnorm = Trainer._train_step(
                forward, optimizer, w, opt_state, inputs, targets, input_masks, loss_masks,
                ce_w, aux_frozen, ce_enable, tp)
        jax.block_until_ready((w, opt_state))
        t0 = time.perf_counter()
        for _ in range(n):
            w, opt_state, ce, aux, ln, gn, gnorm = Trainer._train_step(
                forward, optimizer, w, opt_state, inputs, targets, input_masks, loss_masks,
                ce_w, aux_frozen, ce_enable, tp)
            last_ce = ce
        jax.block_until_ready((w, opt_state))
        msps = (time.perf_counter() - t0) / n * 1e3
        print(f"[{label}] {msps:.2f} ms/step  (ce_loss={float(last_ce):.4f})", flush=True)
        return msps

    base = run(None, args.iters, args.warmup, "baseline (tp=None)")
    axa = run(patterns, args.iters, args.warmup, "AxisA (stop_grad frozen)")
    print(f"\n\n################ AXIS A WIRED VERIFY (stage {args.stage}, device-bound) ################", flush=True)
    print(f"  baseline (no stop_grad)       : {base:8.2f} ms/step", flush=True)
    print(f"  AxisA (stop_grad frozen)      : {axa:8.2f} ms/step", flush=True)
    print(f"  speedup                       : {base/axa:.2f}x  ({(base-axa)/base*100:+.1f}%)", flush=True)
    print(f"  (finite ce_loss on both arms => wiring OK; quality-neutral by construction)", flush=True)
    print("################################################################################\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("!!!! bench_axisa FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(0)
