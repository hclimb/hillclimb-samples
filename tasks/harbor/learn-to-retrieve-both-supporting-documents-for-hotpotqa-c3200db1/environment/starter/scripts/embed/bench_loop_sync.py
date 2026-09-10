"""Isolate bottleneck #2 — the per-step device sync in the training loop.

Uses the SAME synthetic batch as profile_train_step.py, so **data loading (bottleneck #1) is
entirely removed** and we measure only the loop mechanics. Times the REAL jitted
Trainer._train_step under two loop patterns (representative stage-3 config: all-trainable, ce=1,
the 85%-of-steps regime):

  pipelined     dispatch `iters` steps back-to-back, pull NOTHING to host per step,
                block_until_ready once at the end -> device-bound throughput ceiling (JAX async
                dispatch keeps the device busy with no host<->device serialization).
  trainer_sync  replicate trainer.py's per-step host pulls EXACTLY (int(loss_nan), int(grad_nan),
                float(ce_loss), float(aux['total']), float(grad_norm)) each step, as the real loop
                does (trainer.py:383-407, minus wandb) -> the pattern the run uses today.

  gap = trainer_sync - pipelined  ==  wall-clock lost per step to the per-step sync pattern itself.

NOTE (honesty): the real loop ALSO calls wandb.log(...) every step and next(iterator)/
process_train_pairs in the same non-pipelined window; this bench excludes those, so `gap` is a
LOWER BOUND on bottleneck #2. A best-effort wandb.log offline-mode per-call cost is timed
separately (the dominant exposed cost, since it logs every step) — the fix for both is
dispatch-ahead + pull/log every K steps, which is quality-neutral.

Usage (on the box, from repo root):
  uv run python scripts/embed/bench_loop_sync.py            # pipelined vs trainer_sync + wandb cost
  uv run python scripts/embed/bench_loop_sync.py --check    # compose+imports only, no accelerator
"""
import argparse
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp

# Reuse the profiler's config/batch/loss scaffold (DRY — identical shapes & wiring).
from profile_train_step import _compose_cfg, _one_batch, _stage_loss_cfg


def _time_wandb(iters):
    """Best-effort per-call cost of wandb.log in OFFLINE mode with a representative ~15-scalar dict
    (mirrors trainer.py:394-406). Offline (no login/network) is a rough proxy for the real exposed
    cost — labeled as such. Returns (median_ms, note)."""
    try:
        import wandb
        run = wandb.init(project="loop-sync-bench", mode="offline", reinit=True)
    except Exception as e:  # noqa: BLE001
        return None, f"skipped ({type(e).__name__}: {e})"
    times = []
    for i in range(iters):
        d = {f"train/m{j}": float(j) for j in range(12)}
        d["step"] = i
        t0 = time.perf_counter()
        wandb.log(d, step=i)
        times.append(time.perf_counter() - t0)
    try:
        wandb.finish()
    except Exception:  # noqa: BLE001
        pass
    return statistics.median(times) * 1e3, "offline-mode proxy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg = _compose_cfg()
    from models import get_model  # noqa: F401  (validates sys.path; cheap)
    from utils import setup_optimizer, freeze_dict

    if args.check:
        aux_cfg, ce = _stage_loss_cfg(cfg, 3)
        print(f"[CHECK] compose+imports OK  model={cfg.model.name}  dataset={cfg.dataset.name}  "
              f"ce_weight(stage3)={ce}  aux={{k: v.get('weight') for k, v in aux_cfg.items()}}", flush=True)
        return

    from utils import init_jax_distributed; init_jax_distributed()
    model = get_model(cfg.model, cfg.trainer.tp_devices)
    optimizer, model, _ = setup_optimizer(cfg, model)
    inputs, targets, input_masks, loss_masks, ce_enable = _one_batch(cfg)
    aux_cfg, ce_weight_f = _stage_loss_cfg(cfg, 3)
    aux_frozen = freeze_dict(aux_cfg)
    ce_w = jnp.array(float(ce_weight_f))

    from trainer.trainer import Trainer

    w, opt_state = model.weights, optimizer.init(model.weights)

    def step(w, opt_state):
        return Trainer._train_step(
            model.forward, optimizer, w, opt_state,
            inputs, targets, input_masks, loss_masks, ce_w, aux_frozen, ce_enable,
        )

    # Warmup: absorbs the big first-step compile and reaches steady state, sync at the end.
    for _ in range(args.warmup):
        w, opt_state, ce_loss, aux, ln, gn, gnorm = step(w, opt_state)
    jax.block_until_ready((w, opt_state))

    # Arm 1 — pipelined: no per-step host pull; device stays busy; sync once at the end.
    t0 = time.perf_counter()
    for _ in range(args.iters):
        w, opt_state, ce_loss, aux, ln, gn, gnorm = step(w, opt_state)
    jax.block_until_ready((w, opt_state))
    pipelined_ms = (time.perf_counter() - t0) / args.iters * 1e3

    # Arm 2 — trainer_sync: replicate trainer.py's per-step host pulls exactly (minus wandb).
    t0 = time.perf_counter()
    for _ in range(args.iters):
        w, opt_state, ce_loss, aux, ln, gn, gnorm = step(w, opt_state)
        _ = int(ln) + int(gn)                                             # trainer.py:383-384
        _ = float(ce_w) * float(ce_loss) + float(aux.get("total", 0.0))   # trainer.py:387-388
        _ = float(gnorm)                                                  # trainer.py:398
    sync_ms = (time.perf_counter() - t0) / args.iters * 1e3

    wandb_ms, wandb_note = _time_wandb(args.iters)

    gap = sync_ms - pipelined_ms
    gap_pct = 100.0 * gap / pipelined_ms if pipelined_ms else float("nan")
    print("\n\n################ LOOP-SYNC BENCH (median ms/step, synthetic batch, data removed) ################", flush=True)
    print(f"  pipelined (device-bound ceiling) : {pipelined_ms:8.2f} ms/step  ({1000.0/pipelined_ms:.3f} steps/s)", flush=True)
    print(f"  trainer_sync (per-step pulls)    : {sync_ms:8.2f} ms/step  ({1000.0/sync_ms:.3f} steps/s)", flush=True)
    print(f"  gap (bottleneck #2 lower bound)  : {gap:8.2f} ms/step  ({gap_pct:+.2f}%)", flush=True)
    if wandb_ms is not None:
        print(f"  wandb.log per call ({wandb_note}) : {wandb_ms:8.2f} ms  "
              f"(+{100.0*wandb_ms/pipelined_ms:.2f}% if run every step, exposed by the sync)", flush=True)
    else:
        print(f"  wandb.log per call               : {wandb_note}", flush=True)
    print("###############################################################################################\n", flush=True)


if __name__ == "__main__":
    main()
