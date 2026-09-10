"""Real end-to-end training throughput — offline-parquet data + real Trainer._train_step.

Unlike the synthetic-batch benches, this uses the REAL dataloader (get_dataset) so it captures
data-fetch + process_train_pairs + the loop sync together — the ACTUAL steps/sec a run gets. It
confirms two fixes with real steps:
  - Bottleneck #1 (data): offline-parquet keeps the grain buffer full (data-fetch ~0 ms).
  - Bottleneck #2 (loop): 'pipelined' (float() every K steps) vs 'baseline' (float() every step).

Two arms over ONE continuous weight stream (step TIME is value-independent, so both arms share the
single ~230 s compile and one evolving `w`, no per-arm weight copy / re-compile):
  baseline   float(ce_loss) every step, as trainer.py:387 does today.
  pipelined  float() only every --sync-k steps -> JAX dispatches ahead, device stays busy.

Requires HF_HUB_OFFLINE=1 + GROUND_HF_PARQUET (offline parquet) in the env — set by the .sh; live-HF
429-stalls (see bench_data_throughput). --stage picks the freeze/ce regime (0 = warmup/frozen-main;
a few-hundred-step run is entirely stage 0).

Usage (on the box, from repo root):
  HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=~/hf_parquet uv run python scripts/embed/bench_train_throughput.py --steps 200 --sync-k 20
  uv run python scripts/embed/bench_train_throughput.py --check   # compose only, no TPU/data
"""
import argparse
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np

from profile_train_step import _compose_cfg, _stage_loss_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sync-k", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    print(f"[train-thru] HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
          f"GROUND_HF_PARQUET={os.environ.get('GROUND_HF_PARQUET')}", flush=True)
    cfg = _compose_cfg()
    from models import get_model
    from utils import setup_optimizer_for_stage, process_train_pairs, freeze_dict
    from data import get_dataset
    from trainer.trainer import Trainer

    stages = list(cfg.trainer.get("training_stages") or [])
    stage = stages[args.stage]
    aux_cfg, ce_weight = _stage_loss_cfg(cfg, args.stage)

    if args.check:
        print(f"[CHECK] ok  stage={args.stage}  ce_weight={ce_weight}  "
              f"trainable={list(stage['trainable_params'])}  steps={args.steps} sync_k={args.sync_k}", flush=True)
        return

    from utils import init_jax_distributed; init_jax_distributed()
    model = get_model(cfg.model, cfg.trainer.tp_devices)
    optimizer, model, _ = setup_optimizer_for_stage(cfg, model, stage_config=stage, all_stages=stages)
    forward = model.forward
    dataset = get_dataset(cfg.dataset, model)
    aux_frozen = freeze_dict(aux_cfg)
    ce_w = jnp.array(float(ce_weight))
    w = model.weights
    opt_state = optimizer.init(model.weights)
    gen = dataset.generator()

    def one_step(w, opt_state, tk, mk):
        inputs, targets, input_masks, loss_masks, ce_enable = process_train_pairs(tk, mk)
        return Trainer._train_step(forward, optimizer, w, opt_state,
                                   inputs, targets, input_masks, loss_masks, ce_w, aux_frozen, ce_enable)

    # Warmup absorbs the ~230 s compile + the 34 s 100k-shuffle-fill data startup + reaches steady state.
    print(f"[train-thru] warmup {args.warmup} steps (compile + data startup)...", flush=True)
    t_ws = time.perf_counter()
    for i in range(args.warmup):
        tk, mk = next(gen)
        w, opt_state, ce, aux, ln, gn, gnorm = one_step(w, opt_state, tk, mk)
        float(ce)
        if i == 0:
            print(f"[train-thru] first step done (compile+startup) in {time.perf_counter()-t_ws:.1f}s", flush=True)
    jax.block_until_ready((w, opt_state))
    print(f"[train-thru] warmup done in {time.perf_counter()-t_ws:.1f}s; timing {args.steps} steps/arm", flush=True)

    def run(sync_every, n, label):
        nonlocal w, opt_state
        dts = []
        t0 = time.perf_counter()
        for i in range(n):
            td = time.perf_counter()
            tk, mk = next(gen)
            dts.append(time.perf_counter() - td)
            w, opt_state, ce, aux, ln, gn, gnorm = one_step(w, opt_state, tk, mk)
            if (i + 1) % sync_every == 0:
                float(ce)
        jax.block_until_ready((w, opt_state))
        tot = time.perf_counter() - t0
        d = np.array(dts)
        msps = tot / n * 1e3
        print(f"[{label}] {n} steps {tot:.2f}s -> {msps:.2f} ms/step ({1000/msps:.3f} steps/s) | "
              f"data-fetch median/p95/max {np.median(d)*1e3:.2f}/{np.percentile(d,95)*1e3:.2f}/{d.max()*1e3:.2f} ms",
              flush=True)
        return msps, d

    base, dbase = run(1, args.steps, "baseline sync-each")
    pipe, dpipe = run(args.sync_k, args.steps, f"pipelined sync-{args.sync_k}")
    gain = (base - pipe) / base * 100

    print(f"\n\n################ REAL TRAIN THROUGHPUT (offline-parquet, stage {args.stage}) ################", flush=True)
    print(f"  baseline (sync each step)   : {base:8.2f} ms/step  ({1000/base:.3f} steps/s)", flush=True)
    print(f"  pipelined (sync every {args.sync_k:>2})    : {pipe:8.2f} ms/step  ({1000/pipe:.3f} steps/s)", flush=True)
    print(f"  loop-sync fix REAL gain     : {gain:+.2f}%   (synthetic bench predicted +6.1%)", flush=True)
    print(f"  data-fetch (both arms) max  : {max(dbase.max(), dpipe.max())*1e3:.1f} ms  "
          f"(offline-parquet keeps up => data off critical path)", flush=True)
    print("#############################################################################\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("!!!! bench_train_throughput FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
