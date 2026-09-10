"""Diagnose the capstone's grad_norm=nan: which param group is non-finite on REAL data (stage 0)?

The capstone recipe verify (train.py, offline data) skipped every update because global grad_norm was
nan — but the synthetic-batch benches had finite grads. So real-data edge cases (padding / empty-doc
rows / missing positives in doc_access_loss) trigger it. This localizes it: compute the grad on a few
REAL offline batches and report, per group (mem / embed_model / embed_proj_conv / main_model), the
grad norm + whether it is finite, plus the TRAINABLE-only norm (what actually gets applied). Tells us
if the nan is in the FROZEN main model (guard over-conservative — it norms ALL grads incl. frozen) or
in the TRAINABLE params (a real loss/data bug). Read-only; no training, no fix.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import optax

from profile_train_step import _compose_cfg, _stage_loss_cfg, _frozen_mask, _make_loss_fn


def _grp_report(g, label):
    import re
    groups = {"mem_": None, "embed_proj_conv": None, "embed_model": None, "main_model": None}
    for grp in groups:
        leaves = [v for k, v in g.items() if grp in k and getattr(v, "size", 0) > 0]
        if not leaves:
            continue
        norm = float(optax.global_norm(leaves))
        anynan = any(bool(jnp.any(~jnp.isfinite(v))) for v in leaves)
        n_nan_leaves = sum(int(bool(jnp.any(~jnp.isfinite(v)))) for v in leaves)
        groups[grp] = (norm, anynan, n_nan_leaves, len(leaves))
        print(f"    [{label}] {grp:16s} norm={norm:.3e}  finite={not anynan}  ({n_nan_leaves}/{len(leaves)} leaves non-finite)", flush=True)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--stage", type=int, default=0)
    args = ap.parse_args()

    cfg = _compose_cfg()
    from models import get_model
    from utils import process_train_pairs, freeze_dict  # noqa: F401
    from data import get_dataset

    stages = list(cfg.trainer.get("training_stages") or [])
    stage = stages[args.stage]
    patterns = tuple(stage["trainable_params"])
    aux_cfg, ce_weight = _stage_loss_cfg(cfg, args.stage)

    from utils import init_jax_distributed; init_jax_distributed()
    model = get_model(cfg.model, cfg.trainer.tp_devices)
    dataset = get_dataset(cfg.dataset, model)
    frozen_mask = _frozen_mask(model.weights, patterns)
    ce_w = jnp.array(float(ce_weight))
    import re
    pats = [re.compile(p) for p in patterns]
    is_trainable = lambda k: any(p.search(k) for p in pats)

    gen = dataset.generator()
    grad_fn = jax.jit(jax.grad(lambda w, b: _make_loss_fn(model.forward, b, frozen_mask, ce_w, aux_cfg, sg=False)(w)[0]))

    print(f"\n[grad-finite] stage {args.stage}, trainable={patterns}", flush=True)
    for i in range(args.batches):
        tokens, masks = next(gen)
        batch = process_train_pairs(tokens, masks)
        g = grad_fn(model.weights, batch)
        total = float(optax.global_norm([v for v in g.values() if getattr(v, "size", 0) > 0]))
        trainable_norm = float(optax.global_norm([v for k, v in g.items() if is_trainable(k) and getattr(v, "size", 0) > 0]))
        print(f"\n  --- batch {i}: total grad_norm={total:.3e} (finite={jnp.isfinite(jnp.array(total))})  "
              f"TRAINABLE-only grad_norm={trainable_norm:.3e} (finite={jnp.isfinite(jnp.array(trainable_norm))}) ---", flush=True)
        _grp_report(g, f"b{i}")
    print("\n[grad-finite] VERDICT: if TRAINABLE-only norm is finite but total is nan -> the nan is in the "
          "FROZEN main model and the guard (global_norm over ALL grads) is over-conservative, skipping "
          "good trainable updates. If TRAINABLE-only is also nan -> a real loss/data bug in mem/embed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("!!!! bench_grad_finite FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(0)
