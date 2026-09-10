"""Stage-aware, quality-neutral speed profiler for the qa_hard_neg_think_sft4b train step.

Follow-on to bench_approx_topk.py. approx-topk removed the exact-sort (~40% of the OLD step);
this profiler attributes what's LEFT of the ~500ms step so we can target the next
quality-neutral win. It measures, per training stage (the freeze mask differs by stage — that
is the whole point), the wall-clock of three variants of the REAL train step and derives a
fwd / backward / recoverable-backward split:

  full       the real jitted step (value_and_grad + optimizer update), current behavior.
  fwd_only   the loss forward with NO grad  ->  forward cost.
  sg_frozen  AXIS A: stop_gradient the FROZEN weights inside loss_fn before forward, so XLA
             prunes their backward. Provably quality-neutral (a detached frozen weight gets 0
             grad; optax.freeze zeroed its UPDATE anyway -> identical trainable-param updates).

Derived per stage:
  backward_total       = full - fwd_only                 (all backward the step pays now)
  backward_recoverable = full - sg_frozen                (AXIS A prize this stage: frozen bwd)
  backward_necessary   = sg_frozen - fwd_only            (bwd that must run even under Axis A)
  bwd/fwd ratio        = backward_necessary / fwd_only    (>~2 flags heavy jax.remat recompute
                          in the still-needed backward -> the REMAT-POLICY axis, which unlike
                          Axis A helps stage 3 / all 100% of steps).

Whole-run weighting: the staged schedule spends 5k/5k/5k/85k steps in stages 0/1/2/3, so Axis A
(nonzero only in 0-2) is bounded to ~15% of steps. The summary projects the per-stage prizes onto
those weights so the whole-run wall-clock benefit is not overstated.

Mirrors train.py wiring (get_model -> setup_optimizer_for_stage -> get_dataset shapes) and
replicates Trainer._train_step's loss_fn byte-for-byte (same compute_aux_losses, same CE gate),
adding only the sg toggle. Does NOT modify the trainer/model. Synthetic batch of the true shapes
(step time is shape-driven; building the real bank hammers the HF API — see bench_approx_topk.py).

Usage (on the box, from repo root):
  uv run python scripts/embed/profile_train_step.py                 # stages 0,1,2,3 (slow compile)
  uv run python scripts/embed/profile_train_step.py --stages 0 3    # quick: biggest-prize + control
  uv run python scripts/embed/profile_train_step.py --check         # compose+imports only, no TPU
"""
import argparse
import os
import re
import statistics
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")
# scripts/embed/ (not the repo root) is sys.path[0]; add the root so repo modules import as
# they do for train.py (identical rationale to bench_approx_topk.py).
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from hydra import compose, initialize_config_dir

# Exactly the qa_hard_neg_think_sft4b run's model/data, minus wandb/eval/checkpoint machinery.
_OVERRIDES = [
    "model=qwen3_mem_embed",
    "model.main_model.model_id=Qwen/Qwen3-4B",
    "model.memory.mem_top_k=64",
    "dataset=qa_hard_neg_think_sft4b",
    "trainer.use_wandb=false",
    "eval_set@trainer.evals=none",   # empty eval set; we never build the Trainer (see bench_approx_topk.py)
    "trainer.eval_interval=0",
]

# Staged schedule step budget per stage (max_step deltas for steps=100000): 0-5k,5k-10k,10k-15k,15k-100k.
_STAGE_STEPS = [5000, 5000, 5000, 85000]


def _compose_cfg():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base="1.2"):
        return compose(config_name="train", overrides=_OVERRIDES)


def _stage_loss_cfg(cfg, stage_idx):
    """(aux_loss_config dict, ce_weight float) for a stage — base aux_losses overlaid with the
    stage's aux_losses + ce_weight, matching what the Trainer freezes into JIT for that stage."""
    from utils import freeze_dict, unfreeze_dict

    base = {k: dict(v) for k, v in cfg.trainer.aux_losses.items()}
    stages = cfg.trainer.get("training_stages") or []
    stage = stages[stage_idx] if stage_idx < len(stages) else {}
    for loss_name, ov in dict(stage.get("aux_losses", {})).items():
        base.setdefault(loss_name, {})
        base[loss_name].update(dict(ov))
    ce_weight = float(stage.get("ce_weight", cfg.trainer.get("ce_weight", 1.0)))
    # round-trip through freeze/unfreeze so the dict is the same hashable-friendly plain dict the
    # trainer hands compute_aux_losses.
    return unfreeze_dict(freeze_dict(base)), ce_weight


def _frozen_mask(weights, trainable_patterns):
    """Bool pytree, True where FROZEN — replicates setup_optimizer_for_stage.map_fn's decision
    (regex over the top-level dotted weight key). True => stop_gradient candidate for Axis A."""
    pats = [re.compile(p) for p in trainable_patterns]

    def is_frozen(path, _leaf):
        key = path[0].key if hasattr(path[0], "key") else str(path[0])
        return not any(p.search(key) for p in pats)

    return jax.tree_util.tree_map_with_path(is_frozen, weights)


def _one_batch(cfg):
    """Synthetic batch with the EXACT shapes/dtypes data/qa.py yields for this config, through the
    real process_train_pairs. Shapes only drive step time (see bench_approx_topk.py::_one_batch)."""
    from utils import process_train_pairs

    B = int(cfg.dataset.batch_size)
    seq_len = int(cfg.dataset.seq_len)
    M = int(cfg.dataset.num_chunks_per_doc)
    d_seq = int(cfg.dataset.doc_chunk_seq_len)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    vocab = 1000
    tokens = {
        "batch": jax.random.randint(k1, (B, seq_len), 1, vocab, dtype=jnp.int32),
        "docs": jax.random.randint(k2, (B * M, d_seq), 1, vocab, dtype=jnp.int32),
    }
    masks = {
        "batch_mask": jnp.ones((B, seq_len), dtype=jnp.int32),
        "docs_mask": jnp.ones((B * M, d_seq), dtype=jnp.float32),
        "loss_mask": jnp.ones((B, seq_len), dtype=jnp.float32),
        "pos_doc_mask": jnp.ones((B, M), dtype=jnp.int32),
        "ce_enable": jnp.ones((B,), dtype=jnp.float32),
    }
    return process_train_pairs(tokens, masks)


def _make_loss_fn(forward, batch, frozen_mask, ce_weight, aux_cfg, sg):
    """Replica of Trainer._train_step.loss_fn (losses/registry.compute_aux_losses, per-row CE gate),
    plus the Axis-A sg toggle: when sg=True, frozen weights are stop_gradient'd before forward."""
    import optax
    from losses import compute_aux_losses

    inputs, targets, input_masks, loss_masks, ce_enable = batch
    collect_aux = aux_cfg is not None and len(aux_cfg) > 0

    def loss_fn(w):
        if sg:
            w = jax.tree_util.tree_map(
                lambda leaf, frz: jax.lax.stop_gradient(leaf) if frz else leaf, w, frozen_mask)
        pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
        output = forward(inputs, w, pad_mask=pad_mask, collect_aux=collect_aux)
        one_hot = jax.nn.one_hot(targets, output.logits.shape[-1])
        ce_loss = optax.softmax_cross_entropy(output.logits, one_hot)
        ce_mask = loss_masks if ce_enable is None else loss_masks * ce_enable[:, None]
        main_loss = (ce_loss * ce_mask).sum() / (ce_mask.sum() + 1e-9)
        aux_result = compute_aux_losses(output.aux, loss_masks, input_masks, inputs, aux_cfg)
        total = main_loss * ce_weight + aux_result["total"]
        return total, main_loss

    return loss_fn


def _time(fn, w, opt_state, iters, warmup, label):
    """Median wall-clock of fn(w, opt_state) with a device sync each iter (no donation, so w is
    reused unchanged every iter). Returns median ms."""
    times = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        out = fn(w, opt_state)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        print(f"    {label} step {i:>2}{' (warmup)' if i < warmup else ''} {dt*1e3:8.2f} ms", flush=True)
    return statistics.median(times) * 1e3


def profile_stage(cfg, model, stage_idx, batch, iters, warmup):
    import optax
    from utils import setup_optimizer_for_stage

    stages = list(cfg.trainer.get("training_stages") or [])
    stage = stages[stage_idx]
    patterns = list(stage["trainable_params"])
    optimizer, model, _ = setup_optimizer_for_stage(cfg, model, stage_config=stage, all_stages=stages)
    forward = model.forward
    w = model.weights
    opt_state = optimizer.init(w)
    frozen_mask = _frozen_mask(w, patterns)
    aux_cfg, ce_weight = _stage_loss_cfg(cfg, stage_idx)

    n_frozen = sum(int(f) for f in jax.tree_util.tree_leaves(frozen_mask))
    n_total = len(jax.tree_util.tree_leaves(frozen_mask))
    print(f"\n=== STAGE {stage_idx}  trainable={patterns}  ce_weight={ce_weight}  "
          f"frozen {n_frozen}/{n_total} weight tensors ===", flush=True)

    def make_step(sg):
        loss_fn = _make_loss_fn(forward, batch, frozen_mask, ce_weight, aux_cfg, sg)

        @jax.jit
        def step(w, opt_state):
            (_, main_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(w)
            updates, new_s = optimizer.update(grads, opt_state, w)
            new_w = optax.apply_updates(w, updates)
            return new_w, new_s, main_loss
        return step

    loss_full = _make_loss_fn(forward, batch, frozen_mask, ce_weight, aux_cfg, sg=False)
    fwd = jax.jit(lambda w, _os: loss_fn_forward(loss_full, w))

    full_ms = _time(make_step(False), w, opt_state, iters, warmup, "full ")
    fwd_ms = _time(fwd, w, opt_state, iters, warmup, "fwd  ")
    if n_frozen == 0:
        # Nothing frozen (e.g. stage 3): stop_gradient(frozen)==identity, so sg_frozen is
        # byte-identical to full — skip the redundant compile, Axis-A prize is 0 by definition.
        print("    sg    skipped (0 frozen tensors -> sg_frozen == full)", flush=True)
        sg_ms = full_ms
    else:
        sg_ms = _time(make_step(True), w, opt_state, iters, warmup, "sg   ")

    bwd_total = full_ms - fwd_ms
    bwd_recoverable = full_ms - sg_ms
    bwd_necessary = sg_ms - fwd_ms
    ratio = bwd_necessary / fwd_ms if fwd_ms > 0 else float("nan")
    return {
        "stage": stage_idx, "full": full_ms, "fwd": fwd_ms, "sg": sg_ms,
        "bwd_total": bwd_total, "bwd_recoverable": bwd_recoverable,
        "bwd_necessary": bwd_necessary, "bwd_over_fwd": ratio,
    }


def loss_fn_forward(loss_fn, w):
    total, _ = loss_fn(w)
    return total


def _print_summary(rows):
    print("\n\n################ SUMMARY (median ms per step) ################", flush=True)
    hdr = f"{'stage':>5} {'full':>9} {'fwd':>9} {'sg_frozen':>10} {'bwd_tot':>9} {'AxisA_recov':>12} {'bwd_needed':>11} {'bwd/fwd':>8}"
    print(hdr, flush=True)
    for r in rows:
        print(f"{r['stage']:>5} {r['full']:>9.2f} {r['fwd']:>9.2f} {r['sg']:>10.2f} "
              f"{r['bwd_total']:>9.2f} {r['bwd_recoverable']:>12.2f} {r['bwd_necessary']:>11.2f} "
              f"{r['bwd_over_fwd']:>8.2f}", flush=True)

    # Whole-run projection: weight each measured stage by its step budget. Stages not measured are
    # skipped and flagged so the projection isn't silently under-counted.
    by_stage = {r["stage"]: r for r in rows}
    tot_steps = sum(_STAGE_STEPS)
    base_time = axisa_time = covered = 0.0
    missing = []
    for s, n in enumerate(_STAGE_STEPS):
        r = by_stage.get(s)
        if r is None:
            missing.append(s)
            continue
        covered += n
        base_time += r["full"] * n
        axisa_time += r["sg"] * n
    if base_time > 0:
        print(f"\nWhole-run projection over measured stages "
              f"({int(covered)}/{tot_steps} steps"
              + (f"; MISSING stages {missing} — projection partial" if missing else "") + "):", flush=True)
        saved = 1.0 - axisa_time / base_time
        print(f"  Axis A (frozen->stop_gradient) whole-run wall-clock saving: {saved*100:5.2f}%", flush=True)
        print(f"  (bounded by the {sum(_STAGE_STEPS[:3])/tot_steps*100:.0f}% of steps in frozen stages 0-2; "
              f"stage 3 is unaffected — see bwd/fwd there for the remat-policy axis)", flush=True)
    print("################################################################\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true", help="compose+imports only; no TPU/model")
    args = ap.parse_args()

    cfg = _compose_cfg()
    from models import get_model  # noqa: F401  (validates sys.path fix; cheap)

    if args.check:
        stages = list(cfg.trainer.get("training_stages") or [])
        print(f"[CHECK] compose+imports OK  model={cfg.model.name}  dataset={cfg.dataset.name}  "
              f"n_stages={len(stages)}  steps={cfg.trainer.steps}  stages_req={args.stages}", flush=True)
        for i in args.stages:
            aux_cfg, ce = _stage_loss_cfg(cfg, i)
            print(f"  stage {i}: ce_weight={ce}  trainable={list(stages[i]['trainable_params'])}  "
                  f"aux={ {k: v.get('weight') for k, v in aux_cfg.items()} }", flush=True)
        return

    from utils import init_jax_distributed; init_jax_distributed()
    model = get_model(cfg.model, cfg.trainer.tp_devices)
    batch = _one_batch(cfg)

    rows = []
    for s in args.stages:
        rows.append(profile_stage(cfg, model, s, batch, args.iters, args.warmup))
    _print_summary(rows)


if __name__ == "__main__":
    main()
