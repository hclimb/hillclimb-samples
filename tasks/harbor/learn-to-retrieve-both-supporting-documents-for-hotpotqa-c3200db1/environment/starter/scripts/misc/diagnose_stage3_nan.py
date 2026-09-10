"""Pinpoint the stage-3 NaN: WHICH parameter's gradient goes non-finite, and does the top-k mode
change it?

Context (wiki/experiments — the run qa_hard_neg_think_sft4b_...-2026-07-16-17-55-17):
from step 15000 (stage 3, main_model unfrozen) ~93% of steps logged
`loss=0.8, grad_norm=nan -> skipping update`. The LOSS is healthy; only the gradient is NaN, and
`train/grad_norm_main` is the NaN group while `grad_norm_mem` / `grad_norm_embed` stay finite.
Because those group norms are SUBSTRING matches, `grad_norm_mem` already covers
`main_model.layers.14.mem_*` — so the NaN lives in main_model params that are NOT mem_*, i.e. the
4B's own attention/MLP. That contradicts the only functional config delta since the last known-good
run (707a128, April): `mem_approx_topk: true` (exact top_k -> jax.lax.approx_max_k). This script
settles it empirically instead of by reading.

It reuses the real machinery (get_model / get_dataset / setup_optimizer_for_stage /
process_train_pairs / compute_aux_losses) and mirrors Trainer._train_step's loss_fn exactly — it
does NOT instrument trainer.py (see CLAUDE.md).

Run (on a box), once per arm:
    MEM_APPROX_TOPK=1 uv run python scripts/misc/diagnose_stage3_nan.py  <ckpt_gs_path> [n_batches]
    MEM_APPROX_TOPK=0 uv run python scripts/misc/diagnose_stage3_nan.py  <ckpt_gs_path> [n_batches]

Prints, per batch: total_loss, global grad norm, the per-group norms the trainer logs, and — the
point — the FIRST parameters whose grad is non-finite, grouped so the failing subsystem is obvious.
"""
import sys
import os

sys.path.insert(0, os.getcwd())

import jax
import jax.numpy as jnp
import optax
from hydra import compose, initialize_config_dir

from data import get_dataset
from models import get_model
from utils import (load_checkpoint, process_train_pairs, setup_optimizer_for_stage,
                   freeze_dict, unfreeze_dict, setup_gcs_credentials)
from losses.registry import compute_aux_losses

setup_gcs_credentials()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ckpt = sys.argv[1].rstrip("/")
    n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    # resume_from=<dir>/<step> -> setup_checkpointing parses the step off the end
    ckpt_dir, _, step_s = ckpt.rpartition("/")
    step = int(step_s)

    # This file lives at scripts/misc/, i.e. TWO levels below the repo root -> three dirnames.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_dir = os.path.join(root, "configs")
    assert os.path.isdir(cfg_dir), f"configs not found at {cfg_dir}"
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
        cfg = compose(config_name="train", overrides=[
            "model=qwen3_mem_embed",
            "model.main_model.model_id=Qwen/Qwen3-4B",
            "model.memory.mem_top_k=64",
            "dataset=qa_hard_neg_think_sft4b",
            "trainer=staged_telemetry",
            "eval_set@trainer.evals=none",
        ])

    approx = os.environ.get("MEM_APPROX_TOPK", "(unset -> model cfg)")
    print(f"=== ARM: MEM_APPROX_TOPK={approx}  ckpt={ckpt} ===", flush=True)

    from utils import init_jax_distributed; init_jax_distributed()
    model = get_model(cfg.model, cfg.trainer.tp_devices)

    # STAGE 3 = the 4th stage (index 3): trainable = mem + embed + main. This is the config under
    # which the NaN appears; setup_optimizer_for_stage also rebuilds model.forward with a new cfg,
    # which is the ONLY route by which trainable_params can reach the backward, so apply it.
    stages = cfg.trainer.training_stages
    stage3 = stages[3]
    print(f"stage3 trainable_params: {list(stage3['trainable_params'])}", flush=True)
    # setup_optimizer_for_stage prints every trainable layer (~1400 lines) — mute it.
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        optimizer, model, _lr = setup_optimizer_for_stage(cfg, model, stage3, all_stages=stages)
    opt_state = optimizer.init(model.weights)

    # checkpoint_manager=None is fine: load_checkpoint builds its own CheckpointManager whenever
    # resume_from_dir is set. (setup_checkpointing() can't be used here — it calls run_dir_name ->
    # HydraConfig.get(), which only exists under a real @hydra.main run, not under compose().)
    _step, opt_state, _stage = load_checkpoint(
        None, model, opt_state,
        return_stage_idx=True, resume_step=step, resume_from_dir=ckpt_dir)
    print(f"loaded checkpoint step={step} from {ckpt_dir}", flush=True)

    aux_cfg = freeze_dict({k: dict(v) for k, v in cfg.trainer.aux_losses.items()})
    ce_weight = jnp.array(float(stage3.get("ce_weight", 1.0)))
    data = get_dataset(cfg.dataset, model)

    # MUST be inside jax.jit, exactly like Trainer._train_step: models/qwen3.py::load does
    # jax.set_mesh(mesh) with axis_types=(Explicit, Explicit), and the weights carry that
    # sharding. Calling value_and_grad eagerly runs under an Auto-axis context and dies with
    # "context mesh ... should match the aval mesh ...". `forward` and `aux_cfg` are static
    # (hashable) args, mirroring the trainer's static_argnames.
    from functools import partial as _partial

    @_partial(jax.jit, static_argnames=("forward", "aux_cfg_frozen"))
    def grad_step(forward, aux_cfg_frozen, w, inputs, targets, input_masks, loss_masks, ce_enable):
        def loss_fn(w):
            pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
            out = forward(inputs, w, pad_mask=pad_mask, collect_aux=True)
            ce = optax.softmax_cross_entropy(out.logits, jax.nn.one_hot(targets, out.logits.shape[-1]))
            ce_mask = loss_masks if ce_enable is None else loss_masks * ce_enable[:, None]
            main_loss = (ce * ce_mask).sum() / (ce_mask.sum() + 1e-9)
            aux = compute_aux_losses(out.aux, loss_masks, input_masks, inputs,
                                     unfreeze_dict(aux_cfg_frozen))
            return main_loss * ce_weight + aux["total"], (main_loss, aux)
        return jax.value_and_grad(loss_fn, has_aux=True)(w)

    def group_norm(grads, grp):
        leaves = [v for k, v in grads.items() if grp in k]
        return float(optax.global_norm(leaves)) if leaves else 0.0

    for i, (tokens, masks) in enumerate(data.generator()):
        if i >= n_batches:
            break
        inputs, targets, input_masks, loss_masks, ce_enable = process_train_pairs(tokens, masks)
        (total, (main_loss, aux)), grads = grad_step(
            model.forward, aux_cfg, model.weights, inputs, targets, input_masks,
            loss_masks, ce_enable)
        gn = optax.global_norm(grads)
        print(f"\n--- batch {i}: total_loss={float(total):.4f} ce={float(main_loss):.4f} "
              f"grad_norm={float(gn)} ---", flush=True)
        print(f"    grad_norm_mem  ={group_norm(grads, 'mem_'):.4f}   "
              f"grad_norm_embed={group_norm(grads, 'embed_model'):.4f}   "
              f"grad_norm_main ={group_norm(grads, 'main_model'):.4f}", flush=True)

        # THE POINT: which individual params are non-finite?
        bad = [k for k, v in grads.items() if not bool(jnp.isfinite(v).all())]
        print(f"    non-finite grads: {len(bad)} / {len(grads)} params", flush=True)
        if bad:
            # Bucket by subsystem so the failing component is obvious at a glance.
            def bucket(k):
                if "mem_" in k and "main_model" in k: return "main_model.*mem_*  (memory proj in the 4B)"
                if "mem_" in k and "embed_model" in k: return "embed_model.*mem_*"
                if "mem_" in k: return "other mem_*"
                if "main_model" in k: return "main_model (pure 4B: attn/mlp/embed/lm_head)"
                if "embed_model" in k: return "embed_model (pure)"
                return "other"
            from collections import Counter
            for b, n in Counter(bucket(k) for k in bad).most_common():
                print(f"        {n:5d}  {b}", flush=True)
            print(f"    first 8 non-finite: {bad[:8]}", flush=True)
            # Is the aux/CE split implicated? Both are finite in the log, but confirm here.
            print(f"    aux losses: " + ", ".join(
                f"{k}={float(v):.4g}" for k, v in list(aux['losses'].items())[:6]), flush=True)
        else:
            print("    ALL GRADS FINITE for this batch", flush=True)


if __name__ == "__main__":
    main()
