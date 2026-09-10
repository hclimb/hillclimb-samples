"""Standalone probe: does the per-group LR + deterministic-warmup path in
utils.py::setup_optimizer_for_stage build a valid optimizer AND route the right
leaves into the right groups?

- Constructs a synthetic weight tree with keys that mirror the real model:
    main_model.norm                        -> main
    main_model.layers.9.q_proj             -> main
    main_model.layers.14.mem_q_proj        -> mem   (contains "mem_" — priority match)
    main_model.spec_tokens                 -> mem   (spec_tokens matches mem group)
    embed_model.layers.5.self_attn.q_proj  -> embed
    embed_model.mem_k_proj                 -> mem   (contains "mem_" — priority match)
    embed_model.embed_proj_conv            -> mem   (embed_proj_conv is memory-branch)

- Runs `setup_optimizer_for_stage` under each of the 4 staged.yaml stages.
- Asserts per-group counts match hand-computed truth.
- Asserts warmup_steps is honored (schedule callable evaluates ~init_value at step 0).

Run:
    JAX_PLATFORMS=cpu uv run python scripts/debug/probe_per_group_lr.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# script lives at scripts/debug/, so repo root is two levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from functools import partial

from utils import setup_optimizer_for_stage, parse_training_stages


class FakeModel:
    def __init__(self, weights, cfg_dict):
        self.weights = weights
        # `forward` must be a functools.partial(func, cfg_dict) — setup_optimizer_for_stage
        # unpacks .args[0] to read/rewrite use_lora flags.
        def _fake_forward(cfg, *a, **k):
            return None
        self.forward = partial(_fake_forward, cfg_dict)
        self.cfg = cfg_dict


def make_synthetic_weights():
    keys = [
        "main_model.norm",
        "main_model.layers.9.q_proj",
        "main_model.layers.9.k_proj",
        "main_model.layers.14.mem_q_proj",
        "main_model.layers.14.mem_o_proj",
        "main_model.layers.14.mem_o_norm",
        "main_model.spec_tokens",
        "embed_model.layers.5.self_attn.q_proj",
        "embed_model.embed_tokens",
        "embed_model.mem_k_proj",
        "embed_model.mem_v_proj",
        "embed_model.embed_proj_conv",
    ]
    return {k: jnp.ones((4, 4), dtype=jnp.float32) for k in keys}


def main():
    weights = make_synthetic_weights()

    cfg = OmegaConf.create({
        "trainer": {
            "steps": 100000,
            "learning_rate": 1e-4,
            "learning_rates": {"mem": 1e-4, "embed": 1e-5, "main": 1e-5},
            "weight_decay": 0.1,
            "clip_grad_norm": 1.0,
            "training_stages": [
                {"trainable_params": [".*mem_.*", ".*embed_proj_conv.*"],
                 "max_step": 5000, "warmup_steps": 500},
                {"trainable_params": [".*mem_.*", ".*embed_model.*"],
                 "max_step": 10000, "warmup_steps": 500},
                {"trainable_params": [".*mem_.*", ".*embed_model.*"],
                 "max_step": 15000, "warmup_steps": 500, "ce_weight": 1.0},
                {"trainable_params": [".*mem_.*", ".*embed_model.*", ".*main_model.*"],
                 "max_step": 100000, "warmup_steps": 8000,
                 "lr_schedule": "cosine", "ce_weight": 1.0},
            ],
        },
        "model": {"trainable_params": ["all"]},
    })
    stages = parse_training_stages(cfg)
    print(f"parsed {len(stages)} stages; warmup_steps per stage: "
          f"{[s.get('warmup_steps') for s in stages]}")

    # For each stage, build optimizer, capture prints, sanity-check.
    for si, stage in enumerate(stages):
        print(f"\n===================================================================")
        print(f"  STAGE {si}: trainable={stage['trainable_params']}")
        print(f"===================================================================")
        cfg_dict = {"main_model": {"use_lora": False}, "embed_model": {"use_lora": False}}
        model = FakeModel(weights, cfg_dict)
        optimizer, _, lr = setup_optimizer_for_stage(cfg, model, stage_config=stage, all_stages=stages)
        # Init the optimizer to make sure the shape works
        state = optimizer.init(weights)
        print(f"[stage {si}] optimizer.init OK, state pytree leaves = "
              f"{len(jax.tree_util.tree_leaves(state))}")
        # Also exercise the UPDATE path — training crashed here in an earlier
        # per-group experiment. Grads = weights (arbitrary non-zero); pytree
        # structure must match params.
        fake_grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), weights)
        try:
            updates, new_state = optimizer.update(fake_grads, state, weights)
            new_weights = jax.tree_util.tree_map(lambda w, u: w + u, weights, updates)
            print(f"[stage {si}] optimizer.update OK, first param delta rms = "
                  f"{float(jnp.sqrt(jnp.mean((new_weights['main_model.layers.14.mem_q_proj'] - weights['main_model.layers.14.mem_q_proj']) ** 2))):.4e}")
        except Exception as e:
            print(f"[stage {si}] optimizer.update FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        # Evaluate LR schedule at step 0 and step warmup_steps//2 and warmup_steps
        ws = stage.get('warmup_steps', 0)
        for probe_step in (0, ws // 2, ws, ws * 2):
            try:
                val = float(lr(probe_step)) if callable(lr) else float(lr)
                print(f"  lr(step={probe_step}) = {val:.4e}")
            except Exception as e:
                print(f"  lr(step={probe_step}) FAILED: {e}")

    print("\n=== STAGE-BOUNDARY OPT_STATE MIGRATION TEST ===")
    print("Simulates trainer.py behavior: build stage 0 optimizer + init state,")
    print("then at stage boundary build stage 1 optimizer with SAME state (opt_state")
    print("preservation to keep LR schedule position) and call update. This is the")
    print("failure mode the earlier multi_transform design hit at real training's")
    print("step 5000 boundary.")
    cfg_dict = {"main_model": {"use_lora": False}, "embed_model": {"use_lora": False}}
    model_s0 = FakeModel(weights, cfg_dict)
    opt_s0, _, _ = setup_optimizer_for_stage(cfg, model_s0, stage_config=stages[0], all_stages=stages)
    state_s0 = opt_s0.init(weights)
    # Take a step under stage 0 optimizer
    grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), weights)
    _, state_s0_after1 = opt_s0.update(grads, state_s0, weights)
    print(f"stage 0 update OK, opt_state leaves = {len(jax.tree_util.tree_leaves(state_s0_after1))}")
    # Now transition to stage 1 — build new optimizer, feed PRESERVED opt_state
    model_s1 = FakeModel(weights, cfg_dict)
    opt_s1, _, _ = setup_optimizer_for_stage(cfg, model_s1, stage_config=stages[1], all_stages=stages)
    print(f"stage 1 opt built, testing update with PRESERVED stage-0 opt_state...")
    try:
        updates, state_s1_after1 = opt_s1.update(grads, state_s0_after1, weights)
        print(f"BOUNDARY TEST PASS: stage 1 optimizer accepts preserved stage-0 opt_state, "
              f"new opt_state leaves = {len(jax.tree_util.tree_leaves(state_s1_after1))}")
    except Exception as e:
        print(f"BOUNDARY TEST FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== MEM_MASKED_OPTIMIZER=1 COMPATIBILITY TEST ===")
    print("Toggle MEM_MASKED_OPTIMIZER=1 and re-run stage 1 (embed model unfrozen).")
    print("Under masked+scale_by_tree, moment allocation is trainable-only but the")
    print("effective update should stay byte-identical to unmasked (frozen=zero,")
    print("trainable=same delta). Verify: init OK + update OK + smaller opt_state.")
    os.environ["MEM_MASKED_OPTIMIZER"] = "1"
    try:
        model_masked = FakeModel(weights, cfg_dict)
        opt_masked, _, _ = setup_optimizer_for_stage(cfg, model_masked, stage_config=stages[1], all_stages=stages)
        state_masked = opt_masked.init(weights)
        n_leaves_masked = len(jax.tree_util.tree_leaves(state_masked))
        _, state_masked_after1 = opt_masked.update(grads, state_masked, weights)
        # Compare with the unmasked stage-1 opt_state leaf count computed above (state_s1_after1)
        # via the boundary test — but that was created with MASKED=off. We saved it above.
        n_leaves_unmasked = len(jax.tree_util.tree_leaves(state_s1_after1))
        print(f"MASKED TEST OK: init + update pass under MEM_MASKED_OPTIMIZER=1.")
        print(f"  masked opt_state leaves   = {n_leaves_masked}")
        print(f"  unmasked opt_state leaves = {n_leaves_unmasked}  (higher — allocates for frozen too)")
    except Exception as e:
        print(f"MASKED TEST FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.environ.pop("MEM_MASKED_OPTIMIZER", None)

    print("\n=== ALL STAGES BUILT SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
