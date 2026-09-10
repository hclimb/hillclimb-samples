"""Standalone test for apply_checkpoint_model_cfg (evals/shared.py).

The bug it guards: configs/eval.yaml always composes a full default `model` (qwen3_mem_embed,
mem_layers=[14]), so the eval worker's OmegaConf.merge let that default OVERRIDE a checkpoint's
trained architecture — a 4-layer checkpoint (mem_layers=[9,14,20,27]) was silently rebuilt as
mem_layers=[14] and partial_restore dropped the other 3 layers' weights. This asserts the
checkpoint's saved model config is now authoritative, while explicit CLI overrides still win.

Run:  uv run python tests/test_checkpoint_model_authoritative.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from evals.shared import apply_checkpoint_model_cfg


def _eval_cfg():
    # The eval default: one memory layer (this is what silently clobbered).
    return OmegaConf.create({
        "checkpoint_dir": "gs://b/run-2026-07-04-09-42-55/qwen3_mem_embed/38000",
        "model": {"name": "qwen3_mem_embed",
                  "memory": {"mem_layers": [14], "mem_num_heads": 4, "mem_top_k": 128}},
    })


def _train_cfg():
    # The checkpoint's saved config: FOUR memory layers, otherwise-matching dims.
    return OmegaConf.create({
        "model": {"name": "qwen3_mem_embed",
                  "memory": {"mem_layers": [9, 14, 20, 27], "mem_num_heads": 4, "mem_top_k": 128}},
    })


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    ok = True

    # 1. No explicit overrides -> checkpoint's 4 layers win (the core fix).
    cfg = apply_checkpoint_model_cfg(_eval_cfg(), _train_cfg(), cli_overrides=[
        "checkpoint_dir=gs://b/run/qwen3_mem_embed/38000", "tp_devices=1", "use_wandb=false"])
    ok &= check("no-override: mem_layers == [9,14,20,27] (checkpoint authoritative)",
                list(cfg.model.memory.mem_layers) == [9, 14, 20, 27])

    # 2. Explicit model.<field> override -> layered on the trained arch (arch still from checkpoint).
    cfg = apply_checkpoint_model_cfg(_eval_cfg(), _train_cfg(), cli_overrides=[
        "model.memory.mem_top_k=64"])
    ok &= check("field-override: mem_layers still [9,14,20,27]",
                list(cfg.model.memory.mem_layers) == [9, 14, 20, 27])
    ok &= check("field-override: mem_top_k == 64 (explicit override applied)",
                int(cfg.model.memory.mem_top_k) == 64)

    # 3. Explicit whole-model group swap (model=<name>) -> respect cfg.model unchanged.
    swapped = _eval_cfg()
    swapped.model.memory.mem_layers = [14]
    cfg = apply_checkpoint_model_cfg(swapped, _train_cfg(), cli_overrides=["model=qwen3_mem_embed_16q4kv"])
    ok &= check("group-swap: cfg.model left as-is ([14], caller's explicit choice)",
                list(cfg.model.memory.mem_layers) == [14])

    # 4. No checkpoint -> cfg.model is the only source of truth; unchanged.
    nock = _eval_cfg()
    cfg = apply_checkpoint_model_cfg(nock, train_cfg=None, cli_overrides=[])
    ok &= check("no-checkpoint: cfg.model unchanged ([14])",
                list(cfg.model.memory.mem_layers) == [14])

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
