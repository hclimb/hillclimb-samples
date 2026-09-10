"""H_B probe: freeze weights, vary data position, measure doc_access_acc.

Question: was the doc_access_acc jump from 0.94→0.86 at step 28000 caused by
the training model's weight state, or by the data at that stream position?

Approach:
  1. Load ckpt=28000 (weights snapshot AT the jump).
  2. Restore Grain loader state from ckpt=26000's dataloader_state.json (which
     is the state AFTER consuming batch 26000, so next batch = 26001).
  3. Run N=200 forward-only steps, log doc_access_acc per batch.
  4. Rebuild pipeline, restore loader state from ckpt=28000's dataloader_state.json
     (next batch = 28001).
  5. Run N=200 forward-only steps, log doc_access_acc per batch.
  6. Compare pre_avg vs post_avg with SAME weights.

If pre_avg (steps 26001-26200) >> post_avg (steps 28001-28200) with frozen weights,
H_B (data heterogeneity in the stream) is confirmed. If they're equal, some other
cause moved doc_access_acc training-side.

Usage (on a TPU box):
    HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$HOME/hf_parquet \
        uv run python scripts/debug/probe_h_b_data_heterogeneity.py \
        --ckpt_dir gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_main_exact_loader_resume-2026-08-06-09-09-19/qwen3_mem_embed
"""
import argparse
import json
import os
from functools import partial

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Ensure repo root is on path so `import data`, `import models` work
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_manager import MultiprocessingOptions

from data import get_dataset
from models import get_model
from utils import init_jax_distributed, load_checkpoint, process_train_pairs
from losses.doc_access_acc import compute_doc_access_acc


def build_cfg():
    """Minimal Hydra config to reproduce the training run's setup."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    with initialize_config_dir(config_dir=os.path.join(root, "configs"), version_base="1.2"):
        cfg = compose(
            config_name="train",
            overrides=[
                "model=qwen3_mem_embed",
                "model.main_model.model_id=Qwen/Qwen3-4B",
                "model.memory.mem_top_k=64",
                "model.memory.mem_approx_topk=false",
                "dataset=qa_hard_neg_think_sft4b",
                "trainer=staged_telemetry",
                "trainer.checkpoint_interval=2000",
                "trainer.max_to_keep=16",
                "trainer.log_interval=1",
                "trainer.eval_interval=1000000000",
                "eval_set@trainer.evals=none",
                "trainer.use_wandb=false",  # probe doesn't need wandb
            ],
        )
    return cfg


def make_load_manager(ckpt_dir):
    """Construct an orbax CheckpointManager pointing directly at ckpt_dir
    (contains step subdirs like 26000/, 28000/). Bypasses utils.setup_checkpointing
    since we only READ, and setup_checkpointing needs HydraConfig singleton set."""
    options = ocp.CheckpointManagerOptions(
        multiprocessing_options=MultiprocessingOptions(primary_host=0),
    )
    return ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options=options)


def load_json_state(gcs_path):
    with fsspec.open(gcs_path, "r") as f:
        return json.load(f)


def run_probe_at(dataset, model, loader_state, n_steps, label, forward_fn):
    """Set the Grain loader to `loader_state`, then run n_steps forward-only,
    returning list of per-batch doc_access_acc."""
    print(f"\n=== {label} ===", flush=True)
    dataset.set_loader_state(loader_state)
    print(f"[{label}] loader state set; starting generator (Grain will fast-forward)", flush=True)
    gen = dataset.generator(num_epochs=1)
    accs = []
    pbar = tqdm(total=n_steps, desc=label)
    for i, (tokens, masks) in enumerate(gen):
        if i >= n_steps:
            break
        inputs, targets, input_masks, loss_masks, _ = process_train_pairs(tokens, masks)
        aux_data = forward_fn(model.weights, inputs, input_masks)
        # aux_data is a dict with `mem_top_k_indices` populated (collect_aux=True in forward_fn)
        acc = float(compute_doc_access_acc(aux_data, loss_masks, input_masks, inputs))
        accs.append(acc)
        pbar.update(1)
        pbar.set_postfix(acc=f"{acc:.3f}", mean=f"{np.mean(accs):.4f}")
    pbar.close()
    print(f"[{label}] N={len(accs)}  mean_acc={np.mean(accs):.4f}  std={np.std(accs):.4f}", flush=True)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="GCS CheckpointManager directory (contains 26000/, 28000/, ...)")
    ap.add_argument("--weights_step", type=int, default=28000,
                    help="Ckpt step to load WEIGHTS from (default 28000 — post-jump)")
    ap.add_argument("--pre_step", type=int, default=26000,
                    help="Ckpt step to load loader_state from for PRE-jump run (default 26000)")
    ap.add_argument("--post_step", type=int, default=28000,
                    help="Ckpt step to load loader_state from for POST-jump run (default 28000)")
    ap.add_argument("--n_steps", type=int, default=200,
                    help="Forward-only steps per condition (default 200)")
    args = ap.parse_args()

    init_jax_distributed()

    print(f"=== H_B PROBE ===", flush=True)
    print(f"weights from: {args.ckpt_dir}/{args.weights_step}", flush=True)
    print(f"pre-loader from: {args.ckpt_dir}/{args.pre_step}/dataloader_state.json", flush=True)
    print(f"post-loader from: {args.ckpt_dir}/{args.post_step}/dataloader_state.json", flush=True)
    print(f"n_steps per condition: {args.n_steps}\n", flush=True)

    cfg = build_cfg()

    print("Building model...", flush=True)
    model = get_model(cfg.model, cfg.trainer.get("tp_devices", 1))

    print("Building dataset...", flush=True)
    dataset = get_dataset(cfg.dataset, model)

    print(f"Loading weights from ckpt={args.weights_step} in {args.ckpt_dir}...", flush=True)
    # Warm-start path (resume_step != None): only weights are restored, opt_state stays None.
    # Bypasses setup_checkpointing (which requires HydraConfig singleton).
    load_mgr = make_load_manager(args.ckpt_dir)
    step_out, _opt_state_out, stage_idx = load_checkpoint(
        load_mgr, model, None,
        return_stage_idx=True,
        resume_step=args.weights_step,
        resume_from_dir=args.ckpt_dir,
    )
    print(f"  loaded step={step_out}, stage_idx={stage_idx}", flush=True)

    # Load the two loader states from GCS
    pre_state_path = f"{args.ckpt_dir}/{args.pre_step}/dataloader_state.json"
    post_state_path = f"{args.ckpt_dir}/{args.post_step}/dataloader_state.json"
    print(f"\nFetching loader states...", flush=True)
    pre_state = load_json_state(pre_state_path)
    post_state = load_json_state(post_state_path)
    print(f"  pre  worker_counts: {[w['count'] for w in pre_state['iterators_in_use_states']]}", flush=True)
    print(f"  post worker_counts: {[w['count'] for w in post_state['iterators_in_use_states']]}", flush=True)

    # Build the forward function (jit'd, collect_aux=True so mem_top_k_indices are exposed)
    @partial(jax.jit, static_argnames=("collect_aux",))
    def forward_fn(weights, inputs, input_masks, collect_aux=True):
        pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
        output = model.forward(inputs, weights, pad_mask=pad_mask, collect_aux=collect_aux)
        return output.aux

    # Wrapper that reorders args to match run_probe_at's forward_fn signature
    def _fwd(weights, inputs, input_masks):
        return forward_fn(weights, inputs, input_masks, collect_aux=True)

    # Run the two conditions
    pre_accs = run_probe_at(dataset, model, pre_state, args.n_steps, "PRE (loader state = step 26000)", _fwd)
    post_accs = run_probe_at(dataset, model, post_state, args.n_steps, "POST (loader state = step 28000)", _fwd)

    print("\n=== SUMMARY ===")
    print(f"weights: ckpt={args.weights_step} (FROZEN across both conditions)")
    print(f"PRE  (step {args.pre_step+1}..{args.pre_step+args.n_steps}):  mean_acc={np.mean(pre_accs):.4f}  std={np.std(pre_accs):.4f}  n={len(pre_accs)}")
    print(f"POST (step {args.post_step+1}..{args.post_step+args.n_steps}): mean_acc={np.mean(post_accs):.4f}  std={np.std(post_accs):.4f}  n={len(post_accs)}")
    diff = np.mean(pre_accs) - np.mean(post_accs)
    print(f"\nΔ (PRE - POST) = {diff:+.4f}")
    if abs(diff) > 0.03:
        print("VERDICT: Substantial acc difference at FROZEN weights → H_B data heterogeneity CONFIRMED.")
    else:
        print("VERDICT: No significant difference at frozen weights → H_B NOT the cause; look at H9/H10 or others.")


if __name__ == "__main__":
    main()
