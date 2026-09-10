#!/bin/bash
# Reproduce the stage-3 NaN with a REAL training run — no reimplementation.
#
# Uses trainer=staged_debug_stage3: stage 3's settings from step 0, so a WARM-STARTED checkpoint
# (resume_from=<dir>/<step> loads weights but restarts the step counter at 0) lands directly in
# the failing configuration instead of replaying 15000 steps to get there (~2h).
#
# WHY THIS AND NOT A STANDALONE SCRIPT: every attempt to rebuild the train step outside train.py
# diverged from it (mesh/jit context, checkpoint manager, hydra runtime) and failed on the
# scaffolding rather than the bug. train.py IS the code path under test. Resuming from the last
# pre-NaN checkpoint means the run walks into stage 3 (step 15000) on its own, with the real
# weights, data, optimizer and stage-transition logic.
#
# WHAT TO WATCH (wandb, or the box log):
#   train/grad_norm_main_core  — the 4B proper (attn/mlp/embed/lm_head)
#   train/grad_norm_main_mem   — memory read/route params inside main_model
#   train/grad_norm_embed_mem  — embed trunk's mem_k_proj/mem_v_proj
# These are DISJOINT (trainer.py) unlike grad_norm_main/mem, which overlap and made the original
# signal unreadable. Whichever goes NaN first names the failing subsystem.
#
# ARMS: MEM_APPROX_TOPK=1 reproduces today's default (approx_max_k). =0 is April's exact top_k —
# the ONLY functional config delta vs the last known-good run (707a128). If =0 is clean and =1
# NaNs, that is the answer, and it is the Tier-2 A/B that
# wiki/experiments/2026-07-15-approx-topk-training.md deferred ("skipped by decision").
#
#   RESUME_FROM=gs://.../qwen3_mem_embed/14000 MEM_APPROX_TOPK=1 bash scripts/embed/debug_stage3_nan.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

RESUME_FROM="${RESUME_FROM:?set RESUME_FROM=gs://.../qwen3_mem_embed/<step>}"
APPROX="${MEM_APPROX_TOPK:-1}"
STOP_AT="${STOP_AT:-60}"             # stage 3 is live from step 0 here; the NaN showed within ~100 steps
TRAINER="${TRAINER:-staged_debug_stage3}"
# LR: pin the peak LR to decouple "main_model is trainable" from "main_model is UPDATED hard".
# The stage2-vs-stage3 control was confounded: with trainer.steps=20, warmup_frac=0.1 gives
# warmup_steps=2, so stage 3's LR hit the 1e-4 peak within 2 steps and the 4B was updated 1e4x
# harder than in the real run (which NaN'd at LR=1e-8, count=0, i.e. BEFORE any update landed).
# LR=1e-8 => Adam's ~unit-norm update is scaled to ~1e-8/step => weights are frozen to ~1e-7 over
# 20 steps. Any NaN under LR=1e-8 is therefore the BACKWARD AT W_ckpt, not divergence.
LR="${LR:-}"
TAG="${TAG:-debug_nan_approx${APPROX}}"

# Offline data (the cache is on this box); live-HF livelocks — wiki/data/hf-rate-limits.md
export HF_HUB_OFFLINE=1
export GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}"
export MEM_APPROX_TOPK="$APPROX"

echo "=== DEBUG NaN REPRO: MEM_APPROX_TOPK=$APPROX  resume=$RESUME_FROM  stop=$STOP_AT ==="

# log_interval=1: per-step logging so the exact step the grads flip is visible (the +10%
# throughput from pipelining is irrelevant for a 400-step debug run).
# checkpoint_interval huge: this run is throwaway, don't write 24G checkpoints.
# Distinct run_name per arm => distinct run-dir => distinct wandb id (utils.run_dir_name), so the
# two arms cannot land in the same curve.
uv run train.py \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    dataset=qa_hard_neg_think_sft4b \
    trainer="$TRAINER" \
    ${LR:+trainer.learning_rate=$LR} \
    trainer.steps="$STOP_AT" \
    trainer.resume_from="$RESUME_FROM" \
    trainer.checkpoint_interval=1000000 \
    trainer.log_interval=1 \
    trainer.eval_interval=1000000000 \
    'eval_set@trainer.evals=none' \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="$TAG"
