#!/bin/bash
# Stage-aware, quality-neutral speed profiler for the qa_hard_neg_think_sft4b train step.
# Runs the whole profile in ONE process (unlike bench_approx_topk.sh, no import-time env toggle),
# tee'ing to $PROFILE_LOG so results survive an SSH disconnect (read the log back on the box).
# Launch from a worktree with:
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/profile_train_step.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
# STAGES overrides which stages to profile. Default is the QUICK scope "0 3" — stage 0 (biggest
# Axis-A prize) + stage 3 (85%-of-steps control + remat signal), ~5 compiles. Set STAGES="0 1 2 3"
# for the full whole-run projection (~12 compiles, ~2x slower).
PROFILE_LOG="${PROFILE_LOG:-$HOME/profile_train_step.log}"
STAGES="${STAGES:-0 3}"
{
  echo "############ host $(hostname)  $(date -u) ############"
  echo "############ stages: ${STAGES} ############"
  # Fast wiring gate before the expensive XLA compiles: compose+imports only, no accelerator.
  uv run python scripts/embed/profile_train_step.py --check --stages ${STAGES} || { echo "!!!! CHECK FAILED"; exit 1; }
  uv run python scripts/embed/profile_train_step.py --stages ${STAGES}
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$PROFILE_LOG"
