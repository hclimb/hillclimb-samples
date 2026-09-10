#!/bin/bash
# Bottleneck #2 microbench: per-step device sync in the training loop (data removed via synthetic
# batch). One XLA compile then fast timing arms, so it's quick. Tees to $LOOP_LOG so results
# survive an SSH disconnect. Launch from a worktree with:
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/bench_loop_sync.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
LOOP_LOG="${LOOP_LOG:-$HOME/bench_loop_sync.log}"
{
  echo "############ host $(hostname)  $(date -u) ############"
  uv run python scripts/embed/bench_loop_sync.py --check || { echo "!!!! CHECK FAILED"; exit 1; }
  uv run python scripts/embed/bench_loop_sync.py
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$LOOP_LOG"
