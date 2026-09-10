#!/bin/bash
# Real end-to-end training throughput (offline-parquet + real _train_step). Baseline sync-each vs
# pipelined sync-K. Tees to $TT_LOG. Launch from a worktree with:
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/bench_train_throughput.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
# STEPS / SYNC_K / STAGE override the run (defaults: 200 steps/arm, sync every 20, stage 0).
TT_LOG="${TT_LOG:-$HOME/bench_train_throughput.log}"
STEPS="${STEPS:-200}"; SYNC_K="${SYNC_K:-20}"; STAGE="${STAGE:-0}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a
{
  echo "############ host $(hostname)  $(date -u)  steps=$STEPS sync_k=$SYNC_K stage=$STAGE ############"
  HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="$HOME/hf_parquet" \
    uv run python scripts/embed/bench_train_throughput.py --steps "$STEPS" --sync-k "$SYNC_K" --stage "$STAGE" \
    || echo "!!!! RUN FAILED"
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$TT_LOG"
