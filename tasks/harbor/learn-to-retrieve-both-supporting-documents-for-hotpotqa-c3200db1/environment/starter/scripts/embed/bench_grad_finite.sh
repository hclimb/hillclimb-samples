#!/bin/bash
# Diagnose the capstone grad_norm=nan: which param group is non-finite on real offline data (stage 0).
GF_LOG="${GF_LOG:-$HOME/bench_grad_finite.log}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a
{
  echo "############ host $(hostname)  $(date -u) ############"
  HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="$HOME/hf_parquet" \
    uv run python scripts/embed/bench_grad_finite.py --batches 3 --stage 3
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$GF_LOG"
