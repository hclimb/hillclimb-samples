#!/bin/bash
# Test "pre-resolve once -> stream" vs the original name-based streaming, LIVE (no offline), 16 workers.
PRS_LOG="${PRS_LOG:-$HOME/bench_preresolve.log}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a
{
  echo "############ host $(hostname)  $(date -u) ############"
  echo "===== ARM 1: PRE-RESOLVE (resolve shard URLs once in main, stream via hf:// data_files) ====="
  ( unset HF_HUB_OFFLINE; uv run python scripts/embed/bench_preresolve_stream.py --arm preresolve --batches 200 --max-seconds 300 ) \
    || echo "!!!! PRERESOLVE ARM FAILED"
  echo
  echo "===== ARM 2: NAME (control = original load_dataset(name) streaming; bounded, may 429-stall) ====="
  ( unset HF_HUB_OFFLINE; timeout 200 uv run python scripts/embed/bench_preresolve_stream.py --arm name --batches 100 --max-seconds 170 ) \
    || echo "==== NAME arm stopped (timeout/429 — that itself is the control signal) ===="
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$PRS_LOG"
