#!/bin/bash
# A/B microbenchmark for MEM_APPROX_TOPK on the qa_hard_neg_think_sft4b step.
# Runs each timing variant as its own process (the approx toggle is read at import time),
# then the env-independent recall-fidelity pass. Run from the repo root on a TPU box after
# `source scripts/infrastructure/setup_shell.sh`. All output is tee'd to $BENCH_LOG so results
# survive an SSH disconnect (read the log back on the box).
BENCH_LOG="${BENCH_LOG:-$HOME/bench_approx_topk.log}"
{
  echo "############ host $(hostname)  $(date -u) ############"
  echo "############ EXACT  top_k (MEM_APPROX_TOPK=0) ############"
  MEM_APPROX_TOPK=0 uv run python scripts/embed/bench_approx_topk.py --mode time

  echo "############ APPROX top_k  recall_target=0.95 ############"
  MEM_APPROX_TOPK=1 MEM_APPROX_RECALL=0.95 uv run python scripts/embed/bench_approx_topk.py --mode time

  echo "############ APPROX top_k  recall_target=0.99 ############"
  MEM_APPROX_TOPK=1 MEM_APPROX_RECALL=0.99 uv run python scripts/embed/bench_approx_topk.py --mode time

  echo "############ RECALL fidelity (approx vs exact top-64) ############"
  uv run python scripts/embed/bench_approx_topk.py --mode recall

  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$BENCH_LOG"
