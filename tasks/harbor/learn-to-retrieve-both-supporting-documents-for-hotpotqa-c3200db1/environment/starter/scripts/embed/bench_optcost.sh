#!/bin/bash
# Optimizer-update cost decomposition (lever F bound), stage 3. Tees to $OPT_LOG.
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/bench_optcost.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
OPT_LOG="${OPT_LOG:-$HOME/bench_optcost.log}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a
{
  echo "############ host $(hostname)  $(date -u) ############"
  uv run python scripts/embed/bench_optcost.py --iters 20 --warmup 5 --stage 3
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$OPT_LOG"
