#!/bin/bash
# Verify + measure the wired Axis-A path in Trainer._train_step (stage 0). Tees to $AXISA_LOG.
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/bench_axisa.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
AXISA_LOG="${AXISA_LOG:-$HOME/bench_axisa.log}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a
{
  echo "############ host $(hostname)  $(date -u) ############"
  uv run python scripts/embed/bench_axisa.py --iters 20 --warmup 5 --stage 0
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$AXISA_LOG"
