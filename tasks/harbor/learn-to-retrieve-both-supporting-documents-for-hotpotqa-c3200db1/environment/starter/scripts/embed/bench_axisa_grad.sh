#!/bin/bash
# Prove Axis A quality-neutrality: trainable-param grads identical (stop_gradient on vs off). Stage 0.
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/bench_axisa_grad.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
GRAD_LOG="${GRAD_LOG:-$HOME/bench_axisa_grad.log}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a
{
  echo "############ host $(hostname)  $(date -u) ############"
  uv run python scripts/embed/bench_axisa_grad.py --stage 0
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$GRAD_LOG"
