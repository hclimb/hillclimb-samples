#!/bin/bash
# Both arms of the stage-3 NaN A/B, back-to-back on ONE box so the data and weights are identical
# and MEM_APPROX_TOPK is the only variable.
#   RESUME_FROM=gs://.../qwen3_mem_embed/16000 bash scripts/embed/debug_stage3_nan_ab.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
RESUME_FROM="${RESUME_FROM:?set RESUME_FROM=gs://.../qwen3_mem_embed/<step>}"
N="${STOP_AT:-60}"

for approx in 1 0; do
  echo
  echo "##################################################################"
  echo "# ARM MEM_APPROX_TOPK=$approx   ($([ "$approx" = 1 ] && echo 'today: approx_max_k' || echo 'April: exact top_k'))"
  echo "##################################################################"
  RESUME_FROM="$RESUME_FROM" MEM_APPROX_TOPK=$approx STOP_AT="$N" TAG="debug_nan_approx${approx}" \
    bash scripts/embed/debug_stage3_nan.sh 2>&1 \
    | grep -viE "^WARNING:absl|Grain multiprocess|^(embed_model|main_model)\." \
    | grep -E "DEBUG NaN REPRO|run id:|Starting training|not finite|Loss:|grad_norm|Traceback|Error|it/s\]" \
    | tail -25
done
