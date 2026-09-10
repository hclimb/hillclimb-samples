#!/bin/bash
# Stage the multihop hard-neg data on the box (no training yet).
set -uo pipefail
LOG="${STAGE_LOG:-$HOME/stage_multihop.log}"
{
  echo "######## staging | $(hostname) | $(date -u) ########"
  uv run python datagen/download_multihop_hardneg.py ${MAX_ROWS:+--max-rows $MAX_ROWS}
  echo "STAGE_EXIT=$?"
  echo "######## files ########"
  ls -la ~/hf_parquet/ 2>/dev/null
  ls -la ~/hf_parquet/mihir-1999__multihop_qa_sft-hard-neg-train/ 2>/dev/null
  echo "######## DONE $(date -u) ########"
} 2>&1 | tee "$LOG"
