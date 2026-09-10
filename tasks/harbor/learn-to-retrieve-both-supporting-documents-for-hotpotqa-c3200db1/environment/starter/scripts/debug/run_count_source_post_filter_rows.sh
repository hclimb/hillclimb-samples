#!/bin/bash
set -uo pipefail
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
JAX_PLATFORMS=cpu \
uv run python scripts/debug/count_source_post_filter_rows.py \
  --dataset "${DATASET:-qa_hard_neg_no_multihop_sft4b}" \
  --source "${SOURCE:-combined_hard_neg_sft4b}" \
  --trainer "${TRAINER:-staged_ground}"
