#!/bin/bash
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

TARGET_ROWS_PER_SOURCE="${TARGET_ROWS_PER_SOURCE:?set TARGET_ROWS_PER_SOURCE=<the smallest sources exact post-filter count>}"

HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
JAX_PLATFORMS=cpu \
uv run python data/preprocess_arrayrecord.py \
  --dataset "${DATASET:-qa_hard_neg_no_multihop_sft4b}" \
  --tokenizer "${TOKENIZER:-Qwen/Qwen3-4B}" \
  --out "${OUT:-gs://memory-layers-training/indexed/}" \
  --target-rows-per-source "$TARGET_ROWS_PER_SOURCE"
