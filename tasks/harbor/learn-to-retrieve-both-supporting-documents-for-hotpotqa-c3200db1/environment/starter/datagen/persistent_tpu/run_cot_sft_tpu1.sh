#!/bin/bash
# Dataset 1/4: msmarco-triplets (10 parquets)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export TPU_NAME=cot-sft-tpu-1
export ZONE=europe-west4-a
export PROJECT_ID=memory-layers-484918

INPUT_REPO="vm2825/msmarco-triplets-hard-neg-reasoning-embedding-modified"
PARQUETS="0-9"
CHUNK_SIZE=10
PARQUET_TEMPLATE="data/train-{p:05d}.parquet"

python "$SCRIPT_DIR/orchestrator.py" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --tpu-name "$TPU_NAME" \
  --tpu-type v6e-8 \
  --env-file "$REPO_ROOT/.env" \
  --parquets "$PARQUETS" \
  --chunk-size "$CHUNK_SIZE" \
  --state-file cot_sft_state_tpu1.json \
  --setup-script setup_scienceqa.sh \
  --run-command "uv run datagen/persistent_tpu/generate_cot_sft.py --input-repo $INPUT_REPO --parquet-numbers '{CHUNKS}' --parquet-template '$PARQUET_TEMPLATE' --tp-size 8"
