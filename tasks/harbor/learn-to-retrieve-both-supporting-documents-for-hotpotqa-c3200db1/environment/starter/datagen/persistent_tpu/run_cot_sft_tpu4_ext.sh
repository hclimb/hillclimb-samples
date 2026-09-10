#!/bin/bash
# Dataset 4/4: triviaqa-pairs (2 parquets) — external-IP variant (no NAT)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export TPU_NAME=suhas-machine-2
export ZONE=europe-west4-a
export PROJECT_ID=memorylayers

INPUT_REPO="vm2825/triviaqa-pairs-hard-neg-reasoning-embedding-modified"
PARQUETS="0"
CHUNK_SIZE=1
PARQUET_TEMPLATE="data/train-{p:05d}.parquet"

python "$SCRIPT_DIR/orchestrator_ext.py" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --tpu-name "$TPU_NAME" \
  --tpu-type v6e-8 \
  --env-file "$REPO_ROOT/.env" \
  --parquets "$PARQUETS" \
  --chunk-size "$CHUNK_SIZE" \
  --state-file cot_sft_state_tpu4_ext.json \
  --setup-script setup_scienceqa.sh \
  --run-command "uv run datagen/persistent_tpu/generate_cot_sft.py --input-repo $INPUT_REPO --parquet-numbers '{CHUNKS}' --parquet-template '$PARQUET_TEMPLATE' --tp-size 8"
