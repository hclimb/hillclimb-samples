#!/bin/bash
set -e

# Load .env
set -a; source "$(dirname "$0")/../../.env"; set +a

CHECKPOINT_DIR=gs://memory-layers-training/4B_pretraining_cot_unfreeze_all_topk_128_norm-2026-04-12-00-36-01/qwen3_mem_embed/100000

# Prepare NovelHopQA books dataset (idempotent HF push)
echo "=== Preparing NovelHopQA books ==="
uv run python data/utils/prepare_novelhopqa.py

echo "=== Running eval ==="
uv run eval.py \
    checkpoint_dir=${CHECKPOINT_DIR} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=long_context_evals' \
    tp_devices=1
