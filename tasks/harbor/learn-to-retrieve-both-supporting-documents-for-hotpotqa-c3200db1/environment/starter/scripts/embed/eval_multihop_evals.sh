#!/bin/bash
set -e

# Load .env
set -a; source "$(dirname "$0")/../../.env"; set +a

CHECKPOINT_DIR=gs://memory-layers-training/4B_pretraining_cot_unfreeze_all_topk_128_norm-2026-04-12-00-36-01/qwen3_mem_embed/100000

uv run eval.py \
    checkpoint_dir=${CHECKPOINT_DIR} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=multihop_evals' \
    tp_devices=1
