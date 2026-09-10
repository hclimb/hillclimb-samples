#!/bin/bash
set -e

# Load .env
set -a; source "$(dirname "$0")/../../../.env"; set +a

# Eval the generation_embed setting with batch sizes that keep the memory layer
# close to the training distribution (~32k tokens), unlike gen_large_mem which
# loads millions of tokens.
#   MS MARCO:  batch_size=128
#   HotpotQA:  batch_size=64
CHECKPOINT_DIR=gs://memory-layers-training/4B_pretraining_cot_unfreeze_all_topk_128_norm-2026-04-12-00-36-01/qwen3_mem_embed/100000

echo "=== Running generation_embed eval at training-scale memory ==="
uv run eval.py \
    checkpoint_dir=${CHECKPOINT_DIR} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=embed_training_scale' \
    tp_devices=1 \
    '++aux_losses.doc_access_acc.enabled=true'

echo "=== Plotting memory scale sweep ==="
STEP=$(basename ${CHECKPOINT_DIR})
uv run python analysis/plot_mem_scale_sweep.py \
    --step "${STEP}" \
    --results_dir results/
