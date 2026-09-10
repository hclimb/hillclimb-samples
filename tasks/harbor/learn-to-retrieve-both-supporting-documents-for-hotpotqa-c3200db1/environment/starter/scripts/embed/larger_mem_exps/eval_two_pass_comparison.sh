#!/bin/bash
set -e
set -a; source "$(dirname "$0")/../../../.env"; set +a

CKPT_256K=gs://memory-layers-training/4B_pretraining_two_pass-2026-04-18-04-21-44/qwen3_mem_embed/101000
CKPT_128K=gs://memory-layers-training/4B_pretraining_two_pass-2026-04-17-20-11-27/qwen3_mem_embed/102250

OUT_DIR_256K=outputs/two_pass_eval/256k_mem
OUT_DIR_128K=outputs/two_pass_eval/128k_mem

echo "=== Running 256k mem checkpoint (step 101000) ==="
uv run eval.py \
    checkpoint_dir=${CKPT_256K} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=embed_training_scale_sparse' \
    tp_devices=1 \
    '++aux_losses.doc_access_acc.enabled=true' \
    hydra.run.dir=${OUT_DIR_256K}

echo "=== Running 128k mem checkpoint (step 102250) ==="
uv run eval.py \
    checkpoint_dir=${CKPT_128K} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=embed_training_scale_sparse' \
    tp_devices=1 \
    '++aux_losses.doc_access_acc.enabled=true' \
    hydra.run.dir=${OUT_DIR_128K}

echo "=== Writing results ==="
uv run python analysis/write_two_pass_md.py \
    --dir_a ${OUT_DIR_256K} \
    --dir_b ${OUT_DIR_128K} \
    --label_a "256k mem" \
    --label_b "128k mem" \
    --step_a 101000 \
    --step_b 102250 \
    --out_dir results/two_pass_comparison
