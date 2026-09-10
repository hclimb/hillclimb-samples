#!/bin/bash
set -e

# Load .env (sourced by the launcher, not executed, so $0 isn't the script path here — and .env
# lands at $HOME/.env on the box, scp'd separately from the repo checkout, not inside it).
set -a; . "$HOME/.env"; set +a

CHECKPOINT_DIR=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55/qwen3_mem_embed/90000

uv run eval.py \
    checkpoint_dir=${CHECKPOINT_DIR} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=ruler' \
    'evals.ruler.eval.tasks=[niah_s]' \
    'evals.ruler.eval.context_lengths=[131072]' \
    tp_devices=1
