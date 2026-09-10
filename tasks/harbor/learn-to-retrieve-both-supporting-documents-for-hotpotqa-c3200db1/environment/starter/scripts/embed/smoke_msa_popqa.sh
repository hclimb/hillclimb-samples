#!/bin/bash
set -e
set -a; source "$(dirname "$0")/../../.env"; set +a
# Smoke test: MSA-4B on popqa only, few samples, NO llm-judge (drop metrics).
uv run eval.py \
    model=qwen3_msa \
    '~eval_set@evals=pretraining' \
    '+eval/tasks@evals.popqa=gen_large_mem_msa_popqa' \
    'evals.popqa.eval.num_samples=8' \
    'evals.popqa.dataset.batch_size=8' \
    '~evals.popqa.eval.metrics' \
    tp_devices=1 use_wandb=false "$@"
