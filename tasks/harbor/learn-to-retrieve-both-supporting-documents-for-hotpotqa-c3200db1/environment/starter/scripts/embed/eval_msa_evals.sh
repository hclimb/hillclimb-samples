#!/bin/bash
set -e

# Load .env
set -a; source "$(dirname "$0")/../../.env"; set +a

# Evaluate the pretrained MSA-4B model (EverMind-AI/MSA-4B) on the MSA benchmark
# suite (9 QA datasets) using the MSA static-memory evaluator
# (type=generation_large_mem_msa, see evals/gen_large_mem_msa.py).
#
# MSA-4B weights are loaded directly from HuggingFace at model-init time, so there
# is NO checkpoint_dir / GCS checkpoint — we pass model=qwen3_msa instead.
#
# Prereq: the msa-*-{docs,qa}-with-ids datasets must exist under $HF_USERNAME
#   (data/utils/prepare_msa_docs.py + prepare_msa_qa_with_ids.py, idempotent).
#
# Extra hydra overrides may be passed through, e.g.:
#   bash scripts/embed/eval_msa_evals.sh evals.gen_large_mem_msa_popqa.eval.num_samples=16

uv run eval.py \
    model=qwen3_msa \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=msa_evals' \
    tp_devices=1 \
    use_wandb=false \
    "$@"
