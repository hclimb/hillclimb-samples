#!/bin/bash
# Phase 1: generate MSA-4B answers for each MSA dataset (NO judge), isolated per
# dataset so a big-corpus OOM doesn't lose the others. Judge separately with
# scripts/embed/judge_msa.py.
set -a; source "$(dirname "$0")/../../.env"; set +a
. "$HOME/.local/bin/env" 2>/dev/null || true

DATASETS="${@:-popqa hotpotqa musique natural_questions 2wikimultihopqa narrativeqa dureader msmarco_v1 triviaqa_10m}"

declare -A KEY=( [2wikimultihopqa]=wiki2hop )
for d in $DATASETS; do
  k=${KEY[$d]:-$d}
  echo "================ GEN $d ================"
  uv run --no-sync eval.py \
      model=qwen3_msa \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.${k}=gen_large_mem_msa_${d}" \
      "evals.${k}.eval.doc_dataset.max_docs=4000" \
      "evals.${k}.eval.num_samples=128" \
      "~evals.${k}.eval.metrics" \
      tp_devices=1 use_wandb=false \
      2>&1 | tail -6 || echo "!!!! FAILED $d"
done
echo "GEN_ALL_DONE"
