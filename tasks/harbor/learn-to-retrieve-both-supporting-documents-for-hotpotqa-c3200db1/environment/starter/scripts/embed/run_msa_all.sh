#!/bin/bash
# Run MSA-4B on each MSA benchmark dataset INDIVIDUALLY (isolates per-dataset
# failures / OOM on large corpora), with the LLM judge. Collect numbers from the
# per-dataset output JSONs afterwards.
set -a; source "$(dirname "$0")/../../.env"; set +a
. "$HOME/.local/bin/env" 2>/dev/null || true

DATASETS="${@:-popqa dureader narrativeqa 2wikimultihopqa hotpotqa musique natural_questions msmarco_v1 triviaqa_10m}"

for d in $DATASETS; do
  task="gen_large_mem_msa_${d}"
  echo "================ EVAL $d ================"
  uv run eval.py \
      model=qwen3_msa \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.${d}=${task}" \
      tp_devices=1 use_wandb=false \
      2>&1 | tail -8 || echo "!!!! FAILED $d"
done
echo "ALL_EVALS_DONE"
