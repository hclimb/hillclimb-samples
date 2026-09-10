#!/bin/bash
# Baseline: eval the original qwen3_mem_embed checkpoint (separate 0.6B embed model
# + memory layer) on the SAME 9 MSA datasets via the repo's native gen_large_mem
# evaluator, for direct comparison vs MSA-4B. Gen only (no judge), isolated per dataset.
set -a; source "$(dirname "$0")/../../.env"; set +a
. "$HOME/.local/bin/env" 2>/dev/null || true
# resolve_train_cfg reads the GCS checkpoint .hydra before setup_gcs_credentials
# runs, so export the ADC up front (GCS_USER_EMAIL=ra3440 has bucket access).
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

CKPT=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-04-19-20-17-51/qwen3_mem_embed/100000

# Hydra keys can't start with a digit -> alias.
declare -A KEY=( [2wikimultihopqa]=wiki2hop )

DATASETS="${@:-popqa hotpotqa musique natural_questions 2wikimultihopqa narrativeqa dureader msmarco_v1 triviaqa_10m}"

for d in $DATASETS; do
  k=${KEY[$d]:-$d}
  echo "================ MEMBED $d ================"
  uv run --no-sync eval.py \
      checkpoint_dir=${CKPT} \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.${k}=gen_large_mem_msa_${d}" \
      "evals.${k}.eval.type=generation_large_mem" \
      "evals.${k}.eval.doc_access_acc=false" \
      "evals.${k}.eval.doc_dataset.max_docs=4000" \
      "evals.${k}.eval.num_samples=128" \
      "~evals.${k}.eval.metrics" \
      tp_devices=1 use_wandb=false \
      2>&1 | tail -6 || echo "!!!! FAILED $d"
done
echo "MEMBED_GEN_DONE"
