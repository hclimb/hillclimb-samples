#!/bin/bash
# Eval qwen3_mem_embed checkpoint on ALL 9 MSA datasets in ONE eval.py (checkpoint
# loaded once; membed shards its mem bank natively so big corpora are fine).
set -a; source "$(dirname "$0")/../../.env"; set +a
. "$HOME/.local/bin/env" 2>/dev/null || true
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

CKPT=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-04-19-20-17-51/qwen3_mem_embed/100000

# task -> hydra eval key (keys can't start with a digit)
DS="popqa hotpotqa musique natural_questions 2wikimultihopqa narrativeqa dureader msmarco_v1 triviaqa_10m"
ARGS=()
for d in $DS; do
  k=$d; [ "$d" = "2wikimultihopqa" ] && k=wiki2hop
  ARGS+=( "+eval/tasks@evals.${k}=gen_large_mem_msa_${d}" )
  ARGS+=( "evals.${k}.eval.type=generation_large_mem" )
  ARGS+=( "evals.${k}.eval.doc_access_acc=false" )
  ARGS+=( "evals.${k}.eval.doc_dataset.max_docs=4000" )
  ARGS+=( "evals.${k}.eval.num_samples=128" )
  ARGS+=( "~evals.${k}.eval.metrics" )
done

uv run --no-sync eval.py \
    checkpoint_dir=${CKPT} \
    '~eval_set@evals=pretraining' \
    "${ARGS[@]}" \
    tp_devices=1 use_wandb=false
echo "MEMBED_ALL_DONE"
