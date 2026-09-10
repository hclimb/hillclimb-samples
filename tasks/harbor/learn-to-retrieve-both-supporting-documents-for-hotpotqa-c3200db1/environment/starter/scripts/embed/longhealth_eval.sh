#!/bin/bash
# LongHealth on the memory layer. THROUGHPUT + RETRIEVAL are the deliverables; generation
# accuracy is OUT OF DISTRIBUTION (no checkpoint trained on documents this long) and must be
# labelled as such wherever it is reported.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

BUCKET="${CKPT_BUCKET:-memory-layers-training-usc1}"
RUN_DIR="${RUN_DIR:-musique_ground4layer_midtrain_topk128_seq1024_chunks20_bs16-2026-07-19-00-52-42}"
STEP="${STEP:-1500}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
TP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]*$'); [ "$TP" -gt 0 ] 2>/dev/null || TP=4
CKPT="gs://$BUCKET/$RUN_DIR/qwen3_mem_embed/$STEP"

echo "[longhealth] ckpt=$CKPT n=$NUM_SAMPLES tp=$TP"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0
pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

uv run --no-sync eval.py \
    checkpoint_dir="$CKPT" \
    '~eval_set@evals=pretraining' \
    "+eval/tasks@evals.longhealth=gen_large_mem_longhealth" \
    "evals.longhealth.eval.num_samples=$NUM_SAMPLES" \
    "+evals.longhealth.eval.metrics.llm_judge_accuracy.model_id=Qwen/Qwen3-4B" \
    "+evals.longhealth.eval.metrics.llm_judge_accuracy.tensor_parallel_size=$TP" \
  && CKPT="$CKPT" EVAL_TASK="longhealth" uv run --no-sync python scripts/misc/log_eval_to_wandb.py \
  || echo "[longhealth] FAILED"
echo "[longhealth] DONE"
