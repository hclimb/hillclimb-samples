#!/bin/bash
# Runs ON a box, unattended. Scores a LIST of checkpoints on one eval task, so a box keeps
# working through a queue instead of needing a launch per point.
#
#   RUN_DIR=<run-dir basename> STEPS="500 1000 1500" \
#     bash scripts/embed/sweep_ckpt_accuracy.sh
#
# Each point publishes to GCS + the training wandb run via scripts/misc/log_eval_to_wandb.py, so
# a crash mid-sweep loses only the point in flight. Already-published steps are SKIPPED, making
# the whole script resumable — just re-run it.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

BUCKET="${CKPT_BUCKET:-memory-layers-training-usc1}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the run-dir basename}"
# Underscores are accepted as separators AND normalised to spaces: multi-vm-tpu-run.sh's RUN_ENV
# splits KEY=VAL pairs on whitespace, so a space-separated list cannot survive the trip to the box.
#   STEPS="500_1000_1500"  (via RUN_ENV)   or   STEPS="500 1000 1500"  (running directly)
STEPS="${STEPS:?set STEPS, e.g. \"500_1000_1500\"}"
STEPS="${STEPS//_/ }"
EVAL_TASK="${EVAL_TASK:-gen_large_mem_musique_c512}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"
TAG="${RUN_TAG:-}"
TP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]*$'); [ "$TP" -gt 0 ] 2>/dev/null || TP=4

echo "[sweep] run_dir=$RUN_DIR steps=[$STEPS] task=$EVAL_TASK n=$NUM_SAMPLES tp=$TP"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

for step in $STEPS; do
  name="${EVAL_TASK}${TAG:+-$TAG}"
  dst="gs://$BUCKET/$RUN_DIR/eval/step_${step}/${name}.json"
  if gsutil -q stat "$dst" 2>/dev/null; then
    echo "[sweep] step $step already done ($dst) — skipping"; continue
  fi
  echo "[sweep] ===== step $step ====="
  # A leftover judge from the previous point still holds /dev/vfio and the next eval dies on
  # device-busy AFTER loading the checkpoint.
  pkill -f "[v]llm serve" 2>/dev/null || true
  sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
  sleep 3

  CKPT="gs://$BUCKET/$RUN_DIR/qwen3_mem_embed/$step"
  uv run --no-sync eval.py \
      checkpoint_dir="$CKPT" \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.musique=$EVAL_TASK" \
      "evals.musique.eval.num_samples=$NUM_SAMPLES" \
      "evals.musique.eval.max_new_tokens=$MAX_NEW_TOKENS" \
      "+evals.musique.eval.metrics.llm_judge_accuracy.model_id=Qwen/Qwen3-4B" \
      "+evals.musique.eval.metrics.llm_judge_accuracy.tensor_parallel_size=$TP" \
      ${EXTRA_EVAL_OVERRIDES:-} \
    && CKPT="$CKPT" EVAL_TASK="$name" uv run --no-sync python scripts/misc/log_eval_to_wandb.py \
    || echo "[sweep] step $step FAILED — continuing to next"
done
echo "[sweep] DONE run_dir=$RUN_DIR steps=[$STEPS]"
