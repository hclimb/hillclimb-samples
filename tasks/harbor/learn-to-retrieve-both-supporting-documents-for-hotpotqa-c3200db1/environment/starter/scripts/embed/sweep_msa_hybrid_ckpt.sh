#!/bin/bash
# Runs ON a box, unattended. Scores a LIST of checkpoints from ONE run-dir on the MSA hybrid
# eval for one dataset, so a box works a queue instead of needing a launch per point (mirrors
# sweep_ckpt_accuracy.sh's loop/resume/cleanup structure, but wraps eval_msa_hybrid.sh's exact
# invocation -- task=gen_large_mem_msa_<DS>_hybrid, auto-K, gather_bank, judge model/tp_size
# preset in the task config -- instead of a generic single-metric task).
#
#   DS=hotpotqa RUN_DIR=<run-dir basename> STEPS="500_1000_1500" \
#     bash scripts/embed/sweep_msa_hybrid_ckpt.sh
#
# Each point publishes to GCS + the training wandb run via scripts/misc/log_eval_to_wandb.py
# (which picks the newest-mtime local results file, so sequential points in one session never
# cross-attribute a stale result to the wrong step). Already-published steps are SKIPPED (checked
# via gsutil stat on the destination JSON), so the whole sweep is resumable -- a crash or
# preemption loses only the point in flight; just re-run with the same STEPS.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"
export MEM_APPROX_TOPK=1
export VLLM_TPU_LOCAL_ONLY=1

BUCKET="${CKPT_BUCKET:-memory-layers-training}"
DS="${DS:?set DS (msa dataset key, e.g. hotpotqa)}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the run-dir basename}"
# Underscores accepted + normalised to spaces -- multi-vm-tpu-run.sh's RUN_ENV splits KEY=VAL
# pairs on whitespace, so a space-separated list can't survive the trip to the box. Same
# convention as sweep_ckpt_accuracy.sh.
STEPS="${STEPS:?set STEPS, e.g. \"500_1000_1500\"}"
STEPS="${STEPS//_/ }"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
# Extra Hydra overrides, COMMA-separated (RUN_ENV forwarding cannot carry spaces in values).
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES//,/ }"

name="msa_${DS}_c10000_hybrid_autok${NAME_SUFFIX}"
echo "[sweep-msa-hybrid] run_dir=$RUN_DIR ds=$DS steps=[$STEPS] n=$NUM_SAMPLES name=$name"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

for step in $STEPS; do
  dst="gs://$BUCKET/$RUN_DIR/eval/step_${step}/${name}.json"
  if gsutil -q stat "$dst" 2>/dev/null; then
    echo "[sweep-msa-hybrid] step $step already done ($dst) — skipping"; continue
  fi
  echo "[sweep-msa-hybrid] ===== step $step ====="
  # A leftover judge from the previous point still holds /dev/vfio and the next eval dies on
  # device-busy AFTER loading the checkpoint (same gotcha sweep_ckpt_accuracy.sh guards against).
  pkill -f "[v]llm serve" 2>/dev/null || true
  sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
  sleep 3

  CKPT="gs://$BUCKET/$RUN_DIR/qwen3_mem_embed/$step"
  uv run --no-sync eval.py \
      checkpoint_dir="$CKPT" \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.msa_hybrid=gen_large_mem_msa_${DS}_hybrid" \
      "evals.msa_hybrid.eval.num_samples=$NUM_SAMPLES" \
      "evals.msa_hybrid.eval.max_new_tokens=$MAX_NEW_TOKENS" \
      "evals.msa_hybrid.eval.output_file=outputs/${name}.json" \
      $EXTRA_OVERRIDES
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[sweep-msa-hybrid] step $step FAILED (eval.py rc=$rc) — continuing to next"
    continue
  fi

  if find outputs -path "*eval_results*" -name "${name}.json" 2>/dev/null | grep -q .; then
    CKPT="$CKPT" EVAL_TASK="$name" \
    RESULTS_GLOB="outputs/**/eval_results/**/${name}.json" \
      uv run --no-sync python scripts/misc/log_eval_to_wandb.py \
      || echo "[sweep-msa-hybrid] step $step: log_eval_to_wandb failed"
  else
    echo "[sweep-msa-hybrid] step $step: no local results JSON — skipping upload"
  fi
done
echo "[sweep-msa-hybrid] DONE run_dir=$RUN_DIR steps=[$STEPS]"
