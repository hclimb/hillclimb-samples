#!/bin/bash
# Runs ON a box, unattended. One MSA dataset's PLAIN (non-hybrid, no RAG assist) full-corpus
# memory eval: the model builds its own memory bank over the whole doc corpus and retrieves via
# its trained mem_lookup, no per-query RAG-gathered bank. Mirrors ground_eval_box.py's
# "_corpus" task recipe (gen_large_mem_msa_<DS>.yaml, eval.type overridden generation_large_mem_msa
# -> generation_large_mem since that MSA type wants an 'msa' cfg block our memory-layer model
# lacks), wired for a one-shot checkpoint the way eval_msa_hybrid.sh is.
#
#   DS=<dataset> RUN_DIR=<run-dir> STEP=<n> [NUM_SAMPLES=128] [NAME_SUFFIX=] \
#     bash scripts/embed/eval_msa_plain.sh
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"
export MEM_APPROX_TOPK=1
export VLLM_TPU_LOCAL_ONLY=1

BUCKET="${CKPT_BUCKET:-memory-layers-training}"
DS="${DS:?set DS (msa dataset key, e.g. musique)}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
STEP="${STEP:?set STEP}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES//,/ }"
CKPT="gs://$BUCKET/$RUN_DIR/qwen3_mem_embed/$STEP"
name="msa_${DS}_corpus${NAME_SUFFIX}"
dst="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}.json"

if gsutil -q stat "$dst" 2>/dev/null; then echo "[msa-plain] $name done — skipping"; return 0 2>/dev/null || exit 0; fi
echo "[msa-plain] ds=$DS ckpt=$CKPT n=$NUM_SAMPLES suffix='$NAME_SUFFIX'"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

uv run --no-sync eval.py \
    checkpoint_dir="$CKPT" \
    '~eval_set@evals=pretraining' \
    "+eval/tasks@evals.msa_plain=gen_large_mem_msa_${DS}" \
    "evals.msa_plain.eval.type=generation_large_mem" \
    "evals.msa_plain.eval.num_samples=$NUM_SAMPLES" \
    "evals.msa_plain.dataset.batch_size=8" \
    "evals.msa_plain.eval.output_file=outputs/${name}.json" \
    tp_devices=1 \
    $EXTRA_OVERRIDES \
  || { echo "[msa-plain] $name FAILED"; return 1 2>/dev/null || exit 1; }

if find outputs -path "*eval_results*" -name "${name}.json" 2>/dev/null | grep -q .; then
  CKPT="$CKPT" EVAL_TASK="$name" \
  RESULTS_GLOB="outputs/**/eval_results/**/${name}.json" \
    uv run --no-sync python scripts/misc/log_eval_to_wandb.py \
    || echo "[msa-plain] $name: log_eval_to_wandb failed"
else
  echo "[msa-plain] $name: no local results JSON (secondary slice host) — skipping upload"
fi
echo "[msa-plain] DONE"
