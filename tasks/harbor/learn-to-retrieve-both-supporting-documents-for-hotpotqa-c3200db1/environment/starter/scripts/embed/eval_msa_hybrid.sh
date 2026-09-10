#!/bin/bash
# Runs ON a box (BOTH slice workers), unattended. One MSA dataset's RAG->memory hybrid arm
# at c10000 (or full corpus when smaller): per-query gathered banks, B=8 data-parallel,
# auto-K (smallest k with all_golds >= 0.96, cap 200), judge accuracy + 1-5 score,
# rag_docs saved per sample. Mirrors eval_msmarco_hybrid.sh; task configs are
# gen_large_mem_msa_<DS>_hybrid.yaml.
#
#   DS=<dataset> RUN_DIR=<run-dir> STEP=<n> [NUM_SAMPLES=128] [NAME_SUFFIX=] \
#     bash scripts/embed/eval_msa_hybrid.sh
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
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
STEP="${STEP:?set STEP}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
# Extra Hydra overrides, COMMA-separated (RUN_ENV forwarding cannot carry spaces in values),
# e.g. EXTRA_OVERRIDES=evals.msa_hybrid.eval.rag.top_k=100,evals.msa_hybrid.eval.rag.auto_k_threshold=null
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES//,/ }"
CKPT="gs://$BUCKET/$RUN_DIR/qwen3_mem_embed/$STEP"
name="msa_${DS}_c10000_hybrid_autok${NAME_SUFFIX}"
dst="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}.json"

if gsutil -q stat "$dst" 2>/dev/null; then echo "[msa-hybrid] $name done — skipping"; return 0 2>/dev/null || exit 0; fi
echo "[msa-hybrid] ds=$DS ckpt=$CKPT n=$NUM_SAMPLES max_new=$MAX_NEW_TOKENS suffix='$NAME_SUFFIX'"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

uv run --no-sync eval.py \
    checkpoint_dir="$CKPT" \
    '~eval_set@evals=pretraining' \
    "+eval/tasks@evals.msa_hybrid=gen_large_mem_msa_${DS}_hybrid" \
    "evals.msa_hybrid.eval.num_samples=$NUM_SAMPLES" \
    "evals.msa_hybrid.eval.max_new_tokens=$MAX_NEW_TOKENS" \
    "evals.msa_hybrid.eval.output_file=outputs/${name}.json" \
    $EXTRA_OVERRIDES \
  || { echo "[msa-hybrid] $name FAILED"; return 1 2>/dev/null || exit 1; }

if find outputs -path "*eval_results*" -name "${name}.json" 2>/dev/null | grep -q .; then
  CKPT="$CKPT" EVAL_TASK="$name" \
  RESULTS_GLOB="outputs/**/eval_results/**/${name}.json" \
    uv run --no-sync python scripts/misc/log_eval_to_wandb.py \
    || echo "[msa-hybrid] $name: log_eval_to_wandb failed"
else
  echo "[msa-hybrid] $name: no local results JSON (secondary slice host) — skipping upload"
fi
echo "[msa-hybrid] DONE"
