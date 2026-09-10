#!/bin/bash
# Runs ON a box, unattended. Same MSA dataset RAG->memory hybrid protocol as
# eval_msa_hybrid.sh (per-query gathered banks, auto-K, judge accuracy + 1-5 score), but with
# eval.multi_sample=true and dataset.batch_size=1 (tiled mode) so every query's mesh-tiled
# replicas become k INDEPENDENT temperature-sampled completions instead of one greedy answer —
# a GRPO-readiness diagnostic (pass@k, intra-group reward variance), not a quality benchmark.
#
# GROUP SIZE = the box's mesh data-axis size (all chips, since tp_devices defaults to 1 for
# this model), i.e. GROUP SIZE IS A BOX CHOICE, not a script knob: a single-host v6e-4
# (ct6e-standard-4t) gives k=4 samples/query, a v6e-8 gives k=8. Cost scales with total samples
# generated (num_samples x k), not just num_samples — this is k times the generation work of
# eval_msa_hybrid.sh at the same NUM_SAMPLES, not a free variant of it.
#
# See wiki/evaluation/evaluator-types.md's "multi_sample: true" section and
# wiki/implementations/2026-08-24-grpo-readiness-multisample-hybrid-eval.md.
#
#   DS=<dataset> RUN_DIR=<run-dir> STEP=<n> [NUM_SAMPLES=128] [TEMPERATURE=0.6] \
#     [TOP_K=20] [TOP_P=0.95] [NAME_SUFFIX=] \
#     bash scripts/embed/eval_msa_hybrid_multisample.sh
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
# Non-greedy decode is the whole point here — these mirror the force_thinking generation
# defaults used elsewhere in the repo (configs/eval/generation_embed.yaml), not eval_msa_hybrid.sh's
# greedy default (temperature=0.0), which would make every tiled replica decode identically.
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_K="${TOP_K:-20}"
TOP_P="${TOP_P:-0.95}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES//,/ }"
CKPT="gs://$BUCKET/$RUN_DIR/qwen3_mem_embed/$STEP"
name="msa_${DS}_c10000_hybrid_multisample${NAME_SUFFIX}"
dst="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}.json"

if gsutil -q stat "$dst" 2>/dev/null; then echo "[msa-hybrid-multisample] $name done — skipping"; return 0 2>/dev/null || exit 0; fi
echo "[msa-hybrid-multisample] ds=$DS ckpt=$CKPT n=$NUM_SAMPLES max_new=$MAX_NEW_TOKENS temp=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P suffix='$NAME_SUFFIX'"
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
    "evals.msa_hybrid.eval.multi_sample=true" \
    "evals.msa_hybrid.eval.temperature=$TEMPERATURE" \
    "evals.msa_hybrid.eval.top_k=$TOP_K" \
    "evals.msa_hybrid.eval.top_p=$TOP_P" \
    "evals.msa_hybrid.dataset.batch_size=1" \
    "evals.msa_hybrid.eval.output_file=outputs/${name}.json" \
    $EXTRA_OVERRIDES \
  || { echo "[msa-hybrid-multisample] $name FAILED"; return 1 2>/dev/null || exit 1; }

if find outputs -path "*eval_results*" -name "${name}.json" 2>/dev/null | grep -q .; then
  CKPT="$CKPT" EVAL_TASK="$name" \
  RESULTS_GLOB="outputs/**/eval_results/**/${name}.json" \
    uv run --no-sync python scripts/misc/log_eval_to_wandb.py \
    || echo "[msa-hybrid-multisample] $name: log_eval_to_wandb failed"
else
  echo "[msa-hybrid-multisample] $name: no local results JSON (secondary slice host) — skipping upload"
fi
echo "[msa-hybrid-multisample] DONE"
