#!/bin/bash
set -e

# Load .env
set -a; source "$(dirname "$0")/../../.env"; set +a

DEFAULT_CHECKPOINT_DIR=gs://memory-layers-training/4B_pretraining_cot_unfreeze_all_topk_128_norm-2026-04-12-00-36-01/qwen3_mem_embed/100000
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"

# Prepare doc datasets (idempotent HF pushes)
echo "=== Preparing MuSiQue docs ==="
uv run python data/utils/prepare_musique_docs.py --split validation

echo "=== Preparing HotpotQA distractor docs ==="
uv run python data/utils/prepare_hotpotqa_docs.py --split validation

echo "=== Preparing MS MARCO docs ==="
uv run python data/utils/prepare_flashrag_docs.py --hf_config msmarco-qa --split dev

STEP=$(basename "${CHECKPOINT_DIR}")
RUN_LABEL="${RUN_LABEL:-step ${STEP}}"
SAFE_RUN_LABEL=$(printf '%s' "${RUN_LABEL}" | tr ' /' '__')
LOG_FILE="results/rag_corpus_eval_${SAFE_RUN_LABEL}.log"
METRICS_FILE="results/rag_corpus_eval_${SAFE_RUN_LABEL}_metrics.json"
PLOT_OUT="results/rag_corpus_eval_${SAFE_RUN_LABEL}.png"

echo "=== Running eval ==="
uv run rag_eval.py \
    checkpoint_dir=${CHECKPOINT_DIR} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=corpus_evals' \
    tp_devices=1 \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "=== Extracting metrics ==="
uv run python - <<EOF
import json, sys
text = open("${LOG_FILE}").read()
marker = "All Evaluation Results:\n"
idx = text.rfind(marker)
if idx < 0:
    print("ERROR: could not find metrics in log", file=sys.stderr); sys.exit(1)
metrics, _ = json.JSONDecoder().raw_decode(text[idx + len(marker):].lstrip())
with open("${METRICS_FILE}", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved {len(metrics)} metrics -> ${METRICS_FILE}")
EOF

echo "=== Plotting ==="
uv run python analysis/plot_corpus_evals_single_run.py \
    --metrics "${METRICS_FILE}" \
    --title "RAG Corpus Evals — ${RUN_LABEL}" \
    --out "${PLOT_OUT}"
