#!/bin/bash
set -e

# Load .env
set -a; source "$(dirname "$0")/../../.env"; set +a

CHECKPOINT_DIR=gs://memory-layers-training/4B_pretraining_cot_unfreeze_all_topk_128_norm-2026-04-12-00-36-01/qwen3_mem_embed/100000

# Step 1: Prepare doc corpora (assigns integer IDs to each document)
echo "=== Preparing MuSiQue docs ==="
uv run python data/utils/prepare_musique_docs.py --split validation

echo "=== Preparing HotpotQA distractor docs ==="
uv run python data/utils/prepare_hotpotqa_docs.py --split validation

echo "=== Preparing MS MARCO docs ==="
uv run python data/utils/prepare_flashrag_docs.py --hf_config msmarco-qa --split dev

# Step 2: Augment QA datasets with pos_doc_ids referencing the corpora above.
# Must run after Step 1 so the corpus "id" column is available.
echo "=== Augmenting MuSiQue QA with corpus IDs ==="
uv run python data/utils/prepare_musique_qa_with_ids.py --split validation

echo "=== Augmenting HotpotQA QA with corpus IDs ==="
uv run python data/utils/prepare_hotpotqa_qa_with_ids.py --split validation

echo "=== Augmenting MS MARCO QA with corpus IDs ==="
uv run python data/utils/prepare_flashrag_msmarco_qa_with_ids.py --split dev

STEP=$(basename "${CHECKPOINT_DIR}")
LOG_FILE="results/corpus_eval_step${STEP}.log"
METRICS_FILE="results/corpus_eval_step${STEP}_metrics.json"
PLOT_OUT="results/corpus_eval_step${STEP}.png"

echo "=== Running eval ==="
uv run eval.py \
    checkpoint_dir=${CHECKPOINT_DIR} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=corpus_evals' \
    tp_devices=1 \
    2>&1 | tee "${LOG_FILE}"

echo "=== Extracting metrics ==="
uv run python - <<EOF
import json, sys
text = open("${LOG_FILE}").read()
idx = text.rfind('\n{')
if idx < 0:
    print("ERROR: no JSON found in log", file=sys.stderr); sys.exit(1)
metrics, _ = json.JSONDecoder().raw_decode(text[idx:].lstrip())
with open("${METRICS_FILE}", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved {len(metrics)} metrics -> ${METRICS_FILE}")
EOF

echo "=== Plotting ==="
uv run python analysis/plot_corpus_evals_single_run.py \
    --metrics "${METRICS_FILE}" \
    --title "Corpus Evals — step ${STEP}" \
    --out "${PLOT_OUT}"
