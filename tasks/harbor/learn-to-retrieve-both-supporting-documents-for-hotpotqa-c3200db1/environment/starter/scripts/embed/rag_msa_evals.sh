#!/bin/bash
set -e

# Load .env
set -a; source "$(dirname "$0")/../../.env"; set +a

CHECKPOINT_DIR=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-04-19-20-17-51/qwen3_mem_embed/100000

MSA_DATASETS=(
  msmarco_v1
  natural_questions
  narrativeqa
  2wikimultihopqa
  hotpotqa
  musique
  dureader
  popqa
  triviaqa_10m
)

echo "=== Preparing MSA doc corpora with IDs ==="
for dataset in "${MSA_DATASETS[@]}"; do
  echo "--- ${dataset} docs ---"
  uv run python data/utils/prepare_msa_docs.py --dataset "${dataset}"
done

echo "=== Preparing MSA QA datasets with pos_doc_ids ==="
for dataset in "${MSA_DATASETS[@]}"; do
  echo "--- ${dataset} QA ---"
  uv run python data/utils/prepare_msa_qa_with_ids.py --dataset "${dataset}"
done

STEP=$(basename "${CHECKPOINT_DIR}")
LOG_FILE="results/rag_msa_eval_step${STEP}.log"
METRICS_FILE="results/rag_msa_eval_step${STEP}_metrics.json"
PLOT_OUT="results/rag_msa_eval_step${STEP}.png"

# Shared RAG parameters across all MSA evals
# You can override these from the command line, e.g. bash scripts/embed/rag_msa_evals.sh rag.top_k=10
RAG_DEFAULTS="rag.top_k=5 rag.gen_model=Qwen/Qwen3-4B rag.embedding_model=Qwen/Qwen3-Embedding-0.6B rag.judge_model=Qwen/Qwen3-8B"

echo "=== Running RAG MSA evals ==="
uv run rag_eval.py \
    checkpoint_dir=${CHECKPOINT_DIR} \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=msa_evals' \
    tp_devices=1 \
    ${RAG_DEFAULTS} \
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
    --title "RAG MSA Evals — step ${STEP}" \
    --out "${PLOT_OUT}"
