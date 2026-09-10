#!/bin/bash
# Runs ON ONE worker of the v6e slice (or any single box), unattended. Classic-RAG baseline on
# the msmarco c10000 EVAL-MATCHED corpus: build the corpus+queries from a hybrid eval run's own
# record (datagen/msmarco/build_msmarco_rag_corpus_from_eval.py — exact same haystack and same
# 128 queries as the memory arms, no rows[:N] survivor mismatch), then retrieve -> generate ->
# judge via scripts/misc/rag_only.py.
#
#   RUN_DIR=<run-dir> [STEP=100000] [TARGET_DOCS=10000] [RAG_K=10] \
#     bash scripts/embed/eval_msmarco_rag.sh
#
# SINGLE WORKER ONLY: launch on ONE host. The standalone-TPU env below confines every stage
# (JAX retrieval encoder + vLLM gen/judge) to this host's 4 chips — without it, libtpu reads
# the slice topology from metadata and blocks waiting for the peer host (runbook §2.3).
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"
export VLLM_TPU_LOCAL_ONLY=1
export TPU_SKIP_MDS_QUERY=1
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1

BUCKET="${CKPT_BUCKET:-memory-layers-training}"
RUN_DIR="${RUN_DIR:?set RUN_DIR (the run whose eval dir holds the reference JSON and gets the result)}"
STEP="${STEP:-100000}"
TARGET_DOCS="${TARGET_DOCS:-10000}"
RAG_K="${RAG_K:-10}"
NUM_QUERIES="${NUM_QUERIES:-128}"
# Any hybrid arm's JSON works as the reference — it only needs the prompts in eval order.
EVAL_JSON_NAME="${EVAL_JSON_NAME:-msmarco_c10000_hybrid_k10_docs.json}"

name="msmarco_c${TARGET_DOCS}_rag_top${RAG_K}"
dst="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}.json"
if gsutil -q stat "$dst" 2>/dev/null; then
  echo "[msmarco-rag] $name done — skipping"; return 0 2>/dev/null || exit 0
fi
echo "[msmarco-rag] k=$RAG_K target_docs=$TARGET_DOCS n=$NUM_QUERIES -> $dst"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

# 1. Matched corpus + queries from the eval's own record (idempotent Hub overwrite).
ref="/tmp/msmarco_rag_eval_ref.json"
gsutil cp "gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${EVAL_JSON_NAME}" "$ref"
corpus_repo="${HF_USERNAME}/msmarco-c${TARGET_DOCS}-rag-corpus-evalmatched"
queries_repo="${HF_USERNAME}/msmarco-c${TARGET_DOCS}-eval-queries"
uv run --no-sync python datagen/msmarco/build_msmarco_rag_corpus_from_eval.py \
    --eval-json "$ref" --target-docs "$TARGET_DOCS" \
    --out-corpus-repo "$corpus_repo" --out-queries-repo "$queries_repo" 2>&1 | tail -4 \
  || { echo "[msmarco-rag] corpus build FAILED"; return 1 2>/dev/null || exit 1; }

pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

# 2. retrieve -> generate -> judge (same three stages as rag_eval.py's rag_accuracy).
OUT_DIR="$HOME/rag_msmarco_c${TARGET_DOCS}_top${RAG_K}"
RAG_CORPUS="$corpus_repo" QUERY_DS="$queries_repo" NUM_QUERIES="$NUM_QUERIES" \
MAX_DOCS="$TARGET_DOCS" TOP_K="$RAG_K" GEN_TOP_K_DOCS="$RAG_K" TP=4 OUT_DIR="$OUT_DIR" \
  uv run --no-sync python scripts/misc/rag_only.py 2>&1 | tail -10 \
  || { echo "[msmarco-rag] rag_only FAILED"; return 1 2>/dev/null || exit 1; }

if [ -f "$OUT_DIR/rag_only_summary.json" ]; then
  gsutil cp "$OUT_DIR/rag_only_summary.json" "$dst" && echo "[msmarco-rag] -> $dst"
else
  echo "[msmarco-rag] no summary produced"; return 1 2>/dev/null || exit 1
fi
echo "[msmarco-rag] DONE"
