#!/bin/bash
# Runs ON ONE worker, unattended. One MSA dataset's classic-RAG@10 baseline on the
# EVAL-MATCHED corpus (built from the dataset's hybrid arm JSON), then the viewer-schema
# per-sample JSON with judge accuracy + 1-5 score + rag_docs. Mirrors eval_msmarco_rag.sh
# + rag_viewer_json.sh, folded into one per-dataset step for the overnight sweep driver.
#
#   DS=<dataset> RUN_DIR=<run-dir> STEP=<n> [RAG_K=10] bash scripts/embed/eval_msa_rag.sh
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"
# Standalone-host TPU env: every stage (JAX encoder + vLLM) confined to this host's chips.
export VLLM_TPU_LOCAL_ONLY=1
export TPU_SKIP_MDS_QUERY=1
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1

BUCKET="${CKPT_BUCKET:-memory-layers-training}"
DS="${DS:?set DS}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
STEP="${STEP:-100000}"
TARGET_DOCS="${TARGET_DOCS:-10000}"
RAG_K="${RAG_K:-10}"
NUM_QUERIES="${NUM_QUERIES:-128}"
repo_ds="${DS//_/-}"                       # natural_questions -> natural-questions etc.
# Non-MSA datasets (e.g. DS=qasper -> qasper-chunks-*) override the derived repo names.
qa_repo="${QA_REPO:-${HF_USERNAME}/msa-${repo_ds}-qa-with-ids}"
doc_repo="${DOC_REPO:-${HF_USERNAME}/msa-${repo_ds}-docs-with-ids}"
corpus_repo="${HF_USERNAME}/msa-${repo_ds}-c${TARGET_DOCS}-rag-corpus-evalmatched"
queries_repo="${HF_USERNAME}/msa-${repo_ds}-c${TARGET_DOCS}-eval-queries"
hyb_name="msa_${DS}_c10000_hybrid_autok"
name="msa_${DS}_c10000_rag_top${RAG_K}"
dst_sum="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}.json"
dst_sam="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}_samples.json"

if gsutil -q stat "$dst_sam" 2>/dev/null; then echo "[msa-rag] $name done — skipping"; return 0 2>/dev/null || exit 0; fi
echo "[msa-rag] ds=$DS k=$RAG_K n=$NUM_QUERIES -> $dst_sam"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

ref="/tmp/msa_${DS}_rag_eval_ref.json"
gsutil cp "gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${hyb_name}.json" "$ref" \
  || { echo "[msa-rag] $DS: hybrid reference JSON missing"; return 1 2>/dev/null || exit 1; }
uv run --no-sync python datagen/msmarco/build_msmarco_rag_corpus_from_eval.py \
    --eval-json "$ref" --target-docs "$TARGET_DOCS" \
    --qa-repo "$qa_repo" --doc-repo "$doc_repo" \
    --out-corpus-repo "$corpus_repo" --out-queries-repo "$queries_repo" 2>&1 | tail -4 \
  || { echo "[msa-rag] corpus build FAILED"; return 1 2>/dev/null || exit 1; }

pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

OUT_DIR="$HOME/rag_msa_${DS}_c${TARGET_DOCS}_top${RAG_K}"
RAG_CORPUS="$corpus_repo" QUERY_DS="$queries_repo" NUM_QUERIES="$NUM_QUERIES" \
MAX_DOCS="$TARGET_DOCS" TOP_K="$RAG_K" GEN_TOP_K_DOCS="$RAG_K" TP=4 OUT_DIR="$OUT_DIR" \
  uv run --no-sync python scripts/misc/rag_only.py 2>&1 | tail -8 \
  || { echo "[msa-rag] rag_only FAILED"; return 1 2>/dev/null || exit 1; }
[ -f "$OUT_DIR/rag_only_summary.json" ] && gsutil cp "$OUT_DIR/rag_only_summary.json" "$dst_sum"

pkill -f "[v]llm serve" 2>/dev/null || true
sleep 3
DEST="$HOME/${name}_samples.json"
RAG_OUT_DIR="$OUT_DIR" QUERIES_REPO="$queries_repo" CORPUS_REPO="$corpus_repo" \
DEST="$DEST" TP=4 \
  uv run --no-sync python scripts/misc/rag_to_viewer_json.py \
  || { echo "[msa-rag] viewer-json FAILED"; return 1 2>/dev/null || exit 1; }
gsutil cp "$DEST" "$dst_sam" && echo "[msa-rag] -> $dst_sam"
echo "[msa-rag] DONE"
