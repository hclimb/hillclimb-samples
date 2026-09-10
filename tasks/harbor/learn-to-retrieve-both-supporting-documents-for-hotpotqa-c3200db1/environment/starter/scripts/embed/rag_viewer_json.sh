#!/bin/bash
# Runs ON ONE slice worker: convert the rag_only output dir into a viewer-ready
# {metrics, samples} JSON (per-sample judge verdicts included) and upload beside the arms.
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
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
STEP="${STEP:-100000}"
NAME="${NAME:-msmarco_c10000_rag_top10_samples}"

pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

DEST="$HOME/${NAME}.json"
RAG_OUT_DIR="${RAG_OUT_DIR:-$HOME/rag_msmarco_c10000_top10}" DEST="$DEST" TP=4 \
  uv run --no-sync python scripts/misc/rag_to_viewer_json.py \
  || { echo "[rag-viewer-json] FAILED"; return 1 2>/dev/null || exit 1; }
gsutil cp "$DEST" "gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${NAME}.json" \
  && echo "[rag-viewer-json] -> gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${NAME}.json"
echo "[rag-viewer-json] DONE"
