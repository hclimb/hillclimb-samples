#!/bin/bash
# Runs ON a box (BOTH workers of the slice). SnapKV streaming baseline on MuSiQue c512:
# off-the-shelf Qwen3-4B, whole 512-doc corpus in context, per-query question-conditioned
# KV compression. See scripts/embed/snapkv_stream.py + the 2026-07-21 experiment page.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"
export PYTHONUNBUFFERED=1
# tar-preserved mtimes can leave stale .pyc winning over freshly synced sources
find . -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

COMP="${COMP:-8}"
NUM_QUERIES="${NUM_QUERIES:-128}"
MAX_NEW="${MAX_NEW:-1280}"
CORPUS="${CORPUS:-musique-c512-rag-corpus}"
OUT="${OUT:-outputs/snapkv_${CORPUS%%-rag-corpus}_comp${COMP}.json}"

uv run --no-sync python scripts/embed/snapkv_stream.py \
    --corpus "${HF_USERNAME}/${CORPUS}" \
    --query_dataset "${HF_USERNAME}/msa-musique-qa-with-ids" \
    --num_queries "$NUM_QUERIES" \
    --comp "$COMP" \
    --probe question \
    --max_new "$MAX_NEW" \
    --out "$OUT"
echo "[snapkv] done -> $OUT"
