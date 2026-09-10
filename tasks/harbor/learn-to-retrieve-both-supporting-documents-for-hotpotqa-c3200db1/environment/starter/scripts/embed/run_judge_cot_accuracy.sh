#!/bin/bash
# Runs ON ONE worker: llm_judge_cot_accuracy over a list of results JSONs (INPUTS env).
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"
export VLLM_TPU_LOCAL_ONLY=1 TPU_SKIP_MDS_QUERY=1 TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3
INPUTS="${INPUTS:?set INPUTS}" JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}" TP="${TP:-4}" \
  uv run --no-sync python scripts/misc/judge_cot_accuracy.py
echo "[cot-accuracy] DONE"
