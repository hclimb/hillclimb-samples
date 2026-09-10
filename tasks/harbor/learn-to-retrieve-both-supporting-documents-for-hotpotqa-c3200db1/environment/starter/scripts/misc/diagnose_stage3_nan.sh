#!/bin/bash
# Runs ON a box. A/B the stage-3 NaN against the ONLY functional config delta since the last
# known-good run (707a128, April): mem_approx_topk (exact top_k -> jax.lax.approx_max_k).
# Same checkpoint, same stage-3 config, same data order — one variable.
#   CKPT=gs://.../qwen3_mem_embed/16000 bash scripts/misc/diagnose_stage3_nan.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a
# Be explicit about GCS creds (the bench script hit the compute-SA fallback without this).
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT

CKPT="${CKPT:?set CKPT=gs://.../qwen3_mem_embed/<step>}"
N="${N_BATCHES:-3}"
# Offline data: the cached parquet is on this box; live-HF would livelock (wiki/data/hf-rate-limits.md)
export HF_HUB_OFFLINE=1
export GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}"

echo "############################################################"
echo "# ARM A: MEM_APPROX_TOPK=1  (current default -> approx_max_k)"
echo "############################################################"
MEM_APPROX_TOPK=1 PYTHONPATH=. uv run python scripts/misc/diagnose_stage3_nan.py "$CKPT" "$N" 2>&1 \
  | grep -viE "^WARNING:absl|Grain multiprocess|^(embed_model|main_model)\."

echo
echo "############################################################"
echo "# ARM B: MEM_APPROX_TOPK=0  (April behaviour -> exact top_k)"
echo "############################################################"
MEM_APPROX_TOPK=0 PYTHONPATH=. uv run python scripts/misc/diagnose_stage3_nan.py "$CKPT" "$N" 2>&1 \
  | grep -viE "^WARNING:absl|Grain multiprocess|^(embed_model|main_model)\."
