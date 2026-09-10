#!/bin/bash
# One-time (idempotent) box-side data staging for multihop_hard_neg_full training scripts --
# see datagen/download_multihop_hardneg.py's header. Run this BEFORE any
# train_multihop_ground*.sh script on a fresh box: those scripts set HF_HUB_OFFLINE=1 and
# read local parquet + the doc corpus only, they do not stage data themselves.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
uv run python datagen/download_multihop_hardneg.py
