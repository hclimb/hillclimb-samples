#!/bin/bash
# One-time (idempotent) box-side staging of the private ragrawal36/multihop_qa_sft-hard-neg-cot
# dataset -- see datagen/download_multihop_hardneg_cot.py's header. Run this (and
# stage_multihop_hardneg_data.sh, for the doc corpus) BEFORE any *_cot*/*_cot_ce_only training
# script on a fresh box.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

HF_HUB_OFFLINE=0 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
uv run python datagen/download_multihop_hardneg_cot.py "$@"
