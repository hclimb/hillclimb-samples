#!/bin/bash
# Row-balanced raw-parquet precache for qa_hard_neg_no_multihop_sft4b, sized around
# combined_hard_neg_sft4b's measured ceiling (578,097 post-filter rows -- see
# wiki/experiments/2026-08-10-ground-s1-zeroinit-4layer-no-multihop.md). No source sets a
# `weight`, so download_hf_data.py's target_rows_for split is uniform: TOTAL_ROWS/3 raw
# rows per source. Sizing TOTAL_ROWS off combined's own accept rate (578097/0.7608 ~=
# 759858, i.e. its full dataset) gives combined everything it has, and gives
# science_qa/diverse_qa (higher accept rates, ~86%/92%) comfortable post-filter headroom
# above 578,097 once preprocess_arrayrecord.py's --target-rows-per-source caps them there.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

uv run python data/download_hf_data.py \
  --dataset "${DATASET:-qa_hard_neg_no_multihop_sft4b}" \
  --total-rows "${TOTAL_ROWS:-2279574}" \
  --out "${GROUND_HF_PARQUET:-$HOME/hf_parquet}"
