#!/bin/bash
# Runs ON a box. Downloads an epoch-sized, source-balanced parquet subset for the hard-neg
# (think) run: 100k steps x batch 16 = 1.6M rows, 400k per source, ~15 GB at 1.25x headroom.
# See wiki/data/epoch-sized-data.md.
#
# Deliberately does NOT set HF_HUB_OFFLINE — this script's whole job is to talk to the Hub.
# Training then reads the result with HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=<out>.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

OUT="${GROUND_HF_PARQUET:-$HOME/hf_parquet}"

echo "=== plan ==="
uv run python data/download_hf_data.py \
    --dataset qa_hard_neg_think_sft4b \
    --steps "${STEPS:-100000}" --batch-size "${BATCH_SIZE:-16}" \
    --headroom "${HEADROOM:-1.25}" \
    --select "${SELECT:-random}" --seed "${SEED:-42}" \
    --out "$OUT" \
    --plan || exit 1

echo
echo "=== download ==="
uv run python data/download_hf_data.py \
    --dataset qa_hard_neg_think_sft4b \
    --steps "${STEPS:-100000}" --batch-size "${BATCH_SIZE:-16}" \
    --headroom "${HEADROOM:-1.25}" \
    --select "${SELECT:-random}" --seed "${SEED:-42}" \
    --out "$OUT" || exit 1

echo
echo "=== on-disk result ==="
du -sh "$OUT" 2>/dev/null
du -sh "$OUT"/* 2>/dev/null
df -h / | tail -1
