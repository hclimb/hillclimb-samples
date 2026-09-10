#!/bin/bash
# Runs ON a box. Stages the MuSiQue SFT parquets locally for offline training.
#
# ragrawal36/multihop_qa_sft has 1,341,045 train rows (~5.4 GB parquet), so unlike the MuSiQue
# staging script this genuinely IS epoch-sized sizing: STEPS x BATCH_SIZE (10,000 x 32 = 320,000
# rows, ~1.6 GB at 1.25x headroom) is a QUARTER of one epoch, and download_hf_data.py picks a
# random shard subset rather than pulling the whole repo. No repetition, so none of the
# "short source silently repeated" concern applies. See wiki/data/epoch-sized-data.md.
#
# Deliberately does NOT set HF_HUB_OFFLINE — this script's job is to talk to the Hub.
# Training then reads the result with HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=<out>.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

OUT="${GROUND_HF_PARQUET:-$HOME/hf_parquet_multihop}"

echo "=== plan ==="
uv run python data/download_hf_data.py \
    --dataset multihop_qa_sft_midtraining \
    --steps "${STEPS:-10000}" --batch-size "${BATCH_SIZE:-32}" \
    --headroom "${HEADROOM:-1.25}" \
    --select "${SELECT:-random}" --seed "${SEED:-42}" \
    --out "$OUT" \
    --plan || exit 1

echo
echo "=== download ==="
uv run python data/download_hf_data.py \
    --dataset multihop_qa_sft_midtraining \
    --steps "${STEPS:-10000}" --batch-size "${BATCH_SIZE:-32}" \
    --headroom "${HEADROOM:-1.25}" \
    --select "${SELECT:-random}" --seed "${SEED:-42}" \
    --out "$OUT" || exit 1

echo
echo "=== on-disk result ==="
du -sh "$OUT" 2>/dev/null
du -sh "$OUT"/* 2>/dev/null
df -h / | tail -1
