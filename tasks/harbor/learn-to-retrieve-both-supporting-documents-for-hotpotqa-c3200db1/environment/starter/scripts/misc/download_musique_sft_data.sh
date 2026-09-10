#!/bin/bash
# Runs ON a box. Stages the MuSiQue SFT parquets locally for offline training.
#
# musique_sft is ONE source of 13,153 rows, so unlike the hard-neg mix this is not really
# "epoch-sized" sizing — the whole dataset is ~250 MB and always fits. STEPS x BATCH_SIZE
# (1250 x 32 = 40,000) exceeds the 13,153 rows on purpose: with a single source, the
# interleave's stopping_strategy='all_exhausted' repetition IS the epoch loop (~3 epochs).
# The wiki's warning about a short source being "silently repeated" applies to MIXES, where
# repetition means one source is over-sampled relative to the others. See
# wiki/data/epoch-sized-data.md.
#
# Deliberately does NOT set HF_HUB_OFFLINE — this script's job is to talk to the Hub.
# Training then reads the result with HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=<out>.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

OUT="${GROUND_HF_PARQUET:-$HOME/hf_parquet_musique}"

echo "=== plan ==="
uv run python data/download_hf_data.py \
    --dataset musique_sft \
    --steps "${STEPS:-1250}" --batch-size "${BATCH_SIZE:-32}" \
    --headroom "${HEADROOM:-1.25}" \
    --select "${SELECT:-random}" --seed "${SEED:-42}" \
    --out "$OUT" \
    --plan || exit 1

echo
echo "=== download ==="
uv run python data/download_hf_data.py \
    --dataset musique_sft \
    --steps "${STEPS:-1250}" --batch-size "${BATCH_SIZE:-32}" \
    --headroom "${HEADROOM:-1.25}" \
    --select "${SELECT:-random}" --seed "${SEED:-42}" \
    --out "$OUT" || exit 1

echo
echo "=== on-disk result ==="
du -sh "$OUT" 2>/dev/null
du -sh "$OUT"/* 2>/dev/null
df -h / | tail -1
