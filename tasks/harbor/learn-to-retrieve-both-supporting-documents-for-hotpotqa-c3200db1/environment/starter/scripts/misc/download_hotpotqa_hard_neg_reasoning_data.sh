#!/bin/bash
# Runs ON a box. Stages the vm2825/hotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1
# parquets locally for offline training (78,755 rows, ~156 MB on the Hub).
#
# One source, same as musique_sft's download script: unlike the hard-neg mix this is not
# "epoch-sized" balancing across sources, so STEPS x BATCH_SIZE exceeding the row count is
# expected -- the interleave's stopping_strategy='all_exhausted' repetition IS the epoch loop.
# See wiki/data/epoch-sized-data.md.
#
# Deliberately does NOT set HF_HUB_OFFLINE — this script's job is to talk to the Hub.
# Training then reads the result with HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=<out>.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

OUT="${GROUND_HF_PARQUET:-$HOME/hf_parquet_hotpotqa_hardneg_reasoning}"

echo "=== plan ==="
uv run python data/download_hf_data.py \
    --dataset hotpotqa_hard_neg_reasoning_modified_finetune \
    --steps "${STEPS:-10000}" --batch-size "${BATCH_SIZE:-16}" \
    --headroom "${HEADROOM:-1.25}" \
    --select "${SELECT:-random}" --seed "${SEED:-42}" \
    --out "$OUT" \
    --plan || exit 1

echo
echo "=== download ==="
uv run python data/download_hf_data.py \
    --dataset hotpotqa_hard_neg_reasoning_modified_finetune \
    --steps "${STEPS:-10000}" --batch-size "${BATCH_SIZE:-16}" \
    --headroom "${HEADROOM:-1.25}" \
    --select "${SELECT:-random}" --seed "${SEED:-42}" \
    --out "$OUT" || exit 1

echo
echo "=== on-disk result ==="
du -sh "$OUT" 2>/dev/null
du -sh "$OUT"/* 2>/dev/null
df -h / | tail -1
