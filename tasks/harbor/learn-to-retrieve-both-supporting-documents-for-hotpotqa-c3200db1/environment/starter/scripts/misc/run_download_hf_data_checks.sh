#!/bin/bash
# Runs ON a box. Unit-tests download_hf_data.py's planning math, then does a real --plan
# (network reads only, no shard downloads) for the hard-neg mix so the numbers can be checked
# against the plan the test asserts.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

echo "=============== 1. planning unit test ==============="
uv run python tests/test_download_hf_data.py || exit 1

echo
echo "=============== 2. real --plan (no downloads) ==============="
uv run python data/download_hf_data.py \
    --dataset qa_hard_neg_think_sft4b \
    --steps 100000 --batch-size 16 \
    --headroom 1.25 --select random --seed 42 \
    --out "$HOME/hf_parquet" \
    --plan
