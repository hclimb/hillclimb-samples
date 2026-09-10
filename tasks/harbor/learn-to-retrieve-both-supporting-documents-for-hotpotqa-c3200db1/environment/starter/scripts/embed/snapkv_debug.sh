#!/bin/bash
# Runs ON a box (BOTH slice workers). Debug isolation for the newline-degeneration:
# arm A = identity compaction (comp=1: top-4096-of-4096 gather), arm B = no compaction at all.
# Coherent B + broken A => the gather corrupts the cache; both broken => base plumbing.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export PYTHONUNBUFFERED=1
find . -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

COMMON="--corpus ${HF_USERNAME}/musique-c512-rag-corpus --query_dataset ${HF_USERNAME}/msa-musique-qa-with-ids \
  --num_queries 1 --max_new 96 --comp 1 --max_corpus_tokens 7900"

echo "[dbg] === arm A: identity compaction ==="
uv run --no-sync python scripts/embed/snapkv_stream.py $COMMON --out outputs/snapkv_dbg_identity.json
echo "[dbg] === arm B: no compaction ==="
uv run --no-sync python scripts/embed/snapkv_stream.py $COMMON --no_compact --out outputs/snapkv_dbg_nocompact.json
echo "[dbg] DONE"
