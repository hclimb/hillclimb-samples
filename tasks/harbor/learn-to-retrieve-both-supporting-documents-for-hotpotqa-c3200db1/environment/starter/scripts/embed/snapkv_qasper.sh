#!/bin/bash
# Runs ON a box (BOTH slice workers). SnapKV-inspired streaming compression on QASPER:
# the full ~6.3M-token corpus streamed into ONE query-agnostic compressed cache (segment-tail
# scoring, --probe self), then answerable dev questions answered against it. At this scale the
# compression is necessarily extreme (~comp=256: keep 16 of every 4096 tokens) — that IS the
# honest cost of context-stuffing QASPER; label the variant SnapKV-inspired, not faithful.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export PYTHONUNBUFFERED=1
find . -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

COMP="${COMP:-256}"
NUM_QUERIES="${NUM_QUERIES:-64}"
MAX_NEW="${MAX_NEW:-512}"
OUT="${OUT:-outputs/snapkv_qasper_comp${COMP}.json}"

uv run --no-sync python scripts/embed/snapkv_stream.py \
    --corpus_format qasper \
    --num_queries "$NUM_QUERIES" \
    --comp "$COMP" \
    --probe self \
    --max_new "$MAX_NEW" \
    --out "$OUT"
echo "[snapkv] done -> $OUT"
