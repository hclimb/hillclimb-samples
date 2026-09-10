#!/bin/bash
# Runs ON a box. RAG baseline ALONE on the MuSiQue 512-doc corpus — no memory model, no
# checkpoint, no JAX eval worker. See scripts/misc/rag_only.py for why that is a separate path
# from rag_eval.py (which always runs the memory eval first).
#
#   bash scripts/embed/rag_only_musique_c512.sh
#
# Corpus must be the matched one from datagen/musique/build_musique_c512_rag_corpus.py
# (512 docs = 293 gold + 219 distractor, all golds present) so the number is comparable to the
# memory model's on the same haystack.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"

echo "[rag_only] pinning vllm HTTP server deps..."
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0
pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

uv run --no-sync python scripts/misc/rag_only.py
