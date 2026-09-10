#!/bin/bash
# Runs ON a box: CPU-only test of the RAG->memory hybrid mask machinery.
# JAX_PLATFORMS=cpu skips TPU backend init entirely, so this is safe to run on a SINGLE
# worker of a multi-host slice (no peer-wait hang) and leaves the chips untouched.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"

JAX_PLATFORMS=cpu uv run --no-sync python tests/test_rag_hybrid_mask.py
