#!/bin/bash
# One-time (idempotent) box-side weights staging for qwen3_mem_embed (configs/model/
# qwen3_mem_embed.yaml): main_model=Qwen/Qwen3-4B, embed_model=Qwen/Qwen3-Embedding-0.6B.
# models/qwen3.py::load() requires each snapshot to already exist under
# ~/weights/huggingface/<model_id> once HF_HUB_OFFLINE=1 (training scripts set that to avoid the
# HF rate-limit livelock -- see wiki/data/hf-rate-limits.md), and does not fetch it itself in
# offline mode. Update the MODEL_IDS list if a recipe's model config pulls in a value_model too.
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
for model_id in ("Qwen/Qwen3-4B", "Qwen/Qwen3-Embedding-0.6B"):
    tgt = os.path.expanduser(f"~/weights/huggingface/{model_id}")
    if os.path.isdir(tgt) and os.listdir(tgt):
        print("model exists:", tgt)
    else:
        snapshot_download(repo_id=model_id, local_dir=tgt, token=os.environ.get("HF_TOKEN"))
        print("staged:", tgt)
PY
