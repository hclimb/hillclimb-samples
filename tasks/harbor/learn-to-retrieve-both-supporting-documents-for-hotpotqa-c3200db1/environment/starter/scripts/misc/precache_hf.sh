#!/usr/bin/env bash
# Pre-cache a SUBSET of each training dataset's parquet shards onto the BOOT DISK (durable —
# unlike /dev/shm, which a periodic cleanup wipes mid-run on these VMs), plus the model
# checkpoints. Training then reads the local parquet directly (HF_HUB_OFFLINE=1, qa.py's offline
# branch: load_dataset("parquet", data_files=..., streaming=True)) => ZERO HF API calls (no 429),
# no arrow-build doubling, and only ~half the data so it fits the ~67G boot disk.
#
# Selection: GROUND_DATA_SHARDS=<N> takes the first N shards per repo (use N=2 for a smoke);
#            else GROUND_DATA_FRAC (default 0.5) takes that fraction. Re-run per box (fast — no
#            re-download of shards already present).
#
# Usage:  GROUND_DATA_SHARDS=2 bash scripts/misc/precache_hf.sh     # smoke
#         bash scripts/misc/precache_hf.sh                          # half, for real runs
#         MODELS_ONLY=1 bash scripts/misc/precache_hf.sh             # model weights only
set -uo pipefail
set -a; . "$HOME/.env"; set +a                 # HF_TOKEN, HF_USERNAME
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/memory-layers"
source .venv/bin/activate 2>/dev/null || true

# Boot-disk location the offline branch in data/qa.py reads (keep these two in sync).
export GROUND_HF_PARQUET=${GROUND_HF_PARQUET:-$HOME/hf_parquet}
mkdir -p "$GROUND_HF_PARQUET"

dl () {
  echo "=== $(date +%H:%M:%S) parquet subset: $1 ==="
  .venv/bin/python - "$1" <<'PY'
import os, sys, math, time
from huggingface_hub import HfApi, hf_hub_download
repo = sys.argv[1]
tok  = os.environ.get("HF_TOKEN")
out  = os.path.join(os.environ["GROUND_HF_PARQUET"], repo.replace("/", "__"))
def _list():                                  # list_repo_files also trips the 429 quota; back off
    for a in range(12):
        try:
            return HfApi().list_repo_files(repo, repo_type="dataset", token=tok)
        except Exception as e:
            if "429" in str(e) and a < 11:
                time.sleep(min(30 * (a + 1), 240)); continue
            raise
files = sorted(f for f in _list() if f.endswith(".parquet"))
n = os.environ.get("GROUND_DATA_SHARDS")
sel = files[:int(n)] if n else files[:max(1, math.ceil(len(files) * float(os.environ.get("GROUND_DATA_FRAC", "0.5"))))]
print(f"{repo}: {len(sel)}/{len(files)} shards -> {out}", flush=True)
import time
def get1(f):
    for a in range(10):                       # back off on 429 so concurrent per-box precaches
        try:                                  # (4 boxes at once) don't hard-fail on the quota
            hf_hub_download(repo, f, repo_type="dataset", local_dir=out, token=tok); return
        except Exception as e:
            if "429" in str(e) and a < 9:
                time.sleep(min(30 * (a + 1), 240)); continue
            raise
for f in sel:
    get1(f)                                   # hf_hub_download skips files already on disk => resumable
print("done", repo, flush=True)
PY
}

# --- models: qwen3.load fetches each to ~/weights/huggingface/<id> only if missing, so under
#     HF_HUB_OFFLINE a NEW model (the Stage-2 value model Qwen3-0.6B-Base) would fail. Pre-pull. ---
echo "=== $(date +%H:%M:%S) ensuring model checkpoints ==="
.venv/bin/python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
# Qwen3-0.6B-Base is the OPTIONAL Stage-2 value model; not used by these runs. Keep it last and
# non-fatal. All wrapped in 429-backoff (concurrent per-box precaches trip the quota).
for m in ["Qwen/Qwen3-4B", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-0.6B-Base"]:
    tgt = os.path.expanduser("~/weights/huggingface/" + m)
    if os.path.isdir(tgt):
        print("model exists:", m, flush=True); continue
    print("downloading model:", m, flush=True)
    for a in range(12):
        try:
            snapshot_download(repo_id=m, local_dir=tgt, token=os.environ.get("HF_TOKEN")); break
        except Exception as e:
            if "429" in str(e) and a < 11:
                time.sleep(min(30 * (a + 1), 240)); continue
            print("WARN model dl failed (non-fatal):", m, str(e)[:120], flush=True); break
PY

# --- training datasets (subset of parquet shards) ---
# MODELS_ONLY=1 skips these. Useful when a run's data is staged separately (e.g.
# scripts/misc/download_musique_sft_data.sh) but the box still needs the model weights on disk,
# because HF_HUB_OFFLINE=1 blocks the snapshot_download that qwen3.load would otherwise do.
if [ "${MODELS_ONLY:-0}" = "1" ]; then
  echo "=== MODELS_ONLY=1: skipping dataset shards ==="
else
dl vm2825/science-qa-hard-neg-think
dl vm2825/diverseqa-hard-neg-think
dl vm2825/triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-sft4b
dl ragrawal36/multihop_qa_sft
fi

echo "=== on-disk size ==="; du -sh "$GROUND_HF_PARQUET" 2>/dev/null | tail -1; df -h "$HOME" | tail -1
echo "DONE. Launch with HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$GROUND_HF_PARQUET"
