#!/usr/bin/env bash
# Precache the 3 sources of qa_hard_neg_no_multihop_sft4b for indexed preprocessing,
# sized to balance around combined_hard_neg_sft4b's ceiling: 100% of combined (it's the
# smallest and caps the whole balanced set), and a modest fraction of science_qa/diverseqa
# (their accept rates are ~86%/92% vs combined's ~66%, so even 25% of their raw rows is
# far more than enough to match combined's post-filter count). Also pulls the Qwen3-4B
# tokenizer/weights, needed by data/preprocess_arrayrecord.py's tokenizer load.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/memory-layers"
source .venv/bin/activate 2>/dev/null || true

export GROUND_HF_PARQUET=${GROUND_HF_PARQUET:-$HOME/hf_parquet}
mkdir -p "$GROUND_HF_PARQUET"

dl () {
  local repo="$1" frac="$2"
  echo "=== $(date +%H:%M:%S) parquet: $repo (frac=$frac) ==="
  GROUND_DATA_FRAC="$frac" .venv/bin/python - "$repo" <<'PY'
import os, sys, math, time
from huggingface_hub import HfApi, hf_hub_download
repo = sys.argv[1]
tok  = os.environ.get("HF_TOKEN")
out  = os.path.join(os.environ["GROUND_HF_PARQUET"], repo.replace("/", "__"))
def _list():
    for a in range(12):
        try:
            return HfApi().list_repo_files(repo, repo_type="dataset", token=tok)
        except Exception as e:
            if "429" in str(e) and a < 11:
                time.sleep(min(30 * (a + 1), 240)); continue
            raise
files = sorted(f for f in _list() if f.endswith(".parquet"))
frac = float(os.environ.get("GROUND_DATA_FRAC", "1.0"))
sel = files[:max(1, math.ceil(len(files) * frac))]
print(f"{repo}: {len(sel)}/{len(files)} shards -> {out}", flush=True)
def get1(f):
    for a in range(10):
        try:
            hf_hub_download(repo, f, repo_type="dataset", local_dir=out, token=tok); return
        except Exception as e:
            if "429" in str(e) and a < 9:
                time.sleep(min(30 * (a + 1), 240)); continue
            raise
for f in sel:
    get1(f)
print("done", repo, flush=True)
PY
}

echo "=== $(date +%H:%M:%S) ensuring Qwen3-4B weights ==="
.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
tgt = os.path.expanduser("~/weights/huggingface/Qwen/Qwen3-4B")
if os.path.isdir(tgt):
    print("model exists")
else:
    snapshot_download(repo_id="Qwen/Qwen3-4B", local_dir=tgt, token=os.environ.get("HF_TOKEN"))
PY

dl vm2825/triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-sft4b 1.0
dl vm2825/science-qa-hard-neg-think 0.25
dl vm2825/diverseqa-hard-neg-think 0.25

echo "=== on-disk size ==="; du -sh "$GROUND_HF_PARQUET" 2>/dev/null | tail -1; df -h "$HOME" | tail -1
echo "DONE."
