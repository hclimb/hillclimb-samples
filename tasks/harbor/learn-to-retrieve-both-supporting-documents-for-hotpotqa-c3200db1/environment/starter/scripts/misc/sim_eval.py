"""End-of-night MSA-eval subset for the simpair campaign.

For a given idx: kills the training worker on that VM, evaluates the run's LATEST
checkpoint on 3 MSA datasets (popqa, natural_questions, hotpotqa; 64 samples each,
binary LLM-judge scoring), uploads each result JSON to
    gs://memory-layers-training/simpair_eval/<run_name>/step<STEP>/<dataset>.json
(idempotent skip if exists), then optionally restarts training via sim_train.py kick.

Usage:
    python scripts/misc/sim_eval.py kick [idx]     # launch eval worker on VM(s)
    python scripts/misc/sim_eval.py collect        # print results from GCS (no SSH)
"""
import base64, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sim_train import (RUNS, VM_NAMES, ZONE, PROJECT_ID, GCS_EMAIL, BRANCH,
                       adc_b64, env_b64, KILL_SCRIPT, _ssh)

EVAL_DATASETS = [("popqa", "popqa"), ("natural_questions", "natural_questions"),
                 ("hotpotqa", "hotpotqa")]
import os
NUM_SAMPLES = int(os.environ.get("SIM_EVAL_SAMPLES", "64"))
RESULT_PREFIX = "gs://memory-layers-training/simpair_eval"


def build_eval_worker(idx, step_pin=None):
    run_name, _ = RUNS[idx]
    pairs = " ".join(f"{a}:{d}" for a, d in EVAL_DATASETS)
    # Pin a specific step (e.g. matched-step comparisons) instead of the latest.
    step_override = f"STEP={step_pin}" if step_pin else ""
    return f"""set +e
export DEBIAN_FRONTEND=noninteractive
. $HOME/.local/bin/env
set -a; . $HOME/.env; set +a
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT HYDRA_FULL_ERROR=1
cd $HOME/memory-layers
git fetch -q origin "+{BRANCH}:refs/remotes/origin/{BRANCH}"
git checkout -q -B {BRANCH} origin/{BRANCH}; git reset -q --hard origin/{BRANCH}
. .venv/bin/activate
gpy() {{ PYTHONPATH=. .venv/bin/python "$@"; }}

RUN={run_name}
INFO=$(gpy scripts/misc/sim_find_latest.py $RUN)
STEP=$(echo $INFO | awk '{{print $1}}')
CKPT_DIR=$(echo $INFO | awk '{{print $3}}')
if [ -z "$CKPT_DIR" ] || [ "$STEP" = "-1" ]; then echo "NO_CKPT for $RUN"; exit 1; fi
{step_override}
CKPT=$CKPT_DIR/$STEP
echo "=== SIM_EVAL idx={idx} run=$RUN ckpt=$CKPT ==="

for pair in {pairs}; do
  ALIAS="${{pair%%:*}}"; DS="${{pair##*:}}"
  DST="{RESULT_PREFIX}/$RUN/step$STEP/$DS.json"
  echo "======== $DS -> $DST ========"
  if gpy -c "import gcsfs,sys; sys.exit(0 if gcsfs.GCSFileSystem().exists('$DST') else 1)"; then
    echo "SKIP $DS (exists)"; continue
  fi
  OUT=$HOME/evalout/$DS; rm -rf "$OUT"
  gpy eval.py \\
    checkpoint_dir=$CKPT \\
    '~eval_set@evals=pretraining' \\
    "+eval/tasks@evals.$ALIAS=gen_large_mem_msa_$DS" \\
    "evals.$ALIAS.eval.type=generation_large_mem" \\
    "evals.$ALIAS.eval.num_samples={NUM_SAMPLES}" \\
    "evals.$ALIAS.eval.doc_access_acc=false" \\
    tp_devices=1 use_wandb=false \\
    "hydra.run.dir=$OUT" 2>&1 | tail -30
  rc=${{PIPESTATUS[0]}}
  if [ $rc -ne 0 ]; then echo "!!!! FAILED $DS rc=$rc"; continue; fi
  RES=$(ls "$OUT"/eval_results/step_$STEP/$ALIAS/outputs/*.json 2>/dev/null | head -1)
  if [ -z "$RES" ]; then echo "!!!! NO RESULT FILE for $DS"; continue; fi
  gpy -c "import gcsfs; gcsfs.GCSFileSystem().put('$RES','$DST'); print('UPLOADED $DST')"
done
echo "SIM_EVAL_DONE idx={idx} run=$RUN step=$STEP"
"""


def build_eval_ssh(idx, step_pin=None):
    worker_b64 = base64.b64encode(build_eval_worker(idx, step_pin).encode()).decode()
    return f"""set +e
# stop training first (frees the TPU)
{KILL_SCRIPT}
sleep 5
echo {worker_b64} | base64 -d > $HOME/run_eval.sh
setsid nohup bash $HOME/run_eval.sh > $HOME/eval.log 2>&1 < /dev/null &
echo $! > $HOME/worker.pid
sleep 2
echo "EVAL_LAUNCHED idx={idx} pid $(cat $HOME/worker.pid)"
"""


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "kick"
    if mode == "kick":
        idxs = [int(sys.argv[2])] if len(sys.argv) > 2 else list(RUNS)
        step_pin = int(sys.argv[3]) if len(sys.argv) > 3 else None
        for idx in idxs:
            print(f"--- eval kick {VM_NAMES[idx]} ({RUNS[idx][0]}) step_pin={step_pin} ---")
            _ssh(VM_NAMES[idx], build_eval_ssh(idx, step_pin))
    elif mode == "collect":
        import json, os, subprocess
        out = subprocess.run(["gsutil", "ls", "-r", RESULT_PREFIX], capture_output=True, text=True,
                             env={**os.environ, "CLOUDSDK_CORE_ACCOUNT": GCS_EMAIL})
        paths = [l for l in out.stdout.splitlines() if l.endswith(".json")]
        for p in paths:
            body = subprocess.run(["gsutil", "cat", p], capture_output=True, text=True,
                                  env={**os.environ, "CLOUDSDK_CORE_ACCOUNT": GCS_EMAIL}).stdout
            try:
                d = json.loads(body)
                m = d.get("metrics", d)
                acc = m.get("llm_judge_accuracy", m.get("llm_judge_binary_accuracy"))
                print(f"{p.split(RESULT_PREFIX+'/')[-1]}: llm_judge_accuracy={acc}")
            except Exception as e:
                print(f"{p}: parse error {e}")
