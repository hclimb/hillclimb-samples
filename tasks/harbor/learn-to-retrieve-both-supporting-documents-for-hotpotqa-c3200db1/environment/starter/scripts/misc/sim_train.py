"""Resilient text-pair-similarity training campaign via tpunanny.

5 from-scratch training runs of qwen3_mem_embed (Qwen3-4B + memory layer) with
different QA/similarity data mixtures, one per TPU:

    idx 0 rohun-v6e-8-0    simpair_ctrl     dataset=qa_hard_neg_think_sft4b (control)
    idx 1 rohun-v6e-8-1    simpair_sim15    dataset=qa_sim15    (15% similarity)
    idx 2 rohun-v6e-8-2    simpair_sim40    dataset=qa_sim40    (40% similarity)
    idx 3 rohun-v6e-8-3    simpair_sim15sym dataset=qa_sim15_sym (15%, symmetrized)
    idx 4 rohun-v6e-8-b-1  simpair_curr     curriculum: qa_simheavy until step 15k,
                                            then qa_hard_neg_think_sft4b (stage 3)

Preemption-safe: every (re)kick finds the latest ckpt across run-dir generations
(sim_find_latest.py) and relaunches with a FULL resume (+trainer.resume_from=<dir>,
no step suffix) and a fresh dataloader shuffle seed (42 + n_restarts).
Idempotent: bootstrap exits fast if the worker is already running (PID file).

Usage (repo root; tpunanny venv for `run`):
    python scripts/misc/sim_train.py kick [idx]   # (re)launch worker on VM(s)
    python scripts/misc/sim_train.py kill [idx]   # kill worker + training
    python scripts/misc/sim_train.py run          # babysit idxs 0-3 (blocking; bg it)
                                                  # (b-1 recreation handled manually)
"""
import base64, os, subprocess, sys
from pathlib import Path

ZONE = "europe-west4-a"
PROJECT_ID = "memorylayers"
TPU_TYPE = "v6e-8"
# Inherit the GCS identity from the environment (GCS_USER_EMAIL, set in .env). sim_eval.py imports
# GCS_EMAIL from here, so both stay on whatever .env says.
GCS_EMAIL = os.environ.get("GCS_USER_EMAIL") or "rohunagrawal@gmail.com"
BRANCH = "sim-pairs"

# v3 no-interrupt set: 4 training TPUs (0-3); b-1 = dedicated eval box (sim_eval_box.py).
# In-training evals disabled (eval_interval huge) so runs never pause; b-1 evaluates saved
# GCS checkpoints at matched steps. ALL FOUR ARE FRESH FROM SCRATCH at stage-3 peak LR 5e-5
# (STAGE3_LR below) — new names so none resume the older 1e-5 (v2) checkpoints. Runs differ
# only in data mixture + CE-masking, so sim40 (CE-on) vs sim40nce (CE-masked) is a clean A/B.
STAGE3_LR = "5e-5"
RUNS = {
    0: ("simpair_ctrl_v3", "qa_hard_neg_think_sft4b"),  # baseline
    1: ("simpair_sim40_v3", "qa_sim40"),                # CE-on 40%
    2: ("simpair_sim40nce_v3", "qa_sim40_nce"),         # CE-masked 40%
    3: ("simpair_sim15nce_v3", "qa_sim15_nce"),         # CE-masked 15%
}
VM_NAMES = {i: f"rohun-{TPU_TYPE}-{i}" for i in range(4)}

adc_path = Path(f"~/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json").expanduser()
env_path = Path("/Users/rohunagrawal/Code/repos/memory-layers/.env")
adc_b64 = base64.b64encode(adc_path.read_bytes()).decode()
env_b64 = base64.b64encode(env_path.read_bytes()).decode()


def build_worker(idx):
    run_name, dataset = RUNS[idx]
    ds_logic = f'DS={dataset}'
    return f"""set +e
export DEBIAN_FRONTEND=noninteractive
. $HOME/.local/bin/env
set -a; . $HOME/.env; set +a
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT HYDRA_FULL_ERROR=1

cd $HOME
if [ ! -d memory-layers/.git ]; then
  rm -rf memory-layers
  git clone -q -b {BRANCH} https://$GIT_PAT@github.com/rohunagrawal/memory-layers
fi
cd memory-layers
git remote set-url origin https://$GIT_PAT@github.com/rohunagrawal/memory-layers
git fetch -q origin "+{BRANCH}:refs/remotes/origin/{BRANCH}"
git checkout -q -B {BRANCH} origin/{BRANCH}
git reset -q --hard origin/{BRANCH}
uv venv >/dev/null 2>&1; . .venv/bin/activate
uv pip install -q -e . > $HOME/pipinstall.log 2>&1

RUN={run_name}
# Stagger dataset-resolution bursts across VMs (shared HF API quota).
sleep $(( (RANDOM % 60) + {idx} * 45 ))

ATTEMPT=0
while [ $ATTEMPT -lt 10 ]; do
  ATTEMPT=$((ATTEMPT+1))
  INFO=$(PYTHONPATH=. .venv/bin/python scripts/misc/sim_find_latest.py $RUN)
  STEP=$(echo $INFO | awk '{{print $1}}')
  NDIRS=$(echo $INFO | awk '{{print $2}}')
  CKPT=$(echo $INFO | awk '{{print $3}}')
  # Fixed seed: dataloader state is now checkpointed and fast-forward resume
  # requires the SAME shuffled stream across restarts. (Pre-loader-state runs
  # bumped the seed per restart to avoid replaying the same prefix.)
  SEED=42
  RESUME=""
  if [ -n "$CKPT" ]; then RESUME="+trainer.resume_from=$CKPT"; fi
  {ds_logic}
  echo "=== SIM_TRAIN idx={idx} run=$RUN attempt=$ATTEMPT latest_step=$STEP ndirs=$NDIRS seed=$SEED ds=$DS resume=[$RESUME] ==="
  PYTHONPATH=. .venv/bin/python train.py \\
    model=qwen3_mem_embed \\
    model.main_model.model_id="Qwen/Qwen3-4B" \\
    model.memory.mem_top_k=64 \\
    dataset=$DS \\
    trainer=staged_sim \\
    eval_set@trainer.evals=qa_hard_neg_think_sft4b \\
    +trainer.run_name="$RUN" \\
    dataset.shuffle_seed=$SEED \\
    dataset.num_workers=4 \\
    trainer.eval_interval=100000000 \\
    trainer.training_stages.3.learning_rate={STAGE3_LR} \\
    $RESUME
  RC=$?
  echo "SIM_TRAIN_EXITED idx={idx} rc=$RC attempt=$ATTEMPT"
  if [ $RC -eq 0 ]; then break; fi
  sleep $(( 120 + RANDOM % 180 ))
done
"""


def build_ssh_script(idx):
    worker_b64 = base64.b64encode(build_worker(idx).encode()).decode()
    return f"""set -e
export DEBIAN_FRONTEND=noninteractive
if [ -f $HOME/worker.pid ] && kill -0 $(cat $HOME/worker.pid) 2>/dev/null; then
  echo WORKER_ALREADY_RUNNING pid $(cat $HOME/worker.pid); exit 0; fi
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
. $HOME/.local/bin/env
mkdir -p $HOME/.config/gcloud/legacy_credentials/{GCS_EMAIL}
echo {adc_b64} | base64 -d > $HOME/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json
echo {env_b64} | base64 -d > $HOME/.env
echo {worker_b64} | base64 -d > $HOME/run_train.sh
setsid nohup bash $HOME/run_train.sh > $HOME/train.log 2>&1 < /dev/null &
echo $! > $HOME/worker.pid
sleep 2
echo "WORKER_LAUNCHED idx={idx} pid $(cat $HOME/worker.pid)"
"""


def _ssh(tpu_id, script):
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_id,
           f"--zone={ZONE}", f"--project={PROJECT_ID}",
           "--worker=all", f"--command={script}"]
    return subprocess.run(cmd)


KILL_SCRIPT = (
    "if [ -f $HOME/worker.pid ]; then kill -- -$(cat $HOME/worker.pid) 2>/dev/null; "
    "kill $(cat $HOME/worker.pid) 2>/dev/null; rm -f $HOME/worker.pid; echo KILLED; "
    "else echo NO_PID; fi; "
    # group kill alone has been observed to miss train.py — kill it by PID pattern
    # ([.]venv avoids matching this SSH command string itself)
    'PIDS=$(pgrep -f "[.]venv/bin/python train.py"); '
    'if [ -n "$PIDS" ]; then kill $PIDS; sleep 3; kill -9 $PIDS 2>/dev/null; echo "KILLED_TRAIN $PIDS"; fi; true'
)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "kick"
    if mode in ("test", "kick"):
        idxs = [int(sys.argv[2])] if len(sys.argv) > 2 else list(RUNS)
        for idx in idxs:
            print(f"--- kick {VM_NAMES[idx]} ({RUNS[idx][0]}) ---")
            _ssh(VM_NAMES[idx], build_ssh_script(idx))
    elif mode == "kill":
        idxs = [int(sys.argv[2])] if len(sys.argv) > 2 else list(RUNS)
        for idx in idxs:
            print(f"--- kill {VM_NAMES[idx]} ---")
            _ssh(VM_NAMES[idx], KILL_SCRIPT)
    else:
        # keepalive ONLY (recreate preempted TPUs 0-3); workers launched via `kick`.
        sys.path.insert(0, "/Users/rohunagrawal/Code/repos/tpunanny")
        import tpunanny as tn
        tn.babysit(idxs=range(4), tpu_type=TPU_TYPE, zone=ZONE,
                   project_id=PROJECT_ID, ssh_script=None, startup_script=None)
