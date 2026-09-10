"""Resilient Pareto-study eval campaign via tpunanny.

Runs the memory-layer (qwen3_mem_embed) checkpoint on the 9 MSA evals, sharded
across 4 spot TPUs (rohun-v6e-8-{0..3}), auto-recreating + RESUMING on preemption.

Each TPU's ssh_script is fully self-contained (blank VM): embeds ADC + .env (base64),
clones `main`, builds venv, then runs its shard of datasets. Idempotent: every
finished eval's result JSON is uploaded to
    gs://memory-layers-training/pareto_eval/<method>/<dataset>.json
and a dataset is SKIPPED if that object already exists -> preemption-safe + collectible
from the Mac with no live SSH.

Shard = trailing index of `hostname` (= node id rohun-v6e-8-N).

Usage (repo root; needs `pip install -r ~/Code/repos/tpunanny/requirements.txt`):
    python scripts/misc/pareto_eval.py test 0    # run ssh_script ONCE on rohun-v6e-8-0 (debug)
    python scripts/misc/pareto_eval.py run        # babysit range(4) (blocking; run in background)
"""
import base64, os, subprocess, sys
from pathlib import Path

ZONE = "europe-west4-a"
PROJECT_ID = "memorylayers"
TPU_TYPE = "v6e-8"
# GCS identity: inherit from the environment (GCS_USER_EMAIL, as set in .env) so this agrees with
# utils.py::setup_gcs_credentials and the box run scripts. Override with GCS_USER_EMAIL=... .
GCS_EMAIL = os.environ.get("GCS_USER_EMAIL") or "rohunagrawal@gmail.com"
METHOD = "membed"
CKPT = "gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-04-19-20-17-51/qwen3_mem_embed/100000"
STEP = "100000"
RESULT_PREFIX = f"gs://memory-layers-training/pareto_eval/{METHOD}"

# 9 MSA datasets sharded over 4 VMs. Each shard leads with a SMALL corpus so every
# VM validates the path quickly before tackling big corpora (msmarco/triviaqa ~75k docs).
# alias (hydra key, can't start with digit) -> real dataset name.
# Focused set: 3 diverse small-corpus evals (single-hop popqa/nq + multi-hop hotpotqa),
# one per VM. (Full 9-eval sharding archived below.)
SHARDS = {
    0: [("popqa", "popqa")],
    1: [("natural_questions", "natural_questions")],
    2: [("hotpotqa", "hotpotqa")],
    3: [],
}
# Full 9-eval set (restore by swapping into SHARDS above):
#   0: popqa, msmarco_v1
#   1: natural_questions, triviaqa_10m
#   2: hotpotqa, narrativeqa, 2wikimultihopqa(alias d2wiki)
#   3: musique, dureader
ACTIVE_IDXS = [0, 1, 2]

adc_path = Path(f"~/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json").expanduser()
env_path = Path("/Users/rohunagrawal/Code/repos/memory-layers/.env")
adc_b64 = base64.b64encode(adc_path.read_bytes()).decode()
env_b64 = base64.b64encode(env_path.read_bytes()).decode()


def _shard_lines():
    """Bash assoc-style: emit `SHARD_<idx>="alias:dataset alias:dataset"` lines."""
    out = []
    for idx, pairs in SHARDS.items():
        spec = " ".join(f"{a}:{d}" for a, d in pairs)
        out.append(f'SHARD_{idx}="{spec}"')
    return "\n".join(out)


# Inner worker script (the actual setup + eval loop). Written to ~/run_shard.sh on the
# VM and launched DETACHED (setsid nohup) so the SSH returns immediately, the job
# survives disconnect/preemption-of-the-SSH, and progress is tailable at ~/pareto.log.
def build_worker(idx):
  WORKER = f"""set +e
export DEBIAN_FRONTEND=noninteractive
. $HOME/.local/bin/env
set -a; . $HOME/.env; set +a
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT HF_TOKEN=$HF_TOKEN HYDRA_FULL_ERROR=1

rm -rf memory-layers
git clone -q -b main https://$GIT_PAT@github.com/rohunagrawal/memory-layers
cd memory-layers
uv venv >/dev/null 2>&1; . .venv/bin/activate
uv pip install -q -e . >/dev/null 2>&1
uv pip install -q jax==0.8.0 >/dev/null 2>&1

IDX={idx}
{_shard_lines()}
eval "SPEC=\\$SHARD_$IDX"
echo "=== PARETO_EVAL VM idx=$IDX shard=[$SPEC] ==="

gpy() {{ PYTHONPATH=. .venv/bin/python "$@"; }}

for pair in $SPEC; do
  ALIAS="${{pair%%:*}}"; DS="${{pair##*:}}"
  DST="{RESULT_PREFIX}/$DS.json"
  echo "======== $DS (alias=$ALIAS) -> $DST ========"
  if gpy -c "import gcsfs,sys; sys.exit(0 if gcsfs.GCSFileSystem().exists('$DST') else 1)"; then
    echo "SKIP $DS (result exists)"; continue
  fi
  gpy data/utils/prepare_msa_docs.py --dataset "$DS" || true
  gpy data/utils/prepare_msa_qa_with_ids.py --dataset "$DS" || true
  OUT=$HOME/evalout/$DS; rm -rf "$OUT"
  gpy eval.py \\
    checkpoint_dir={CKPT} \\
    '~eval_set@evals=pretraining' \\
    "+eval/tasks@evals.$ALIAS=gen_large_mem_msa_$DS" \\
    "evals.$ALIAS.eval.type=generation_large_mem" \\
    "evals.$ALIAS.eval.doc_access_acc=false" \\
    tp_devices=1 use_wandb=false \\
    "hydra.run.dir=$OUT" 2>&1 | tail -60
  rc=${{PIPESTATUS[0]}}
  if [ $rc -ne 0 ]; then echo "!!!! FAILED $DS rc=$rc"; continue; fi
  RES=$(ls "$OUT"/eval_results/step_{STEP}/$ALIAS/outputs/*.json 2>/dev/null | head -1)
  if [ -z "$RES" ]; then echo "!!!! NO RESULT FILE for $DS"; continue; fi
  gpy -c "import gcsfs; gcsfs.GCSFileSystem().put('$RES','$DST'); print('UPLOADED $DST')"
done
echo "PARETO_EVAL_SHARD_DONE idx=$IDX"
"""
  return WORKER


# Outer bootstrap: install uv + creds, write per-idx WORKER, launch it DETACHED.
# Idempotent: exits fast if the worker is already running here (PID file).
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
echo {worker_b64} | base64 -d > $HOME/run_shard.sh
setsid nohup bash $HOME/run_shard.sh > $HOME/pareto.log 2>&1 < /dev/null &
echo $! > $HOME/worker.pid
sleep 2
echo "WORKER_LAUNCHED idx={idx} pid $(cat $HOME/worker.pid)"
"""


def _ssh(tpu_id, script):
    # plain gcloud (external IPs enabled on these VMs); no alpha/tunnel needed.
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_id,
           f"--zone={ZONE}", f"--project={PROJECT_ID}",
           "--worker=all", f"--command={script}"]
    return subprocess.run(cmd)


KILL_SCRIPT = (
    "if [ -f $HOME/worker.pid ]; then kill -- -$(cat $HOME/worker.pid) 2>/dev/null; "
    "kill $(cat $HOME/worker.pid) 2>/dev/null; rm -f $HOME/worker.pid; echo KILLED; "
    "else echo NO_PID; fi; "
    "pkill -f run_shard.sh 2>/dev/null; pkill -f '[e]val.py' 2>/dev/null; true"
)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode in ("test", "kick"):
        idxs = [int(sys.argv[2])] if len(sys.argv) > 2 else [0, 1, 2, 3]
        for idx in idxs:
            print(f"--- kick rohun-{TPU_TYPE}-{idx} ---")
            _ssh(f"rohun-{TPU_TYPE}-{idx}", build_ssh_script(idx))
    elif mode == "kill":
        idxs = [int(sys.argv[2])] if len(sys.argv) > 2 else [0, 1, 2, 3]
        for idx in idxs:
            print(f"--- kill rohun-{TPU_TYPE}-{idx} ---")
            _ssh(f"rohun-{TPU_TYPE}-{idx}", KILL_SCRIPT)
    else:
        # keepalive ONLY: recreate preempted TPUs. Worker launch is handled by the
        # `kick` rekick loop (per-idx shard injection), NOT by babysit's generic ssh.
        sys.path.insert(0, "/Users/rohunagrawal/Code/repos/tpunanny")
        import tpunanny as tn
        tn.babysit(idxs=range(4), tpu_type=TPU_TYPE, zone=ZONE,
                   project_id=PROJECT_ID, ssh_script=None, startup_script=None)
