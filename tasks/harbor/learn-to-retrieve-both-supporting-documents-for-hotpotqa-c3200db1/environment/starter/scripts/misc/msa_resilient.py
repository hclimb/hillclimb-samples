"""Resilient MSA training via tpunanny.

Keeps ONE spot TPU (rohun-v6e-8-0) alive and running MSA training, auto-recreating
+ resuming from the latest GCS checkpoint whenever the spot TPU is preempted.

The ssh_script is fully self-contained: a freshly-recreated VM is blank, so it embeds
the ADC + .env (base64), git-clones the pushed `msa-train` branch, sets up the venv,
discovers the latest checkpoint, and launches training detached.

Usage (from repo root, after `pip install -r ~/Code/repos/tpunanny/requirements.txt`):
    python scripts/embed/msa_resilient.py test 0   # run ssh_script ONCE on rohun-v6e-8-0 (debug)
    python scripts/embed/msa_resilient.py run       # babysit (blocking; run in background)
"""
import base64, os, subprocess, sys
from pathlib import Path

ZONE = "europe-west4-a"
PROJECT_ID = "memorylayers"
TPU_TYPE = "v6e-8"
RUN_NAME = "msa_resilient_topk16"
# Inherit the GCS identity from the environment (GCS_USER_EMAIL, set in .env).
GCS_EMAIL = os.environ.get("GCS_USER_EMAIL") or "rohunagrawal@gmail.com"

adc_path = Path(f"~/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json").expanduser()
env_path = Path("/Users/rohunagrawal/Code/repos/memory-layers/.env")
adc_b64 = base64.b64encode(adc_path.read_bytes()).decode()
env_b64 = base64.b64encode(env_path.read_bytes()).decode()

TRAIN_ARGS = (
    "model=qwen3_msa_train model.main_model.model_id=Qwen/Qwen3-4B model.msa.top_k_docs=16 "
    "dataset=qa_hard_neg_think_sft4b dataset.num_workers=0 trainer=staged_msa "
    "trainer.eval_interval=2000 trainer.checkpoint_interval=2000 "
    "eval_set@trainer.evals=msa_hard_neg_think_nll "
    f"+trainer.run_name={RUN_NAME}"
)

SSH_SCRIPT = f"""set -e
export DEBIAN_FRONTEND=noninteractive
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
. $HOME/.local/bin/env
mkdir -p $HOME/.config/gcloud/legacy_credentials/{GCS_EMAIL}
echo {adc_b64} | base64 -d > $HOME/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json
echo {env_b64} | base64 -d > $HOME/.env
set -a; . $HOME/.env; set +a
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT HF_TOKEN=$HF_TOKEN HYDRA_FULL_ERROR=1
# already training? leave it. (match only processes whose EXECUTABLE is python — not
# this ssh script's own bash, whose argv also contains "train.py".)
if pgrep -af 'train\\.py' | awk '{{print $2}}' | grep -qE 'python'; then echo ALREADY_TRAINING; exit 0; fi
rm -rf memory-layers
git clone -q -b msa-train https://$GIT_PAT@github.com/rohunagrawal/memory-layers
cd memory-layers
uv venv >/dev/null 2>&1; . .venv/bin/activate
uv pip install -q -e . >/dev/null 2>&1
uv pip install -q jax==0.8.0 >/dev/null 2>&1
wandb login $WANDB_API_KEY >/dev/null 2>&1
RESUME=$(PYTHONPATH=. .venv/bin/python scripts/embed/find_latest_ckpt.py {RUN_NAME} 2>/dev/null || true)
echo "RESUME_ARG=[$RESUME]"
setsid nohup .venv/bin/python train.py {TRAIN_ARGS} $RESUME > $HOME/msatrain.log 2>&1 < /dev/null &
sleep 8
echo "TRAIN_LAUNCHED pid $(pgrep -f 'python .*train\\.py' | head -1)"
"""


def _ssh(tpu_id, script):
    cmd = ["gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", tpu_id,
           "--tunnel-through-iap", f"--zone={ZONE}", f"--project={PROJECT_ID}",
           "--worker=all", f"--command={script}"]
    return subprocess.run(cmd)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "test":
        idx = sys.argv[2] if len(sys.argv) > 2 else "0"
        _ssh(f"rohun-{TPU_TYPE}-{idx}", SSH_SCRIPT)
    else:
        sys.path.insert(0, "/Users/rohunagrawal/Code/repos/tpunanny")
        import tpunanny as tn
        tn.babysit(idxs=range(1), tpu_type=TPU_TYPE, zone=ZONE,
                   project_id=PROJECT_ID, ssh_script=SSH_SCRIPT, startup_script=None)
