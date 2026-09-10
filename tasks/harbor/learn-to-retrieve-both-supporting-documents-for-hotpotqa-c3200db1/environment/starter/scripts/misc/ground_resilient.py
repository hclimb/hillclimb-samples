"""Resilient 4-TPU launcher for the grounding-exps 4-doc experiments, via tpunanny.

Recreates rohun-v6e-8-{0..3} as spot TPUs and keeps them alive, auto-resuming on preemption.
The ssh_script is self-contained (blank recreated VM): embeds ADC + .env (base64), clones the
pushed `grounding-exps` branch, builds the venv, then DISPATCHES by TPU index:
  idx 0 -> ground_grp4_freeze_from_iso1_20k (main frozen)     resume from latest ckpt
  idx 1 -> ground_grp4_noda_from_iso1_20k   (full-FT no d_a)   resume from latest ckpt
  idx 2,3 -> eval boxes: set up env only, then idle (I SSH evals in manually)

Usage (from repo root, after `pip install -r ~/Code/repos/tpunanny/requirements.txt`):
    python scripts/misc/ground_resilient.py test 0   # run ssh_script ONCE on rohun-v6e-8-0 (debug)
    python scripts/misc/ground_resilient.py run       # babysit all 4 (blocking; run in background)
"""
import base64, os, subprocess, sys
from pathlib import Path

ZONE = "europe-west4-a"
PROJECT_ID = "memorylayers"
TPU_TYPE = "v6e-8"
# Inherit the GCS identity from the environment (GCS_USER_EMAIL, set in .env) so this agrees with
# utils.py::setup_gcs_credentials and the box run scripts instead of pinning a second opinion.
GCS_EMAIL = os.environ.get("GCS_USER_EMAIL") or "rohunagrawal@gmail.com"

adc_path = Path(f"~/.config/gcloud/legacy_credentials/{GCS_EMAIL}/adc.json").expanduser()
env_path = Path("/Users/rohunagrawal/Code/repos/memory-layers/.env")
adc_b64 = base64.b64encode(adc_path.read_bytes()).decode()
env_b64 = base64.b64encode(env_path.read_bytes()).decode()

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
IDX=$(hostname | grep -oE '[0-9]+$')
# idempotent setup: build venv only if missing
if [ ! -f $HOME/memory-layers/.venv/bin/activate ]; then
  rm -rf $HOME/memory-layers
  git clone -q -b grounding-exps https://$GIT_PAT@github.com/rohunagrawal/memory-layers $HOME/memory-layers
  cd $HOME/memory-layers
  uv venv >/dev/null 2>&1; . .venv/bin/activate
  uv pip install -q -e . >/dev/null 2>&1
else
  cd $HOME/memory-layers; . .venv/bin/activate
fi
wandb login $WANDB_API_KEY >/dev/null 2>&1 || true
# SETUP-ONLY babysitter: a TPU worker VM cannot self-identify its box index (hostname is
# t1v-n-<hash>-w-0), so per-box run dispatch is done by the external monitor loop, which knows
# each box's index. Here we only guarantee the box is env-ready + has the offline-HF parquet so
# training (launched externally) issues ZERO HF calls. Idempotent; safe to re-run on preemption.
if [ ! -d $HOME/hf_parquet/vm2825__science-qa-hard-neg-think ]; then
  echo "PRECACHING"; bash scripts/misc/precache_hf.sh > $HOME/precache.log 2>&1 || true
fi
echo "BOX_READY host=$(hostname)"; exit 0
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
        tn.babysit(idxs=range(5), tpu_type=TPU_TYPE, zone=ZONE,
                   project_id=PROJECT_ID, ssh_script=SSH_SCRIPT, startup_script=None)
