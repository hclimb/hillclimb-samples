#!/bin/bash
# Run one cell of the dolci32k global-vs-layerwise architecture axis.
#
# The main dolci32k campaign fixes five layer-wise AdamW methods, so the
# comparison has no "curated data, but one global subset" control. This array
# supplies it: 5 settings x {GlobalRaw, GlobalOptA} = 10 runs, written into the
# same campaign root under the same pinned artifact build, so the loss-curve
# and analysis tooling reads them alongside the original 25.
#
# The immutable ADAMW_METHODS/MUON_METHODS registry is deliberately NOT
# extended — that would shift the 0-24 / 0-39 array contracts of the main
# campaign. Only the setting order is read from it.

#SBATCH --job-name=dolci-global-axis
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --array=0-9%4
#SBATCH --output=logs/dolci32k_global_axis_%A_%a.out

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID must be exported by the launcher}"
SEED="${DRPT_SEED:-42}"
WANDB_PROJECT="${DRPT_WANDB_PROJECT:-drpt_opus}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
ARTIFACT_BUILD_ID="${DRPT_ARTIFACT_BUILD_ID:?global-axis worker requires DRPT_ARTIFACT_BUILD_ID}"
MODEL_PROFILE="${DRPT_MODEL_PROFILE:?global-axis worker requires DRPT_MODEL_PROFILE}"

methods=(GlobalRaw GlobalOptA)

# Setting order comes from the canonical registry so this array can never
# disagree with the campaign it extends.
registry_python="${DRPT_PYTHON:-python}"
if ! settings_payload="$(
    cd "$REPO_ROOT" &&
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
        "$registry_python" -c '
import importlib
module = importlib.import_module("SFT.data.dolci32k.profile")
print("\t".join(module.SETTING_ORDER))
'
)"; then
    echo "ERROR: failed to load the canonical dolci32k setting registry" >&2
    exit 2
fi
IFS=$'\t' read -r -a settings <<< "$settings_payload"
if (( ${#settings[@]} != 5 )); then
    echo "ERROR: dolci32k registry must define exactly 5 settings, got ${#settings[@]}" >&2
    exit 2
fi

task_count=$(( ${#settings[@]} * ${#methods[@]} ))
if (( TASK_ID < 0 || TASK_ID >= task_count )); then
    echo "ERROR: array task must be in [0, $((task_count - 1))], got $TASK_ID" >&2
    exit 2
fi

setting="${settings[$((TASK_ID / ${#methods[@]}))]}"
method="${methods[$((TASK_ID % ${#methods[@]}))]}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TF32="${DRPT_TF32:-True}"
if [[ "$TF32" != "True" && "$TF32" != "False" ]]; then
    echo "ERROR: DRPT_TF32 must be exactly True or False, got $TF32" >&2
    exit 2
fi

args=(
    -c "configs/dolci32k/$setting"
    -m "$method"
    --seed "$SEED"
    --optimizer_type adamw
    --campaign_id "$CAMPAIGN_ID"
    --artifact-build-id "$ARTIFACT_BUILD_ID"
    --model-profile "$MODEL_PROFILE"
    --lr 1e-5
    --report_to wandb
    --wandb_project "$WANDB_PROJECT"
)
if [[ "${DRPT_RETRY_FAILED:-false}" == "true" ]]; then
    args+=(--retry-failed)
fi
if [[ -n "${DRPT_MAX_STEPS:-}" ]]; then
    args+=(--max_steps "$DRPT_MAX_STEPS")
fi
if [[ "${DRPT_DRY_RUN:-false}" == "true" ]]; then
    args+=(--dry-run)
fi

cd "$REPO_ROOT"
echo "[global-axis] campaign=$CAMPAIGN_ID task=$TASK_ID setting=$setting method=$method seed=$SEED TF32=$TF32"
# Selection semantics are fixed by the logical window (N=16, hard mass=8), not
# by the compute chunk; C only trades memory for speed.
echo "[global-axis] dolci32k/$MODEL_PROFILE: logical N=16, target T=2; one global subset of 8 shared by every layer"
exec bash SFT/train/train.sh "${args[@]}"
