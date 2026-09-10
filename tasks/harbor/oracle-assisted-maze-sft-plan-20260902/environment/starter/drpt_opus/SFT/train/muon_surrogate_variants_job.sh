#!/bin/bash
# Run one layerwise Muon-matrix-only surrogate from the focused comparison.

#SBATCH --job-name=muon-sur-variants
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=deadline
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15%4
#SBATCH --output=logs/muon_surrogate_variants_%A_%a.out

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID must be exported by the launcher}"
SEED="${DRPT_SEED:-42}"
WANDB_PROJECT="${DRPT_WANDB_PROJECT:-drpt_opus}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"

settings=(alpaca_samsum less_squad less_tydiqa triviaqa_nq)
methods=(
    LayerwiseMuonSur
    LayerwiseMuonPSur
    LayerwiseMuonSatSur
    LayerwiseMuonSatPSur
)

if (( TASK_ID < 0 || TASK_ID >= 16 )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be in [0, 15], got $TASK_ID" >&2
    exit 2
fi

setting="${settings[$((TASK_ID / 4))]}"
method="${methods[$((TASK_ID % 4))]}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TF32="${DRPT_TF32:-True}"
if [[ "$TF32" != "True" && "$TF32" != "False" ]]; then
    echo "ERROR: DRPT_TF32 must be exactly True or False, got $TF32" >&2
    exit 2
fi

args=(
    -c "configs/$setting"
    -m "$method"
    --seed "$SEED"
    --optimizer_type muon
    --campaign_id "$CAMPAIGN_ID"
    --report_to wandb
    --wandb_project "$WANDB_PROJECT"
)
if [[ -n "${DRPT_MAX_STEPS:-}" ]]; then
    args+=(--max_steps "$DRPT_MAX_STEPS")
fi
if [[ "${DRPT_DRY_RUN:-false}" == "true" ]]; then
    args+=(--dry-run)
fi

cd "$REPO_ROOT"
echo "[muon-surrogate-variants] campaign=$CAMPAIGN_ID task=$TASK_ID setting=$setting method=$method optimizer=muon seed=$SEED TF32=$TF32"
exec bash SFT/train/train.sh "${args[@]}"
