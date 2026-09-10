#!/bin/bash
# Run one optimizer/method pair from the focused soft-constraint comparison.

#SBATCH --job-name=soft-variants
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=deadline
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15%4
#SBATCH --output=logs/soft_variants_%A_%a.out

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID must be exported by the launcher}"
SEED="${DRPT_SEED:-42}"
WANDB_PROJECT="${DRPT_WANDB_PROJECT:-drpt_opus}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"

settings=(alpaca_samsum less_squad less_tydiqa triviaqa_nq)
optimizers=(adamw muon)
methods=(LayerwiseSoft LayerwiseSoftP)

if (( TASK_ID < 0 || TASK_ID >= 16 )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be in [0, 15], got $TASK_ID" >&2
    exit 2
fi

setting_index=$((TASK_ID / 4))
pair_index=$((TASK_ID % 4))
setting="${settings[$setting_index]}"
optimizer_type="${optimizers[$((pair_index / 2))]}"
method="${methods[$((pair_index % 2))]}"

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
    --optimizer_type "$optimizer_type"
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
echo "[soft-variants] campaign=$CAMPAIGN_ID task=$TASK_ID setting=$setting optimizer=$optimizer_type method=$method seed=$SEED TF32=$TF32"
exec bash SFT/train/train.sh "${args[@]}"
