#!/bin/bash
#SBATCH --job-name=muon-score-src
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=deadline
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15%4
#SBATCH --output=logs/muon_score_source_%A_%a.out

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID must be exported by the launcher}"
SEED="${DRPT_SEED:-42}"
WANDB_PROJECT="${DRPT_WANDB_PROJECT:-drpt_opus}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"

settings=(alpaca_samsum less_squad less_tydiqa triviaqa_nq)
methods=(
    GlobalHybridMuonSur
    LayerwiseHybridMuonSur
    GlobalHybridMuonMatrixSur
    LayerwiseHybridMuonMatrixSur
)

if (( TASK_ID < 0 || TASK_ID >= 16 )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be in [0, 15], got $TASK_ID" >&2
    exit 2
fi

setting="${settings[$((TASK_ID / 4))]}"
method="${methods[$((TASK_ID % 4))]}"
case "$method" in
    GlobalHybridMuonMatrixSur|LayerwiseHybridMuonMatrixSur)
        scorer_source="muon_matrices_only" ;;
    *)
        scorer_source="muon_matrices_plus_adamw" ;;
esac

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
    --optimizer_type hybrid
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
echo "[muon-score-source] campaign=$CAMPAIGN_ID array_task=$TASK_ID setting=$setting method=$method scorer_source=$scorer_source optimizer=hybrid seed=$SEED TF32=$TF32"
echo "[muon-score-source] paired legacy-hybrid ablation: all runs use official Muon plus auxiliary AdamW; only the parameters contributing selection scores differ"
exec bash SFT/train/train.sh "${args[@]}"
