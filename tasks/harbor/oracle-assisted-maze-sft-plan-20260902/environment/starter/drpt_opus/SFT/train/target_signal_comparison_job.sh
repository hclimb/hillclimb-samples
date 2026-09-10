#!/bin/bash
# Run one cell of the target-gradient signal comparison.
#
# The dolci32k curation methods lower target validation loss on the math and
# code settings while losing on MATH500 / MBPP+. Target-only SFT (see
# SFT/train/train_target_only.py) showed the same split, which points at D* --
# and at the objective whose gradient D* is summarized by -- rather than at the
# selection machinery. This array varies exactly that objective and nothing
# else: same build, same candidate pool, same traversal, same optimizer.
#
#   nll                       historical token-mean cross entropy (control)
#   answer_only_ce            same loss, final-answer tokens only
#   correct_incorrect_margin  reference above a verified-wrong trajectory
#   reward_weighted_sft       trajectories weighted by verified correctness
#
# The last two read SFT/data/dolci32k_artifacts/target_signals/<build>/, which
# SFT/data/build_target_candidates.py writes once, offline. Nothing here adds a
# rollout, a reward model, or an online RL step to the training loop.
#
# Array index -> cell: setting-major, signal-minor.

#SBATCH --job-name=drpt-target-signal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --array=0-11%4
#SBATCH --output=logs/target_signal_%A_%a.out

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$REPO_ROOT"
source cluster_env.sh

CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID must be exported by the launcher}"
ARTIFACT_BUILD_ID="${DRPT_ARTIFACT_BUILD_ID:?target-signal worker requires DRPT_ARTIFACT_BUILD_ID}"
MODEL_PROFILE="${DRPT_MODEL_PROFILE:-qwen3_1_7b}"
# A method name train.sh resolves: a config stem (LayerWiseSubset-Full) or the
# label it carries (LayerwiseRaw). Run `bash SFT/train/train.sh --list` to see them.
METHOD="${DRPT_TS_METHOD:-LayerWiseSubset-Full}"
SEED="${DRPT_SEED:-42}"
WANDB_PROJECT="${DRPT_WANDB_PROJECT:-drpt_opus}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"

IFS=':' read -r -a settings <<< "${DRPT_TS_SETTINGS:-reason_math:reason_code:inst_if}"
IFS=':' read -r -a signals <<< "${DRPT_TS_SIGNALS:-nll:answer_only_ce:correct_incorrect_margin:reward_weighted_sft}"

task_count=$(( ${#settings[@]} * ${#signals[@]} ))
if (( TASK_ID < 0 || TASK_ID >= task_count )); then
    echo "ERROR: array task must be in [0, $((task_count - 1))], got $TASK_ID" >&2
    exit 2
fi
setting="${settings[$((TASK_ID / ${#signals[@]}))]}"
signal="${signals[$((TASK_ID % ${#signals[@]}))]}"

# Fail before the model is allocated if this cell's offline artifact is absent.
if [[ "$signal" == "correct_incorrect_margin" || "$signal" == "reward_weighted_sft" ]]; then
    target="$("${DRPT_PYTHON:-python}" -c "
from SFT.data.dolci32k.profile import SETTINGS
print(SETTINGS['$setting']['target'])
")"
    candidates="SFT/data/dolci32k_artifacts/target_signals/$ARTIFACT_BUILD_ID/$target/candidates.jsonl"
    if [[ ! -f "$candidates" ]]; then
        echo "ERROR: $signal needs $candidates" >&2
        echo "Build it once with:" >&2
        echo "  $DRPT_PYTHON -m SFT.data.build_target_candidates --target $target \\" >&2
        echo "      --artifact_build_id $ARTIFACT_BUILD_ID --generator_profile qwen3_4b" >&2
        exit 2
    fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

args=(
    -c "configs/dolci32k/$setting"
    -m "$METHOD"
    --seed "$SEED"
    --optimizer_type adamw
    --campaign_id "$CAMPAIGN_ID"
    --artifact-build-id "$ARTIFACT_BUILD_ID"
    --model-profile "$MODEL_PROFILE"
    --target-signal "$signal"
    --lr 1e-5
    --report_to wandb
    --wandb_project "$WANDB_PROJECT"
)
[[ -n "${DRPT_TS_BETA:-}" ]] && args+=(--target-signal-beta "$DRPT_TS_BETA")
[[ -n "${DRPT_TS_MARGIN:-}" ]] && args+=(--target-signal-margin "$DRPT_TS_MARGIN")
[[ -n "${DRPT_TS_INCORRECT_REWARD:-}" ]] && \
    args+=(--target-signal-incorrect-reward "$DRPT_TS_INCORRECT_REWARD")
# Applied to every arm, including nll, so the target prompt set is identical
# across the comparison. Set it when margin-pair coverage is below 64/64.
[[ "${DRPT_TS_ALIGN_PROMPTS:-false}" == "true" ]] && args+=(--target-signal-align-prompts)
[[ -n "${DRPT_MAX_STEPS:-}" ]] && args+=(--max_steps "$DRPT_MAX_STEPS")
[[ "${DRPT_RETRY_FAILED:-false}" == "true" ]] && args+=(--retry-failed)
[[ "${DRPT_DRY_RUN:-false}" == "true" ]] && args+=(--dry-run)

echo "[target-signal] campaign=$CAMPAIGN_ID task=$TASK_ID setting=$setting signal=$signal method=$METHOD seed=$SEED"
exec bash SFT/train/train.sh "${args[@]}"
