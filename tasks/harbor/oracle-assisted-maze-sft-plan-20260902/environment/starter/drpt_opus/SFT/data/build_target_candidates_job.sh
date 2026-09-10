#!/bin/bash
# Build the offline candidate trajectories the alternate target signals need.
#
# One GPU, one pass over the 64 target rows per target pool. This runs once per
# (build, generator) pair; every later SFT run reads the resulting jsonl. Keep
# it out of the training job so no experiment pays generation cost twice.

#SBATCH --job-name=drpt-target-candidates
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --array=0-2
#SBATCH --output=logs/target_candidates_%A_%a.out

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$REPO_ROOT"
source cluster_env.sh

ARTIFACT_BUILD_ID="${DRPT_ARTIFACT_BUILD_ID:?requires DRPT_ARTIFACT_BUILD_ID}"
GENERATOR_PROFILE="${DRPT_TC_GENERATOR:-qwen3_4b}"
NUM_SAMPLES="${DRPT_TC_NUM_SAMPLES:-6}"
TEMPERATURE="${DRPT_TC_TEMPERATURE:-1.0}"
BATCH_SIZE="${DRPT_TC_BATCH_SIZE:-4}"
SEED="${DRPT_TC_SEED:-42}"

IFS=':' read -r -a targets <<< "${DRPT_TC_TARGETS:-math:mbpp:precise_if}"
target="${targets[${SLURM_ARRAY_TASK_ID:-0}]}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[target-candidates] target=$target generator=$GENERATOR_PROFILE k=$NUM_SAMPLES T=$TEMPERATURE"
exec "$DRPT_PYTHON" -m SFT.data.build_target_candidates \
    --target "$target" \
    --artifact_build_id "$ARTIFACT_BUILD_ID" \
    --generator_profile "$GENERATOR_PROFILE" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --data_dir "$DRPT_DATA_DIR" \
    ${DRPT_TC_OVERWRITE:+--overwrite}
