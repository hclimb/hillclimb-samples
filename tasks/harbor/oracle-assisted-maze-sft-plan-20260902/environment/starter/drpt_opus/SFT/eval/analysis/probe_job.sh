#!/bin/bash
# Run the layer-alignment probe for every dolci32k setting on one GPU.
#
# No training and no checkpoint loading: the probe sweeps the BASE model once
# per setting, so it can slot into a gap in the queue rather than competing
# with the multi-day training arrays.
#
# Submit with:
#   sbatch --partition=standard --qos=normal --gres=gpu:1 --ntasks=8 --mem=48G \
#          --time=04:00:00 --chdir="$DRPT_REPO_ROOT" \
#          --export=ALL,DRPT_...  SFT/eval/analysis/probe_job.sh

#SBATCH --job-name=layer-align-probe
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/layer_alignment_probe_%j.out

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$REPO_ROOT"

PYTHON="${DRPT_PYTHON:-python}"
CAMPAIGN="${DRPT_CAMPAIGN_ID:-dolci32k-qwen3_1_7b-s42}"
MODEL_PROFILE="${DRPT_MODEL_PROFILE:-qwen3_1_7b}"
BUILD_ID="${DRPT_ARTIFACT_BUILD_ID:?probe requires DRPT_ARTIFACT_BUILD_ID}"
SPLIT="${DRPT_PROBE_SPLIT:-val}"
WINDOWS="${DRPT_PROBE_WINDOWS:-8}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[probe] campaign=$CAMPAIGN model=$MODEL_PROFILE split=$SPLIT windows=$WINDOWS build=$BUILD_ID"
exec "$PYTHON" -m SFT.eval.analysis.layer_alignment_probe \
    --campaign "$CAMPAIGN" \
    --all-settings \
    --model-profile "$MODEL_PROFILE" \
    --data-dir "${DRPT_DATA_DIR:-SFT/data}" \
    --artifact-build-id "$BUILD_ID" \
    --split "$SPLIT" \
    --windows "$WINDOWS"
