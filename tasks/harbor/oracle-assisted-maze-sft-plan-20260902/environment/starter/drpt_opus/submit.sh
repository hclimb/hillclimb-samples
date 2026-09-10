#!/bin/bash
#SBATCH -J lesstydi_sft
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=normal
#SBATCH -o logs/%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
source "$_repo_root/cluster_env.sh" || exit 1
REPO_ROOT="$DRPT_REPO_ROOT"
unset _repo_root
PARTITION="${PARTITION:-$DRPT_SLURM_PARTITION}"
QOS="${QOS:-$DRPT_SLURM_QOS}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <script> [script args...]"
    exit 1
  fi

  mkdir -p "$DRPT_LOGS_DIR"

  sbatch_args=()
  [[ -n "${JOB_NAME:-}" ]] && sbatch_args+=(--job-name="$JOB_NAME")
  [[ -n "${PARTITION:-}" ]] && sbatch_args+=(--partition="$PARTITION")
  [[ -n "${GPUS:-}" ]] && sbatch_args+=(--gres="gpu:$GPUS")
  [[ -n "${NTASKS:-}" ]] && sbatch_args+=(--ntasks="$NTASKS")
  [[ -n "${QOS:-}" ]] && sbatch_args+=(--qos="$QOS")
  [[ -n "${TIME:-}" ]] && sbatch_args+=(--time="$TIME")
  [[ -n "${DEPEND:-}" ]] && sbatch_args+=(--dependency="$DEPEND")
  sbatch_args+=(--output="$DRPT_LOGS_DIR/%x_%j.out")
  sbatch_args+=(--export="ALL,DRPT_REPO_ROOT=$REPO_ROOT,DRPT_DATA_DIR=$DRPT_DATA_DIR,DRPT_RUNS_DIR=$DRPT_RUNS_DIR,DRPT_REPORTS_DIR=$DRPT_REPORTS_DIR,DRPT_LOGS_DIR=$DRPT_LOGS_DIR,DRPT_PYTHON=$DRPT_PYTHON")

  cd "$REPO_ROOT"
  exec sbatch "${sbatch_args[@]}" "$0" "$@"
fi

SCRIPT="$1"
shift

cd "$REPO_ROOT"
echo "[INFO] JOB_ID=$SLURM_JOB_ID"
echo "[INFO] RUN=$SCRIPT $*"

bash "$SCRIPT" "$@"
