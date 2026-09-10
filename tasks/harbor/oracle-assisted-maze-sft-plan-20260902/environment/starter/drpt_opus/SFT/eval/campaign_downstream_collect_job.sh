#!/bin/bash
# Merge family-specific downstream evaluation status and result files.

#SBATCH --job-name=collect-downstream
#SBATCH --partition=gpu02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --qos=deadline
#SBATCH --time=00:20:00
#SBATCH --output=logs/downstream_collect_%j.out

set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_REPO_ROOT}}"
source "$REPO_ROOT/cluster_env.sh" \
    || { echo "ERROR: $REPO_ROOT/cluster_env.sh not found" >&2; exit 2; }
REPO_ROOT="$DRPT_REPO_ROOT"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID is required}"
OPTIMIZER_FAMILY="${DRPT_DOWNSTREAM_FAMILY:?DRPT_DOWNSTREAM_FAMILY is required}"
DOWNSTREAM_PROFILE="${DRPT_DOWNSTREAM_PROFILE:-baseline9}"
report_root="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/downstream/$OPTIMIZER_FAMILY"

activate_env
PYTHON_BIN="${DRPT_EVAL_PYTHON:-$DRPT_PYTHON}"
if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")" \
        || { echo "ERROR: collector Python not found: $PYTHON_BIN" >&2; exit 2; }
fi
[[ -x "$PYTHON_BIN" ]] \
    || { echo "ERROR: collector Python is not executable: $PYTHON_BIN" >&2; exit 2; }

cd "$REPO_ROOT"
"$PYTHON_BIN" SFT/eval/campaign_downstream.py collect \
    --campaign-id "$CAMPAIGN_ID" \
    --optimizer "$OPTIMIZER_FAMILY" \
    --profile "$DOWNSTREAM_PROFILE" \
    --status-dir "$report_root/task_status" \
    --output-dir "$report_root"
