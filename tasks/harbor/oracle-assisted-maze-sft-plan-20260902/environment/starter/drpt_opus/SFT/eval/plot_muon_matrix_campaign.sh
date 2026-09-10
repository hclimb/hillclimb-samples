#!/bin/bash

set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_REPO_ROOT}}"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID is required}"
SEED="${DRPT_SEED:-42}"

source "$REPO_ROOT/cluster_env.sh"
REPO_ROOT="$DRPT_REPO_ROOT"

activate_env
PLOT_PYTHON="${DRPT_PLOT_PYTHON:-$DRPT_PYTHON}"
if [[ ! -x "$PLOT_PYTHON" ]]; then
    echo "ERROR: plotting Python is not executable: $PLOT_PYTHON" >&2
    exit 2
fi

RUNS_DIR="$DRPT_RUNS_DIR/campaigns/$CAMPAIGN_ID"
OUTPUT_DIR="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID"
MPLCONFIGDIR="${TMPDIR:-/tmp}/drpt-muon-score-source-mpl-${SLURM_JOB_ID:-$$}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR"

cd "$REPO_ROOT"
"$PLOT_PYTHON" SFT/eval/plot_loss_curves.py \
    --profile muon-source \
    --seed "$SEED" \
    --runs-dir "$RUNS_DIR" \
    --output-dir "$OUTPUT_DIR"
