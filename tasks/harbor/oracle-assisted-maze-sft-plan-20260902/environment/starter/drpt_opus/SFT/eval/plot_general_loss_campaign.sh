#!/bin/bash

set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_REPO_ROOT}}"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID is required}"
SEED="${DRPT_SEED:-42}"
REPORT_FAMILY="${DRPT_REPORT_FAMILY:-legacy-combined}"

case "$REPORT_FAMILY" in
    legacy-combined)
        # Compatibility for already-submitted report jobs that predate the
        # optimizer-scoped dependency split.
        REPORT_PROFILE="loss52-legacy"
        REPORT_SUBDIR="combined"
        ;;
    all)
        REPORT_PROFILE="loss52"
        REPORT_SUBDIR="combined"
        ;;
    adamw|loss52-adamw)
        REPORT_PROFILE="loss52-adamw"
        REPORT_SUBDIR="adamw"
        ;;
    muon|loss52-muon)
        REPORT_PROFILE="loss52-muon"
        REPORT_SUBDIR="muon"
        ;;
    dolci32k)
        REPORT_PROFILE="dolci32k"
        REPORT_SUBDIR="combined"
        ;;
    dolci32k-adamw)
        REPORT_PROFILE="dolci32k-adamw"
        REPORT_SUBDIR="adamw"
        ;;
    dolci32k-muon)
        REPORT_PROFILE="dolci32k-muon"
        REPORT_SUBDIR="muon"
        ;;
    baseline9)
        REPORT_PROFILE="baseline9"
        REPORT_SUBDIR="combined"
        ;;
    baseline9-adamw)
        REPORT_PROFILE="baseline9-adamw"
        REPORT_SUBDIR="adamw"
        ;;
    baseline9-muon)
        REPORT_PROFILE="baseline9-muon"
        REPORT_SUBDIR="muon"
        ;;
    muon-surrogate-variants)
        REPORT_PROFILE="muon-surrogate-variants"
        REPORT_SUBDIR="muon-surrogate-variants"
        ;;
    soft-variants)
        REPORT_PROFILE="soft-variants"
        REPORT_SUBDIR="soft-variants"
        ;;
    hybrid)
        # Backwards-compatible report for the earlier mixed Muon+AdamW scorer.
        REPORT_PROFILE="loss52-hybrid"
        REPORT_SUBDIR="hybrid"
        ;;
    *)
        echo "ERROR: DRPT_REPORT_FAMILY must include dolci32k[-adamw|-muon], baseline9[-adamw|-muon], loss52 family aliases, variants, or hybrid" >&2
        exit 2
        ;;
esac

source "$REPO_ROOT/cluster_env.sh"
REPO_ROOT="$DRPT_REPO_ROOT"

activate_env
PLOT_PYTHON="${DRPT_PLOT_PYTHON:-$DRPT_PYTHON}"
if [[ ! -x "$PLOT_PYTHON" ]]; then
    echo "ERROR: plotting Python is not executable: $PLOT_PYTHON" >&2
    exit 2
fi

RUNS_DIR="$DRPT_RUNS_DIR/campaigns/$CAMPAIGN_ID"
OUTPUT_DIR="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/$REPORT_SUBDIR"
MPLCONFIGDIR="${TMPDIR:-/tmp}/drpt-loss52-${REPORT_FAMILY}-mpl-${SLURM_JOB_ID:-$$}"
export MPLCONFIGDIR
# plot_loss_curves.py imports `SFT.eval.campaign_downstream`, so the repo root
# has to be importable. Running it as a plain script path leaves sys.path[0] at
# SFT/eval/, which fails with ModuleNotFoundError: No module named 'SFT'.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR"

cd "$REPO_ROOT"
"$PLOT_PYTHON" SFT/eval/plot_loss_curves.py \
    --profile "$REPORT_PROFILE" \
    --seed "$SEED" \
    --runs-dir "$RUNS_DIR" \
    --output-dir "$OUTPUT_DIR"
