#!/bin/bash
# Submit the isolated 4-task x 2-scope x 2-scorer-source paired ablation.
# This launcher intentionally does not modify or reuse the active loss52 array.

set -euo pipefail

_repo_root="${DRPT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$_repo_root/cluster_env.sh" \
    || { echo "ERROR: $_repo_root/cluster_env.sh not found."; exit 1; }
unset _repo_root
path_export_args="DRPT_REPO_ROOT=$DRPT_REPO_ROOT,DRPT_DATA_DIR=$DRPT_DATA_DIR,DRPT_RUNS_DIR=$DRPT_RUNS_DIR,DRPT_LOGS_DIR=$DRPT_LOGS_DIR,DRPT_REPORTS_DIR=$DRPT_REPORTS_DIR,DRPT_PYTHON=$DRPT_PYTHON"
CAMPAIGN_ID="muon-scorer-source-s42"
SEED=42
ARRAY_RANGE="0-15"
MAX_CONCURRENT=4
PARTITION="gpu02"
QOS="deadline"
TIME_LIMIT="3-00:00:00"
MAX_STEPS=""
WANDB_PROJECT="drpt_opus"
TF32="True"
DRY_RUN=false
SUBMIT_REPORT=true

usage() {
    cat <<'EOF'
Submit the paired Muon scorer-source ablation (4 tasks x 2 scopes x 2 sources).

Every run prioritizes official torch.optim.Muon for eligible matrices and uses
auxiliary AdamW elsewhere. The paired legacy-hybrid methods differ only in
whether AdamW-managed parameters also contribute selection scores.

Options:
  --campaign-id ID       Isolated runs/campaigns/ID namespace
  --seed N               Training seed (default: 42)
  --array RANGE          Array IDs, e.g. 0-15 or 3 for a smoke run
  --max-concurrent N     Maximum simultaneously running array tasks
  --max-steps N          Optional short-run override for smoke testing
  --partition NAME       Slurm partition (default: gpu02)
  --qos NAME             Slurm QoS (default: deadline)
  --time LIMIT           Slurm time limit (default: 3-00:00:00)
  --wandb-project NAME   W&B project (default: drpt_opus)
  --tf32 BOOL            Use TF32 uniformly (default: True)
  --no-report            Do not submit the dependent plotting job
  --dry-run              Print the exact matrix and sbatch command
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --array) ARRAY_RANGE="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --tf32) TF32="$2"; shift 2 ;;
        --no-report) SUBMIT_REPORT=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "ERROR: invalid campaign id: $CAMPAIGN_ID" >&2; exit 2;
}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: seed must be non-negative" >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: max-concurrent must be positive" >&2; exit 2;
}
if [[ -n "$MAX_STEPS" && ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: max-steps must be positive" >&2
    exit 2
fi
[[ "$TF32" == "True" || "$TF32" == "False" ]] || {
    echo "ERROR: tf32 must be exactly True or False" >&2; exit 2;
}
[[ "$ARRAY_RANGE" =~ ^([0-9]+|[0-9]+-[0-9]+)(%[1-9][0-9]*)?$ ]] || {
    echo "ERROR: unsupported array range: $ARRAY_RANGE" >&2; exit 2;
}

array_spec="$ARRAY_RANGE"
[[ "$array_spec" == *%* ]] || array_spec="${array_spec}%${MAX_CONCURRENT}"

settings=(alpaca_samsum less_squad less_tydiqa triviaqa_nq)
methods=(
    GlobalHybridMuonSur
    LayerwiseHybridMuonSur
    GlobalHybridMuonMatrixSur
    LayerwiseHybridMuonMatrixSur
)

echo "Campaign: $CAMPAIGN_ID"
echo "Seed: $SEED | partition: $PARTITION | array: $array_spec | TF32: $TF32"
echo "Matrix: 4 settings x 2 scopes x 2 scorer sources = 16 hybrid-optimizer runs"
if [[ "$DRY_RUN" == "true" ]]; then
    for setting in "${settings[@]}"; do
        for method in "${methods[@]}"; do
            echo -e "$setting\thybrid\t$method"
        done
    done
fi

campaign_root="$DRPT_RUNS_DIR/campaigns/$CAMPAIGN_ID"
if [[ -e "$campaign_root" ]]; then
    echo "ERROR: campaign path already exists; refusing to overwrite: $campaign_root" >&2
    exit 3
fi

export_args="ALL,$path_export_args,DRPT_CAMPAIGN_ID=$CAMPAIGN_ID,DRPT_SEED=$SEED,DRPT_WANDB_PROJECT=$WANDB_PROJECT,DRPT_TF32=$TF32,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[[ -n "$MAX_STEPS" ]] && export_args="$export_args,DRPT_MAX_STEPS=$MAX_STEPS"

train_cmd=(
    sbatch --parsable
    --array="$array_spec"
    --partition="$PARTITION"
    --gres=gpu:1
    --ntasks=8
    --qos="$QOS"
    --time="$TIME_LIMIT"
    --job-name="muonsrc-${CAMPAIGN_ID:0:18}"
    --output="$DRPT_LOGS_DIR/muon_score_source_%A_%a.out"
    --chdir="$DRPT_REPO_ROOT"
    --export="$export_args"
    SFT/train/muon_matrix_surrogate_job.sh
)

if [[ "$DRY_RUN" == "true" ]]; then
    printf '[DRY-RUN]'; printf ' %q' "${train_cmd[@]}"; echo
    exit 0
fi

mkdir -p "$DRPT_LOGS_DIR" "$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID"
train_job_id="$("${train_cmd[@]}")"
train_job_id="${train_job_id%%;*}"
echo "Submitted training array: $train_job_id"

report_job_id=""
if [[ "$SUBMIT_REPORT" == "true" && "$ARRAY_RANGE" == "0-15" && -z "$MAX_STEPS" ]]; then
    report_job_id="$(sbatch --parsable \
        --partition="$PARTITION" \
        --ntasks=1 \
        --cpus-per-task=2 \
        --qos="$QOS" \
        --time=01:00:00 \
        --dependency="afterany:$train_job_id" \
        --job-name="plot-${CAMPAIGN_ID:0:18}" \
        --output="$DRPT_LOGS_DIR/muon_score_source_plot_%j.out" \
        --chdir="$DRPT_REPO_ROOT" \
        --export="ALL,$path_export_args,DRPT_CAMPAIGN_ID=$CAMPAIGN_ID,DRPT_SEED=$SEED" \
        SFT/eval/plot_muon_matrix_campaign.sh)"
    report_job_id="${report_job_id%%;*}"
    echo "Submitted afterany report job: $report_job_id"
fi

manifest="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/submission.tsv"
git_revision="$(git -C "$DRPT_REPO_ROOT" rev-parse HEAD)"
{
    echo -e "campaign_id\tseed\tarray\ttraining_job_id\treport_job_id\tcomparison\toptimizer\ttf32\tpartition\ttime_limit\tmax_steps\tstatus\tgit_revision"
    echo -e "$CAMPAIGN_ID\t$SEED\t$array_spec\t$train_job_id\t$report_job_id\tmuon_plus_adamw_vs_muon_matrix_only_scores\thybrid\t$TF32\t$PARTITION\t$TIME_LIMIT\t${MAX_STEPS:-full}\tsubmitted\t$git_revision"
} > "$manifest"
echo "Submission manifest: $manifest"
