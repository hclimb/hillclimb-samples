#!/bin/bash
#SBATCH --job-name=dolci-cprobe
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dolci_candidate_probe_%j.out
#SBATCH --error=logs/dolci_candidate_probe_%j.err

# One-GPU diagnostic probe for the Dolci32k logical candidate window.
#
# Each method first attempts the non-windowed C=16 path for two optimizer steps.
# The second step is essential: AdamW state is allocated only after step one, so
# a one-step run can miss the steady-state activation+optimizer memory peak. A
# non-zero exit (including CUDA OOM) automatically falls back to windowed C=8.
# Methods run sequentially, so this job never consumes more than one GPU.

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$REPO_ROOT/cluster_env.sh" \
    || { echo "ERROR: $REPO_ROOT/cluster_env.sh not found." >&2; exit 1; }
activate_env

cd "$REPO_ROOT"

ARTIFACT_BUILD_ID="68215f282c0954521df63ca2bbe2fbf626817e14e4041c773c205d87bab5f9ae"
SETTING="reason_math"
MODEL_PROFILE="qwen3_1_7b"
SEED="${DRPT_PROBE_SEED:-42}"
MAX_STEPS="${DRPT_PROBE_MAX_STEPS:-2}"
DRY_RUN="${DRPT_DRY_RUN:-false}"
EXERCISE_FALLBACK="${DRPT_PROBE_EXERCISE_FALLBACK:-false}"
CAMPAIGN_PREFIX="${DRPT_PROBE_CAMPAIGN_PREFIX:-dolci32k-candidate-probe}"
methods=(LayerwiseRaw LayerwiseOptA LayerwiseSoft)

[[ "$SEED" =~ ^[0-9]+$ ]] || {
    echo "ERROR: DRPT_PROBE_SEED must be a non-negative integer, got $SEED" >&2
    exit 2
}
[[ "$MAX_STEPS" =~ ^[1-3]$ ]] || {
    echo "ERROR: DRPT_PROBE_MAX_STEPS must be 1, 2, or 3, got $MAX_STEPS" >&2
    exit 2
}
[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || {
    echo "ERROR: DRPT_DRY_RUN must be true or false, got $DRY_RUN" >&2
    exit 2
}
[[ "$EXERCISE_FALLBACK" == "true" || "$EXERCISE_FALLBACK" == "false" ]] || {
    echo "ERROR: DRPT_PROBE_EXERCISE_FALLBACK must be true or false, got $EXERCISE_FALLBACK" >&2
    exit 2
}
if [[ "$EXERCISE_FALLBACK" == "true" && "$DRY_RUN" != "true" ]]; then
    echo "ERROR: DRPT_PROBE_EXERCISE_FALLBACK is permitted only with DRPT_DRY_RUN=true" >&2
    exit 2
fi
[[ "$CAMPAIGN_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "ERROR: invalid DRPT_PROBE_CAMPAIGN_PREFIX: $CAMPAIGN_PREFIX" >&2
    exit 2
}

export TF32=True
export DRPT_DOLCI_CANDIDATE_PROBE_SKIP_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

job_token="${SLURM_JOB_ID:-local-$$}-r${SLURM_RESTART_COUNT:-0}"
probe_id="${CAMPAIGN_PREFIX}-${job_token}"
result_dir="${DRPT_PROBE_RESULT_DIR:-$DRPT_LOGS_DIR/dolci32k_candidate_microbatch_probe/$probe_id}"
mkdir -p "$result_dir"
results_tsv="$result_dir/results.tsv"
printf 'started_utc\tmethod\tcandidate_microbatch\tcampaign\tresult\tselected\texit_code\telapsed_seconds\tcampaign_root\tlog\n' > "$results_tsv"

LAST_STATUS=0
LAST_RESULT=""
LAST_STARTED_UTC=""
LAST_ELAPSED=0
LAST_LOG=""

method_slug() {
    case "$1" in
        LayerwiseRaw) echo "raw" ;;
        LayerwiseOptA) echo "opta" ;;
        LayerwiseSoft) echo "soft" ;;
        *) echo "ERROR: unsupported probe method: $1" >&2; return 2 ;;
    esac
}

run_attempt() {
    local method="$1"
    local candidate_microbatch="$2"
    local campaign="$3"
    local attempt_log="$result_dir/${method}_c${candidate_microbatch}.log"
    local started_epoch ended_epoch train_status tee_status
    local -a pipeline_status
    local -a args=(
        -c "configs/dolci32k/$SETTING"
        -m "$method"
        --seed "$SEED"
        --optimizer_type adamw
        --campaign-id "$campaign"
        --artifact-build-id "$ARTIFACT_BUILD_ID"
        --model-profile "$MODEL_PROFILE"
        --lr 1e-5
        --report_to none
        --max-steps "$MAX_STEPS"
        --candidate-microbatch-size "$candidate_microbatch"
    )

    if [[ "$candidate_microbatch" == "16" ]]; then
        args+=(--allow-dolci-c16-probe)
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        args+=(--dry-run)
    fi

    LAST_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    started_epoch="$(date +%s)"
    echo "[probe] start=$LAST_STARTED_UTC method=$method C=$candidate_microbatch campaign=$campaign"
    printf '[probe] command:'
    printf ' %q' bash SFT/train/train.sh "${args[@]}"
    printf '\n'

    # A failed C=16 attempt is expected on memory-constrained GPUs.  Disable
    # errexit only around the pipeline so the shell can record it and continue.
    set +e
    bash SFT/train/train.sh "${args[@]}" 2>&1 | tee "$attempt_log"
    pipeline_status=("${PIPESTATUS[@]}")
    train_status="${pipeline_status[0]}"
    tee_status="${pipeline_status[1]}"
    set -e

    if (( train_status == 0 && tee_status != 0 )); then
        train_status="$tee_status"
    fi
    if [[ "$candidate_microbatch" == "16" && "$EXERCISE_FALLBACK" == "true" && "$train_status" == "0" ]]; then
        echo "[probe] dry-run validation: simulating C=16 failure to exercise C=8 fallback"
        train_status=90
    fi

    ended_epoch="$(date +%s)"
    LAST_ELAPSED=$((ended_epoch - started_epoch))
    LAST_STATUS="$train_status"
    LAST_LOG="$attempt_log"
    if (( train_status == 0 )); then
        LAST_RESULT="success"
    elif rg -q 'torch\.OutOfMemoryError|CUDA out of memory|CUDA error: out of memory' "$attempt_log"; then
        LAST_RESULT="oom"
    elif (( train_status == 90 )) && [[ "$EXERCISE_FALLBACK" == "true" ]]; then
        LAST_RESULT="simulated_failure"
    else
        LAST_RESULT="failed"
    fi
    echo "[probe] finish method=$method C=$candidate_microbatch result=$LAST_RESULT exit=$LAST_STATUS elapsed=${LAST_ELAPSED}s log=$LAST_LOG"
}

record_attempt() {
    local method="$1"
    local candidate_microbatch="$2"
    local campaign="$3"
    local selected="$4"
    local campaign_root="$DRPT_RUNS_DIR/campaigns/$campaign"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$LAST_STARTED_UTC" "$method" "$candidate_microbatch" "$campaign" \
        "$LAST_RESULT" "$selected" "$LAST_STATUS" "$LAST_ELAPSED" \
        "$campaign_root" "$LAST_LOG" >> "$results_tsv"
}

overall_status=0
for method in "${methods[@]}"; do
    slug="$(method_slug "$method")"
    campaign_c16="${probe_id}-${slug}-c16"
    run_attempt "$method" 16 "$campaign_c16"
    if (( LAST_STATUS == 0 )); then
        record_attempt "$method" 16 "$campaign_c16" true
        echo "[probe] selected C=16 for $method"
        continue
    fi

    record_attempt "$method" 16 "$campaign_c16" false
    echo "[probe] C=16 did not complete for $method; falling back to C=8"
    campaign_c8="${probe_id}-${slug}-c8"
    run_attempt "$method" 8 "$campaign_c8"
    if (( LAST_STATUS == 0 )); then
        record_attempt "$method" 8 "$campaign_c8" true
        echo "[probe] selected fallback C=8 for $method"
    else
        record_attempt "$method" 8 "$campaign_c8" false
        overall_status=1
        echo "[probe] ERROR: both C=16 and C=8 failed for $method" >&2
    fi
done

echo "[probe] results=$results_tsv"
if (( overall_status != 0 )); then
    echo "[probe] one or more methods failed at both candidate micro-batch sizes" >&2
fi
exit "$overall_status"
