#!/bin/bash
# Evaluate one optimizer-scoped campaign run selected by SLURM_ARRAY_TASK_ID.

#SBATCH --job-name=downstream-campaign
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=deadline
#SBATCH --time=02:00:00
#SBATCH --array=0-27%4
#SBATCH --output=logs/downstream_campaign_%A_%a.out

set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_REPO_ROOT}}"
source "$REPO_ROOT/cluster_env.sh" \
    || { echo "ERROR: $REPO_ROOT/cluster_env.sh not found" >&2; exit 2; }
REPO_ROOT="$DRPT_REPO_ROOT"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID is required}"
OPTIMIZER_FAMILY="${DRPT_DOWNSTREAM_FAMILY:?DRPT_DOWNSTREAM_FAMILY is required}"
DOWNSTREAM_PROFILE="${DRPT_DOWNSTREAM_PROFILE:-loss52}"
SEED="${DRPT_SEED:-42}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
N_TEST="${DRPT_N_TEST:-}"
BATCH_SIZE="${DRPT_EVAL_BATCH_SIZE:-}"
MAX_NEW_TOKENS="${DRPT_MAX_NEW_TOKENS:-}"
FORCE_EVAL="${DRPT_FORCE_EVAL:-false}"
ARTIFACT_BUILD_ID="${DRPT_ARTIFACT_BUILD_ID:-}"
MODEL_PROFILE="${DRPT_MODEL_PROFILE:-}"
MODEL_BASENAME="${DRPT_MODEL_BASENAME:-}"
IS_32K_PROFILE=false
[[ "$DOWNSTREAM_PROFILE" == "dolci32k" ]] \
    && IS_32K_PROFILE=true

settings=(alpaca_samsum less_squad less_tydiqa triviaqa_nq)
run_prefixes=(alpaca_samsum less_squad less_tydiqa triviaqa_nq_open)
train_names=(alpaca less less triviaqa)
tasks=(samsum squad tydiqa nq_open)


case "$DOWNSTREAM_PROFILE:$OPTIMIZER_FAMILY" in
    baseline9:adamw)
        methods=(
            FullTraining LayerwiseRaw LayerwiseSoft LayerwiseSoftP LayerwiseOptA
        )
        ;;
    baseline9:muon)
        methods=(
            FullTraining LayerwiseRaw LayerwiseSoft LayerwiseSoftP
            LayerwiseMuonSur LayerwiseMuonPSur
            LayerwiseMuonSatSur LayerwiseMuonSatPSur
        )
        ;;
    dolci32k:adamw|dolci32k:muon)
        methods=()
        ;;
    loss52:adamw)
        methods=(
            FullTraining GlobalRaw LayerwiseRaw GlobalOptA LayerwiseOptA
            GlobalSoft LayerwiseSoft
        )
        ;;
    loss52:muon)
        methods=(
            FullTraining GlobalRaw LayerwiseRaw LayerwiseMuonSur
            GlobalSoft LayerwiseSoft
        )
        ;;
    loss52:hybrid)
        methods=(
            FullTraining GlobalRaw LayerwiseRaw LayerwiseHybridMuonSur
            GlobalSoft LayerwiseSoft
        )
        ;;
    baseline9:hybrid)
        echo "ERROR: baseline9 has no legacy hybrid family" >&2
        exit 2
        ;;
    *)
        echo "ERROR: unsupported profile/family pair: $DOWNSTREAM_PROFILE/$OPTIMIZER_FAMILY" >&2
        exit 2
        ;;
esac

if [[ "$IS_32K_PROFILE" == "true" ]]; then
    registry_python="${DRPT_PYTHON:-python3}"
    registry_path="$REPO_ROOT/SFT/eval/task_registry.py"
    if ! task_count="$("$registry_python" "$registry_path" --profile "$DOWNSTREAM_PROFILE" --family "$OPTIMIZER_FAMILY" --count)"; then
        echo "ERROR: failed to load $DOWNSTREAM_PROFILE cell count from $registry_path" >&2
        exit 2
    fi
    if [[ ! "$task_count" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $DOWNSTREAM_PROFILE registry returned invalid cell count: $task_count" >&2
        exit 2
    fi
    if (( TASK_ID < 0 || TASK_ID >= task_count )); then
        echo "ERROR: array task for $OPTIMIZER_FAMILY must be in [0, $((task_count - 1))], got $TASK_ID" >&2
        exit 2
    fi
    if ! registry_row="$("$registry_python" "$registry_path" --profile "$DOWNSTREAM_PROFILE" --family "$OPTIMIZER_FAMILY" --index "$TASK_ID")"; then
        echo "ERROR: failed to decode $DOWNSTREAM_PROFILE cell $TASK_ID" >&2
        exit 2
    fi
    IFS=$'\t' read -r registry_index setting method task run_prefix train_name registry_max_new_tokens <<< "$registry_row"
    if [[ "$registry_index" != "$TASK_ID" || -z "$setting" || -z "$method" ||
          -z "$task" || -z "$run_prefix" || -z "$train_name" ||
          ! "$registry_max_new_tokens" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: malformed $DOWNSTREAM_PROFILE registry row: $registry_row" >&2
        exit 2
    fi
    if [[ -n "$MAX_NEW_TOKENS" ]]; then
        echo "ERROR: $DOWNSTREAM_PROFILE max_new_tokens is registry-owned; unset DRPT_MAX_NEW_TOKENS" >&2
        exit 2
    fi
    MAX_NEW_TOKENS="$registry_max_new_tokens"
    if [[ -z "$MODEL_PROFILE" ]]; then
        echo "ERROR: $DOWNSTREAM_PROFILE worker requires DRPT_MODEL_PROFILE" >&2
        exit 2
    fi
    if ! model_registry_row="$("$registry_python" "$registry_path" --profile "$DOWNSTREAM_PROFILE" --model-profile-info "$MODEL_PROFILE")"; then
        echo "ERROR: invalid $DOWNSTREAM_PROFILE model profile: $MODEL_PROFILE" >&2
        exit 2
    fi
    IFS=$'\t' read -r registry_model_alias registry_model registry_revision \
        registry_tokenizer registry_tokenizer_revision canonical_model_basename \
        registry_extra <<< "$model_registry_row"
    if [[ "$registry_model_alias" != "$MODEL_PROFILE" ||
          -z "$canonical_model_basename" || -n "${registry_extra:-}" ||
          ! "$canonical_model_basename" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: malformed model registry row: $model_registry_row" >&2
        exit 2
    fi
    if [[ -n "$MODEL_BASENAME" && "$MODEL_BASENAME" != "$canonical_model_basename" ]]; then
        echo "ERROR: DRPT_MODEL_BASENAME does not match the pinned registry" >&2
        exit 2
    fi
    MODEL_BASENAME="$canonical_model_basename"
else
    method_count="${#methods[@]}"
    task_count=$((${#settings[@]} * method_count))
    if (( TASK_ID < 0 || TASK_ID >= task_count )); then
        echo "ERROR: array task for $OPTIMIZER_FAMILY must be in [0, $((task_count - 1))], got $TASK_ID" >&2
        exit 2
    fi
    setting_index=$((TASK_ID / method_count))
    method_index=$((TASK_ID % method_count))
    setting="${settings[$setting_index]}"
    run_prefix="${run_prefixes[$setting_index]}"
    train_name="${train_names[$setting_index]}"
    task="${tasks[$setting_index]}"
    method="${methods[$method_index]}"
fi

if [[ -z "$N_TEST" ]]; then
    [[ "$IS_32K_PROFILE" == "true" ]] && N_TEST=-1 || N_TEST=500
fi
if [[ -z "$BATCH_SIZE" ]]; then
    [[ "$IS_32K_PROFILE" == "true" ]] && BATCH_SIZE=1 || BATCH_SIZE=64
fi
if [[ "$IS_32K_PROFILE" == "true" && "$N_TEST" != "-1" && \
      "${DRPT_ALLOW_LIMITED_EVAL:-false}" != "true" ]]; then
    echo "ERROR: official $DOWNSTREAM_PROFILE reports require DRPT_N_TEST=-1" >&2
    exit 2
fi
if [[ "$IS_32K_PROFILE" != "true" && -z "$MAX_NEW_TOKENS" ]]; then
    case "$task" in
        ifeval|ifbench|mbpp_plus) MAX_NEW_TOKENS=2048 ;;
        math500) MAX_NEW_TOKENS=4096 ;;
        *) MAX_NEW_TOKENS=128 ;;
    esac
fi
if [[ "$IS_32K_PROFILE" == "true" && "${DRPT_JOB_DRY_RUN:-false}" != "true" ]]; then
    if [[ ! "$ARTIFACT_BUILD_ID" =~ ^[0-9a-f]+$ ]]; then
        echo "ERROR: $DOWNSTREAM_PROFILE worker requires a hexadecimal DRPT_ARTIFACT_BUILD_ID" >&2
        exit 2
    fi
fi


runs_root="${DRPT_RUNS_ROOT:-$DRPT_RUNS_DIR}"
campaign_root="$runs_root/campaigns/$CAMPAIGN_ID"
report_root="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/downstream/$OPTIMIZER_FAMILY"
status_dir="$report_root/task_status"
status_file="$status_dir/$(printf '%03d' "$TASK_ID").tsv"
expected_pattern="${run_prefix}-${method}-${OPTIMIZER_FAMILY}-*-s${SEED}-*"
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    expected_pattern="${run_prefix}-${method}-${OPTIMIZER_FAMILY}-*-s${SEED}-${MODEL_BASENAME}"
fi

echo "[downstream] profile=$DOWNSTREAM_PROFILE model_profile=${MODEL_PROFILE:-n/a} campaign=$CAMPAIGN_ID artifact_build_id=${ARTIFACT_BUILD_ID:-n/a} optimizer=$OPTIMIZER_FAMILY task_id=$TASK_ID setting=$setting method=$method task=$task"
echo "[downstream] expected run: $campaign_root/$expected_pattern"

if [[ "${DRPT_JOB_DRY_RUN:-false}" == "true" ]]; then
    printf '[JOB-DRY-RUN] bash SFT/eval/eval.sh --model_path <%s> --task %q --n_test %q --batch_size %q --max_new_tokens %q --seed %q\n' \
        "$expected_pattern" "$task" "$N_TEST" "$BATCH_SIZE" "$MAX_NEW_TOKENS" "$SEED"
    exit 0
fi

activate_env
PYTHON_BIN="${DRPT_EVAL_PYTHON:-$DRPT_PYTHON}"
if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")" \
        || { echo "ERROR: evaluation Python not found: $PYTHON_BIN" >&2; exit 2; }
fi
[[ -x "$PYTHON_BIN" ]] \
    || { echo "ERROR: evaluation Python is not executable: $PYTHON_BIN" >&2; exit 2; }
HELPER="$REPO_ROOT/SFT/eval/campaign_downstream.py"

mkdir -p "$status_dir"

record_status() {
    local status="$1"
    local run_dir="${2:-}"
    local result_file="${3:-}"
    local detail="${4:-}"
    detail="${detail//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    local temporary="${status_file}.tmp.$$"
    {
        printf 'array_task\tcampaign_id\toptimizer\tsetting\ttrain\ttask\tmethod\tseed\tstatus\trun_dir\tresult_file\tdetail\n'
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$TASK_ID" "$CAMPAIGN_ID" "$OPTIMIZER_FAMILY" "$setting" \
            "$train_name" "$task" "$method" "$SEED" "$status" \
            "$run_dir" "$result_file" "$detail"
    } > "$temporary"
    mv "$temporary" "$status_file"
}

if [[ ! -d "$campaign_root" ]]; then
    message="campaign directory is absent after training dependency: $campaign_root"
    echo "SKIP: $message"
    record_status skipped_missing "" "" "$message"
    exit 0
fi

mapfile -d '' -t candidates < <(
    find "$campaign_root" -mindepth 1 -maxdepth 1 -type d \
        -name "$expected_pattern" -print0
)

if (( ${#candidates[@]} == 0 )); then
    message="no matching run directory"
    echo "SKIP: $message"
    record_status skipped_missing "" "" "$message"
    exit 0
fi
if (( ${#candidates[@]} != 1 )); then
    message="ambiguous run directories (${#candidates[@]})"
    echo "SKIP: $message" >&2
    record_status skipped_ambiguous "" "" "$message"
    exit 0
fi

run_dir="${candidates[0]}"
set +e
completion_output="$($PYTHON_BIN "$HELPER" check-run \
    --run-dir "$run_dir" --optimizer "$OPTIMIZER_FAMILY" --profile "$DOWNSTREAM_PROFILE" 2>&1)"
completion_rc=$?
set -e
if (( completion_rc != 0 )); then
    echo "SKIP: incomplete training run: $completion_output"
    record_status skipped_incomplete "$run_dir" "" "$completion_output"
    exit 0
fi

case "$task" in
    samsum) result_file="$run_dir/samsum_results.json" ;;
    squad) result_file="$run_dir/squad_results.json" ;;
    tydiqa) result_file="$run_dir/tydiqa_results.json" ;;
    nq_open) result_file="$run_dir/nq_open_results.json" ;;
    ifeval) result_file="$run_dir/ifeval_results.json" ;;
    ifbench) result_file="$run_dir/ifbench_results.json" ;;
    math500) result_file="$run_dir/math500_results.json" ;;
    mbpp_plus) result_file="$run_dir/mbpp_plus_results.json" ;;
    *) echo "ERROR: unsupported task $task" >&2; exit 2 ;;
esac

if [[ "$FORCE_EVAL" != "true" && -f "$result_file" ]]; then
    if "$PYTHON_BIN" "$HELPER" check-result --run-dir "$run_dir" --task "$task" \
        --profile "$DOWNSTREAM_PROFILE" >/dev/null 2>&1; then
        echo "SKIP: valid downstream result already exists: $result_file"
        record_status already_evaluated "$run_dir" "$result_file" "valid result already present"
        exit 0
    fi
    echo "Existing downstream result is invalid; evaluating again: $result_file"
fi

eval_command=(
    bash "$REPO_ROOT/SFT/eval/eval.sh"
    --model_path "$run_dir"
    --task "$task"
    --n_test "$N_TEST"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
)
set +e
"${eval_command[@]}"
evaluation_rc=$?
set -e
if (( evaluation_rc != 0 )); then
    message="evaluation command exited with status $evaluation_rc"
    record_status failed_evaluation "$run_dir" "$result_file" "$message"
    exit "$evaluation_rc"
fi

set +e
validation_output="$($PYTHON_BIN "$HELPER" check-result \
    --run-dir "$run_dir" --task "$task" --profile "$DOWNSTREAM_PROFILE" 2>&1)"
validation_rc=$?
set -e
if (( validation_rc != 0 )); then
    record_status failed_validation "$run_dir" "$result_file" "$validation_output"
    echo "ERROR: downstream result validation failed: $validation_output" >&2
    exit 1
fi

record_status evaluated "$run_dir" "$result_file" "evaluation completed and validated"
echo "Downstream result: $result_file"
