#!/bin/bash
# Submit optimizer-scoped downstream evaluation arrays for one campaign.
#
# ``all`` means the current default AdamW + Muon comparison.  The legacy
# mixed-score Hybrid family remains available explicitly as ``--family hybrid``.

set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_REPO_ROOT}}"
source "$REPO_ROOT/cluster_env.sh" \
    || { echo "ERROR: $REPO_ROOT/cluster_env.sh not found" >&2; exit 2; }
REPO_ROOT="$DRPT_REPO_ROOT"
CAMPAIGN_ID=""
OPTIMIZER_FAMILY="all"
COMPARISON_PROFILE="baseline9"
MODEL_PROFILE=""
MODEL_BASENAME=""
SEED=42
ARRAY_RANGE=""
MAX_CONCURRENT=4
PARTITION="gpu02"
QOS="deadline"
TIME_LIMIT="02:00:00"
N_TEST=500
BATCH_SIZE=64
MAX_NEW_TOKENS=128
TIME_EXPLICIT=false
N_TEST_EXPLICIT=false
BATCH_SIZE_EXPLICIT=false
MAX_NEW_TOKENS_EXPLICIT=false
RUNS_ROOT="$DRPT_RUNS_DIR"
FORCE_EVAL=false
DRY_RUN=false
SUBMIT_COLLECTOR=true
COMMON_DEPENDENCY=""
ADAMW_DEPENDENCY=""
MUON_DEPENDENCY=""
HYBRID_DEPENDENCY=""

usage() {
    cat <<'EOF'
Submit downstream evaluation for successful runs in one campaign.

Training dependencies use ``afterany``.  Each array task then validates the
finished model and skips a missing/failed training run instead of becoming
DependencyNeverSatisfied.

Options:
  --campaign-id ID          Required campaign namespace under runs/campaigns
  --profile NAME            dolci32k, baseline9 (default), or loss52
  --model-profile NAME      Required for dolci32k; exact pinned model alias
  --family NAME             all, adamw, muon, or hybrid (default: all=adamw+muon)
  --dependency JOBID[:...]  Training job IDs for a single selected family
  --adamw-dependency IDs    AdamW training array job IDs
  --muon-dependency IDs     Muon training array job IDs
  --hybrid-dependency IDs   Legacy Hybrid training array job IDs
  --seed N                  Training seed (default: 42)
  --array RANGE             Override array range for a single family
  --max-concurrent N        Maximum concurrent evaluation tasks (default: 4)
  --partition NAME          Slurm partition (default: gpu02)
  --qos NAME                Slurm QoS (default: deadline)
  --time LIMIT              Per-evaluation time limit (default: 02:00:00)
  --n-test N                Examples/task (immutable 32K profiles require -1)
  --batch-size N            Generation batch size (32K default: 1)
  --max-new-tokens N        Legacy override; 32K uses fixed task policies
  --runs-root DIR           Training runs root (default: DRPT_RUNS_DIR)
  --force                   Re-evaluate valid existing downstream result files
  --no-collector            Do not submit the family-specific CSV/Markdown job
  --dry-run                 Print matrices and exact sbatch commands only

Examples:
  bash SFT/eval/submit_campaign_downstream.sh \
    --campaign-id run42 --family adamw --dependency 12345

  bash SFT/eval/submit_campaign_downstream.sh \
    --campaign-id run42 --family all \
    --adamw-dependency 12345 --muon-dependency 12346
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
        --profile) COMPARISON_PROFILE="$2"; shift 2 ;;
        --model-profile) MODEL_PROFILE="$2"; shift 2 ;;
        --family|--optimizer-family) OPTIMIZER_FAMILY="$2"; shift 2 ;;
        --dependency) COMMON_DEPENDENCY="$2"; shift 2 ;;
        --adamw-dependency|--adamw-job-id) ADAMW_DEPENDENCY="$2"; shift 2 ;;
        --muon-dependency|--muon-job-id) MUON_DEPENDENCY="$2"; shift 2 ;;
        --hybrid-dependency|--hybrid-job-id) HYBRID_DEPENDENCY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --array) ARRAY_RANGE="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; TIME_EXPLICIT=true; shift 2 ;;
        --n-test) N_TEST="$2"; N_TEST_EXPLICIT=true; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; BATCH_SIZE_EXPLICIT=true; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; MAX_NEW_TOKENS_EXPLICIT=true; shift 2 ;;
        --runs-root) RUNS_ROOT="$2"; shift 2 ;;
        --force) FORCE_EVAL=true; shift ;;
        --no-collector) SUBMIT_COLLECTOR=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "ERROR: --campaign-id is required and must be filesystem-safe" >&2
    exit 2
}
[[ "$OPTIMIZER_FAMILY" == "all" || "$OPTIMIZER_FAMILY" == "adamw" || \
   "$OPTIMIZER_FAMILY" == "muon" || "$OPTIMIZER_FAMILY" == "hybrid" ]] || {
    echo "ERROR: family must be all, adamw, muon, or hybrid" >&2
    exit 2
}
[[ "$COMPARISON_PROFILE" == "baseline9" || "$COMPARISON_PROFILE" == "loss52" || \
   "$COMPARISON_PROFILE" == "dolci32k" ]] || {
    echo "ERROR: profile must be dolci32k, baseline9, or loss52" >&2; exit 2;
}
IS_32K_PROFILE=false
[[ "$COMPARISON_PROFILE" == "dolci32k" ]] \
    && IS_32K_PROFILE=true
if [[ "$COMPARISON_PROFILE" != "loss52" && "$OPTIMIZER_FAMILY" == "hybrid" ]]; then
    echo "ERROR: $COMPARISON_PROFILE has adamw and muon families only; use --profile loss52 for legacy hybrid" >&2
    exit 2
fi
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    if [[ "$N_TEST_EXPLICIT" == "true" && "$N_TEST" != "-1" ]]; then
        echo "ERROR: $COMPARISON_PROFILE official reports require --n-test -1" >&2
        exit 2
    fi
    if [[ "$MAX_NEW_TOKENS_EXPLICIT" == "true" ]]; then
        echo "ERROR: $COMPARISON_PROFILE fixes per-task generation limits; omit --max-new-tokens" >&2
        exit 2
    fi
    N_TEST=-1
    MAX_NEW_TOKENS=""
    [[ "$BATCH_SIZE_EXPLICIT" == "true" ]] || BATCH_SIZE=1
    [[ "$TIME_EXPLICIT" == "true" ]] || TIME_LIMIT="12:00:00"
fi
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: seed must be non-negative" >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: max-concurrent must be positive" >&2; exit 2;
}
[[ "$N_TEST" =~ ^-?[0-9]+$ ]] || { echo "ERROR: n-test must be an integer" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: batch-size must be positive" >&2; exit 2; }
[[ -z "$MAX_NEW_TOKENS" || "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: max-new-tokens must be positive" >&2; exit 2;
}
[[ "$RUNS_ROOT" == /* && "$RUNS_ROOT" != *,* ]] || {
    echo "ERROR: runs-root must be an absolute path without commas" >&2; exit 2;
}
PROFILE_REGISTRY="$REPO_ROOT/SFT/eval/task_registry.py"
REGISTRY_PYTHON="${DRPT_PYTHON:-python3}"
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    if [[ -z "$MODEL_PROFILE" ]]; then
        echo "ERROR: --model-profile is required for dolci32k" >&2
        exit 2
    fi
    if ! model_registry_row="$("$REGISTRY_PYTHON" "$PROFILE_REGISTRY" \
        --profile "$COMPARISON_PROFILE" --model-profile-info "$MODEL_PROFILE")"; then
        echo "ERROR: invalid $COMPARISON_PROFILE model profile: $MODEL_PROFILE" >&2
        exit 2
    fi
    IFS=$'\t' read -r registry_alias registry_model registry_revision \
        registry_tokenizer registry_tokenizer_revision MODEL_BASENAME registry_extra \
        <<< "$model_registry_row"
    if [[ "$registry_alias" != "$MODEL_PROFILE" || -z "$registry_model" ||
          -z "$registry_revision" || -z "$registry_tokenizer" ||
          -z "$registry_tokenizer_revision" || -z "$MODEL_BASENAME" ||
          -n "${registry_extra:-}" || ! "$MODEL_BASENAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: malformed model registry row: $model_registry_row" >&2
        exit 2
    fi
elif [[ -n "$MODEL_PROFILE" ]]; then
    echo "ERROR: --model-profile is only valid for immutable 32K profiles" >&2
    exit 2
fi

ARTIFACT_BUILD_ID=""
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    artifact_pin="$RUNS_ROOT/campaigns/$CAMPAIGN_ID/${COMPARISON_PROFILE}_artifact_build_id.txt"
    if [[ -f "$artifact_pin" ]]; then
        IFS= read -r ARTIFACT_BUILD_ID < "$artifact_pin"
        [[ "$ARTIFACT_BUILD_ID" =~ ^[0-9a-f]+$ ]] || {
            echo "ERROR: invalid $COMPARISON_PROFILE artifact pin: $artifact_pin" >&2
            exit 3
        }
    elif [[ "$DRY_RUN" == "true" ]]; then
        ARTIFACT_BUILD_ID="${DRPT_ARTIFACT_BUILD_ID:-DRY_RUN_UNPINNED}"
    else
        echo "ERROR: $COMPARISON_PROFILE campaign artifact pin is missing: $artifact_pin" >&2
        exit 3
    fi
fi
# Comma-separated ranges are accepted so one submission can cover the cells of
# a single benchmark, whose indices are not contiguous across settings (e.g.
# math500 is 10-14 and 30-34). Slurm accepts this form natively.
if [[ -n "$ARRAY_RANGE" && ! "$ARRAY_RANGE" =~ ^([0-9]+|[0-9]+-[0-9]+)(,([0-9]+|[0-9]+-[0-9]+))*(%[1-9][0-9]*)?$ ]]; then
    echo "ERROR: unsupported array range: $ARRAY_RANGE" >&2
    exit 2
fi
if [[ "$OPTIMIZER_FAMILY" == "all" && -n "$ARRAY_RANGE" ]]; then
    echo "ERROR: --array requires a single --family" >&2
    exit 2
fi
if [[ "$OPTIMIZER_FAMILY" == "all" && -n "$COMMON_DEPENDENCY" ]]; then
    echo "ERROR: use --adamw-dependency and --muon-dependency with --family all" >&2
    exit 2
fi

validate_dependency() {
    local value="$1"
    [[ -z "$value" || "$value" =~ ^[0-9]+(:[0-9]+)*$ ]] || {
        echo "ERROR: dependencies must be colon-separated numeric Slurm job IDs: $value" >&2
        exit 2
    }
}
validate_dependency "$COMMON_DEPENDENCY"
validate_dependency "$ADAMW_DEPENDENCY"
validate_dependency "$MUON_DEPENDENCY"
validate_dependency "$HYBRID_DEPENDENCY"

if [[ "$OPTIMIZER_FAMILY" == "all" ]]; then
    families=(adamw muon)
else
    families=("$OPTIMIZER_FAMILY")
fi

settings=(alpaca_samsum less_squad less_tydiqa triviaqa_nq)
if [[ "$COMPARISON_PROFILE" == "baseline9" ]]; then
    adamw_methods=(FullTraining LayerwiseRaw LayerwiseSoft LayerwiseSoftP LayerwiseOptA)
    muon_methods=(
        FullTraining LayerwiseRaw LayerwiseSoft LayerwiseSoftP
        LayerwiseMuonSur LayerwiseMuonPSur
        LayerwiseMuonSatSur LayerwiseMuonSatPSur
    )
elif [[ "$COMPARISON_PROFILE" == "loss52" ]]; then
    adamw_methods=(FullTraining GlobalRaw LayerwiseRaw GlobalOptA LayerwiseOptA GlobalSoft LayerwiseSoft)
    muon_methods=(FullTraining GlobalRaw LayerwiseRaw LayerwiseMuonSur GlobalSoft LayerwiseSoft)
fi
hybrid_methods=(FullTraining GlobalRaw LayerwiseRaw LayerwiseHybridMuonSur GlobalSoft LayerwiseSoft)

dependency_for() {
    local family="$1"
    if [[ -n "$COMMON_DEPENDENCY" ]]; then
        printf '%s' "$COMMON_DEPENDENCY"
    elif [[ "$family" == "adamw" ]]; then
        printf '%s' "$ADAMW_DEPENDENCY"
    elif [[ "$family" == "muon" ]]; then
        printf '%s' "$MUON_DEPENDENCY"
    else
        printf '%s' "$HYBRID_DEPENDENCY"
    fi
}

echo "Campaign: $CAMPAIGN_ID | profile: $COMPARISON_PROFILE | seed: $SEED | family: $OPTIMIZER_FAMILY"
echo "Runs root: $RUNS_ROOT"
echo "Downstream reports: $DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/downstream/<family>/"

for family in "${families[@]}"; do
    registry_cell_count=""
    if [[ "$IS_32K_PROFILE" == "true" ]]; then
        if ! registry_cell_count="$("$REGISTRY_PYTHON" "$PROFILE_REGISTRY" --profile "$COMPARISON_PROFILE" --family "$family" --count)"; then
            echo "ERROR: failed to load $COMPARISON_PROFILE $family cell count" >&2
            exit 2
        fi
        if [[ ! "$registry_cell_count" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: $COMPARISON_PROFILE registry returned invalid $family cell count: $registry_cell_count" >&2
            exit 2
        fi
        default_array="0-$((registry_cell_count - 1))"
        methods=()
    elif [[ "$family" == "adamw" ]]; then
        methods=("${adamw_methods[@]}")
        default_array="0-$((${#settings[@]} * ${#adamw_methods[@]} - 1))"
    elif [[ "$family" == "muon" ]]; then
        methods=("${muon_methods[@]}")
        default_array="0-$((${#settings[@]} * ${#muon_methods[@]} - 1))"
    else
        methods=("${hybrid_methods[@]}")
        default_array="0-23"
    fi
    raw_array="${ARRAY_RANGE:-$default_array}"
    array_spec="$raw_array"
    [[ "$array_spec" == *%* ]] || array_spec="${array_spec}%${MAX_CONCURRENT}"
    training_dependency="$(dependency_for "$family")"

    echo "Family: $family | array: $array_spec | training dependency: ${training_dependency:-none}"
    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ "$IS_32K_PROFILE" == "true" ]]; then
            if ! registry_matrix="$("$REGISTRY_PYTHON" "$PROFILE_REGISTRY" --profile "$COMPARISON_PROFILE" --family "$family")"; then
                echo "ERROR: failed to load $COMPARISON_PROFILE $family cells" >&2
                exit 2
            fi
            registry_rows=0
            while IFS=$'\t' read -r registry_index setting method task run_prefix train_name registry_max_new_tokens; do
                if [[ ! "$registry_index" =~ ^[0-9]+$ || "$registry_index" != "$registry_rows" ||
                      -z "$setting" || -z "$method" || -z "$task" ||
                      -z "$run_prefix" || -z "$train_name" ||
                      ! "$registry_max_new_tokens" =~ ^[1-9][0-9]*$ ]]; then
                    echo "ERROR: malformed $COMPARISON_PROFILE registry row: $registry_index $setting $method $task $run_prefix $train_name" >&2
                    exit 2
                fi
                printf '%s\t%s\t%s\t%s\n' "$setting" "$family" "$method" "$task"
                registry_rows=$((registry_rows + 1))
            done <<< "$registry_matrix"
            if (( registry_rows != registry_cell_count )); then
                echo "ERROR: $COMPARISON_PROFILE registry row count $registry_rows != $registry_cell_count" >&2
                exit 2
            fi
        else
            for setting in "${settings[@]}"; do
                for method in "${methods[@]}"; do
                    printf '%s\t%s\t%s\n' "$setting" "$family" "$method"
                done
            done
        fi
    fi

    export_args="ALL,DRPT_REPO_ROOT=$REPO_ROOT,DRPT_DATA_DIR=$DRPT_DATA_DIR,DRPT_RUNS_DIR=$RUNS_ROOT,DRPT_REPORTS_DIR=$DRPT_REPORTS_DIR,DRPT_LOGS_DIR=$DRPT_LOGS_DIR,DRPT_CAMPAIGN_ID=$CAMPAIGN_ID,DRPT_DOWNSTREAM_PROFILE=$COMPARISON_PROFILE,DRPT_DOWNSTREAM_FAMILY=$family,DRPT_SEED=$SEED,DRPT_N_TEST=$N_TEST,DRPT_EVAL_BATCH_SIZE=$BATCH_SIZE,DRPT_MAX_NEW_TOKENS=$MAX_NEW_TOKENS,DRPT_FORCE_EVAL=$FORCE_EVAL"
    # IFEval/IFBench cells fail closed without the pinned evaluator checkout,
    # so name it explicitly rather than trusting ALL to survive site policy.
    export_args="$export_args,DRPT_IFBENCH_REPO=${DRPT_IFBENCH_REPO:-}"
    if [[ "$IS_32K_PROFILE" == "true" ]]; then
        export_args="$export_args,DRPT_ARTIFACT_BUILD_ID=$ARTIFACT_BUILD_ID,DRPT_MODEL_PROFILE=$MODEL_PROFILE,DRPT_MODEL_BASENAME=$MODEL_BASENAME"
    fi
    eval_command=(
        sbatch --parsable
        --array="$array_spec"
        --partition="$PARTITION"
        --gres=gpu:1
        --ntasks=8
        --qos="$QOS"
        --time="$TIME_LIMIT"
        --job-name="eval-${family}-${CAMPAIGN_ID:0:16}"
        --output="$DRPT_LOGS_DIR/downstream_${family}_%A_%a.out"
        --chdir="$REPO_ROOT"
        --export="$export_args"
    )
    if [[ -n "$training_dependency" ]]; then
        eval_command+=(--dependency="afterany:$training_dependency")
    fi
    eval_command+=(SFT/eval/campaign_downstream_eval_job.sh)

    if [[ "$DRY_RUN" == "true" ]]; then
        printf '[DRY-RUN]'; printf ' %q' "${eval_command[@]}"; echo
        eval_job_id="<${family}-eval-array-job>"
    else
        mkdir -p "$DRPT_LOGS_DIR" \
            "$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/downstream/$family"
        eval_job_id="$("${eval_command[@]}")"
        eval_job_id="${eval_job_id%%;*}"
        echo "Submitted $family downstream array: $eval_job_id"
    fi

    collector_job_id=""
    if [[ "$SUBMIT_COLLECTOR" == "true" ]]; then
        collector_command=(
            sbatch --parsable
            --partition="$PARTITION"
            --ntasks=1
            --cpus-per-task=2
            --qos="$QOS"
            --time=00:20:00
            --dependency="afterany:$eval_job_id"
            --job-name="collect-${family}-${CAMPAIGN_ID:0:12}"
            --output="$DRPT_LOGS_DIR/downstream_${family}_collect_%j.out"
            --chdir="$REPO_ROOT"
            --export="ALL,DRPT_REPO_ROOT=$REPO_ROOT,DRPT_REPORTS_DIR=$DRPT_REPORTS_DIR,DRPT_LOGS_DIR=$DRPT_LOGS_DIR,DRPT_CAMPAIGN_ID=$CAMPAIGN_ID,DRPT_DOWNSTREAM_PROFILE=$COMPARISON_PROFILE,DRPT_DOWNSTREAM_FAMILY=$family,DRPT_MODEL_PROFILE=$MODEL_PROFILE"
            SFT/eval/campaign_downstream_collect_job.sh
        )
        if [[ "$DRY_RUN" == "true" ]]; then
            printf '[DRY-RUN]'; printf ' %q' "${collector_command[@]}"; echo
            collector_job_id="<${family}-collector-job>"
        else
            collector_job_id="$("${collector_command[@]}")"
            collector_job_id="${collector_job_id%%;*}"
            echo "Submitted $family result collector: $collector_job_id"
        fi
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        report_root="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/downstream/$family"
        manifest="$report_root/submission-${eval_job_id}.tsv"
        git_revision="$(git -C "$REPO_ROOT" rev-parse HEAD)"
        {
            echo -e "campaign_id\tprofile\tmodel_profile\toptimizer\tseed\tarray\ttraining_dependency\tdependency_mode\tevaluation_job_id\tcollector_job_id\tn_test\tbatch_size\tmax_new_tokens\tforce\tartifact_build_id\tgit_revision"
            echo -e "$CAMPAIGN_ID\t$COMPARISON_PROFILE\t${MODEL_PROFILE:-n/a}\t$family\t$SEED\t$array_spec\t${training_dependency:-none}\tafterany\t$eval_job_id\t${collector_job_id:-none}\t$N_TEST\t$BATCH_SIZE\t${MAX_NEW_TOKENS:-task-policy}\t$FORCE_EVAL\t${ARTIFACT_BUILD_ID:-n/a}\t$git_revision"
        } > "$manifest"
        echo "Submission manifest: $manifest"
    fi
done
