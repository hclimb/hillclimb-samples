#!/bin/bash
# Submit the dolci32k global-vs-layerwise architecture-axis array.
#
# 5 settings x {GlobalRaw, GlobalOptA} = 10 runs into an EXISTING dolci32k
# campaign root, reusing that campaign's pinned artifact build so the candidate
# traversal, tokenization, and target splits are bit-identical to the 25 runs
# already there. A 3-step smoke array gates the formal array.
#
# These runs need the windowed global selection path
# (drpt/selection/backward.py::finalize_windowed_global_selection), which
# postdates the campaign's frozen `source/` snapshot. They therefore run from
# their own immutable snapshot in `source_global_axis/`; the original
# `source/` tree is never touched, so the main campaign's snapshot check and
# the already-queued Muon arrays are unaffected.

set -euo pipefail

_repo_root="${DRPT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$_repo_root/cluster_env.sh" \
    || { echo "ERROR: $_repo_root/cluster_env.sh not found."; exit 1; }
unset _repo_root

CAMPAIGN_ID="dolci32k-qwen3_1_7b-s42"
MODEL_PROFILE="qwen3_1_7b"
SEED=42
ARRAY_RANGE="0-9"
MAX_CONCURRENT=4
PARTITION="$DRPT_SLURM_PARTITION"
QOS="$DRPT_SLURM_QOS"
TIME_LIMIT="7-00:00:00"
MEMORY="48G"
WANDB_PROJECT="drpt_opus"
TF32="True"
DRY_RUN=false
RUN_SMOKE=true
RETRY_FAILED=false
REFRESH_SNAPSHOT=false
HOLD_JOBS=""

usage() {
    cat <<'EOF'
Submit the dolci32k GlobalRaw/GlobalOptA architecture-axis array.

Options:
  --campaign-id ID       Existing dolci32k campaign to extend
                         (default: dolci32k-qwen3_1_7b-s42)
  --model-profile NAME   olmo3_7b, qwen3_1_7b, qwen3_4b, or qwen3_8b
  --seed N               Training seed (default: 42)
  --array RANGE          Array IDs, e.g. 0-9 or 0-1 for a partial resubmit
  --max-concurrent N     Maximum simultaneously running tasks (default: 4)
  --partition NAME       Slurm partition
  --qos NAME             Slurm QoS
  --time LIMIT           Slurm time limit (default: 7-00:00:00)
  --memory SIZE          Slurm memory request (default: 48G)
  --wandb-project NAME   W&B project (default: drpt_opus)
  --hold-after JOBIDS    Comma-separated pending job ids to push behind this
                         array, e.g. the queued Muon smoke job. Each is edited
                         with `scontrol update dependency=afterany:<main>`.
  --no-smoke             Skip the 3-step smoke gate (development only)
  --retry-failed         Permit workers to retry failed (never completed) runs
  --refresh-snapshot     Rebuild the immutable source_global_axis/ snapshot from
                         the live repo. Required after fixing code between a
                         failed smoke gate and a resubmit; otherwise workers
                         silently keep running the previous snapshot. Refused
                         once any cell has completed.
  --dry-run              Print the resolved matrix and sbatch commands

Array index -> cell: setting-major, method-minor over
  settings = SFT/data/dolci32k/profile.py::SETTING_ORDER
  methods  = (GlobalRaw, GlobalOptA)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
        --model-profile) MODEL_PROFILE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --array) ARRAY_RANGE="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --hold-after) HOLD_JOBS="$2"; shift 2 ;;
        --no-smoke) RUN_SMOKE=false; shift ;;
        --retry-failed) RETRY_FAILED=true; shift ;;
        --refresh-snapshot) REFRESH_SNAPSHOT=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "ERROR: invalid campaign id: $CAMPAIGN_ID" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: seed must be non-negative" >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-4]$ ]] || {
    echo "ERROR: dolci32k permits at most four concurrent single-GPU jobs" >&2; exit 2; }
[[ "$TF32" == "True" ]] || { echo "ERROR: dolci32k fixes TF32=True" >&2; exit 2; }
[[ "$ARRAY_RANGE" =~ ^([0-9]+|[0-9]+-[0-9]+)(%[1-9][0-9]*)?$ ]] || {
    echo "ERROR: unsupported array range: $ARRAY_RANGE" >&2; exit 2; }
case "$MODEL_PROFILE" in olmo3_7b|qwen3_1_7b|qwen3_4b|qwen3_8b) ;; *)
    echo "ERROR: invalid dolci32k model profile: $MODEL_PROFILE" >&2; exit 2 ;;
esac
if [[ -n "$HOLD_JOBS" && ! "$HOLD_JOBS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: --hold-after takes comma-separated numeric job ids" >&2
    exit 2
fi

array_spec="$ARRAY_RANGE"
[[ "$array_spec" == *%* ]] || array_spec="${array_spec}%${MAX_CONCURRENT}"

campaign_root="$DRPT_RUNS_DIR/campaigns/$CAMPAIGN_ID"
artifact_pin="$campaign_root/dolci32k_artifact_build_id.txt"
[[ -d "$campaign_root" ]] || {
    echo "ERROR: this array extends an existing campaign; not found: $campaign_root" >&2
    exit 3; }
[[ -f "$artifact_pin" ]] || {
    echo "ERROR: campaign lacks its immutable artifact pin: $artifact_pin" >&2
    exit 3; }
ARTIFACT_BUILD_ID="$(<"$artifact_pin")"
[[ "$ARTIFACT_BUILD_ID" =~ ^[0-9a-f]+$ ]] || {
    echo "ERROR: invalid artifact pin in $artifact_pin" >&2; exit 3; }

settings_payload="$(
    cd "$DRPT_REPO_ROOT" &&
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DRPT_REPO_ROOT:${PYTHONPATH:-}" \
        "$DRPT_PYTHON" -c '
import importlib
module = importlib.import_module("SFT.data.dolci32k.profile")
print("\t".join(module.SETTING_ORDER))
'
)"
IFS=$'\t' read -r -a settings <<< "$settings_payload"
(( ${#settings[@]} == 5 )) || {
    echo "ERROR: dolci32k registry must define exactly 5 settings" >&2; exit 2; }
methods=(GlobalRaw GlobalOptA)

echo "Campaign: $CAMPAIGN_ID (extending in place)"
echo "Model profile: $MODEL_PROFILE | seed: $SEED | partition: $PARTITION | array: $array_spec"
echo "Artifact build: $ARTIFACT_BUILD_ID"
echo "Matrix: 5 settings x 2 global methods = 10 runs"
if [[ "$DRY_RUN" == "true" ]]; then
    index=0
    for setting in "${settings[@]}"; do
        for method in "${methods[@]}"; do
            echo -e "$index\t$setting\tadamw\t$method"
            index=$((index + 1))
        done
    done
fi

# Refuse to clobber completed runs. train.sh itself protects a finished run
# directory, but failing here keeps the whole array from starting by mistake.
existing_success=()
for setting in "${settings[@]}"; do
    for method in "${methods[@]}"; do
        for run_dir in "$campaign_root/$setting-$method-"*; do
            [[ -f "$run_dir/_SUCCESS" ]] && existing_success+=("$(basename "$run_dir")")
        done
    done
done
if (( ${#existing_success[@]} > 0 )) && [[ "$RETRY_FAILED" != "true" ]]; then
    echo "ERROR: completed global-axis runs already exist in $campaign_root:" >&2
    printf '  %s\n' "${existing_success[@]}" >&2
    echo "Completed runs are immutable; pass --array for the missing cells only." >&2
    exit 3
fi

# Immutable snapshot for this array, separate from the campaign's original
# `source/` tree (which predates the windowed global selection path).
JOB_REPO_ROOT="$DRPT_REPO_ROOT"
snapshot_root="$campaign_root/source_global_axis"
compute_snapshot_hash() {
    "$DRPT_PYTHON" -c \
        'import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
h = hashlib.sha256()
for file in sorted(p.rglob("*")):
    if not file.is_file() or file.name == ".complete":
        continue
    relative = str(file.relative_to(p)).encode("utf-8")
    payload = file.read_bytes()
    h.update(len(relative).to_bytes(8, "big"))
    h.update(relative)
    h.update(len(payload).to_bytes(8, "big"))
    h.update(payload)
print(h.hexdigest())' \
        "$1"
}
if [[ "$DRY_RUN" != "true" ]]; then
    if [[ "$REFRESH_SNAPSHOT" == "true" && -d "$snapshot_root" ]]; then
        if (( ${#existing_success[@]} > 0 )); then
            echo "ERROR: refusing to refresh a snapshot that completed runs were produced from" >&2
            exit 3
        fi
        echo "Refreshing global-axis snapshot from the live repo: $snapshot_root"
        rm -rf "$snapshot_root"
    fi
    if [[ ! -f "$snapshot_root/.complete" ]]; then
        mkdir -p "$snapshot_root"
        rsync -a --delete \
            --exclude='/.git/***' --exclude='/SFT/runs/***' \
            --exclude='/SFT/eval/reports/***' --exclude='/logs/***' \
            --exclude='/SFT/data/dolci32k_artifacts/***' \
            --exclude='/SFT/data/dolci32k_artifacts/***' \
            --include='*/' --include='*.py' --include='*.sh' \
            --include='*.yaml' --include='*.yml' --include='*.json' \
            --include='*.txt' --include='*.md' --exclude='*' \
            "$DRPT_REPO_ROOT/" "$snapshot_root/"
        printf '%s\n' "$(compute_snapshot_hash "$snapshot_root")" \
            > "$snapshot_root/.complete"
    else
        expected_tree_hash="$(<"$snapshot_root/.complete")"
        actual_tree_hash="$(compute_snapshot_hash "$snapshot_root")"
        if [[ "$actual_tree_hash" != "$expected_tree_hash" ]]; then
            echo "ERROR: immutable global-axis snapshot was modified: $snapshot_root" >&2
            echo "expected=$expected_tree_hash actual=$actual_tree_hash" >&2
            exit 3
        fi
    fi
    JOB_REPO_ROOT="$snapshot_root"
fi

path_export_args="DRPT_REPO_ROOT=$JOB_REPO_ROOT,DRPT_DATA_DIR=$DRPT_DATA_DIR,DRPT_RUNS_DIR=$DRPT_RUNS_DIR,DRPT_LOGS_DIR=$DRPT_LOGS_DIR,DRPT_REPORTS_DIR=$DRPT_REPORTS_DIR,DRPT_PYTHON=$DRPT_PYTHON"
common_export="ALL,$path_export_args,DRPT_SEED=$SEED"
common_export="$common_export,DRPT_WANDB_PROJECT=$WANDB_PROJECT,DRPT_TF32=$TF32"
common_export="$common_export,DRPT_ARTIFACT_BUILD_ID=$ARTIFACT_BUILD_ID,DRPT_MODEL_PROFILE=$MODEL_PROFILE"
common_export="$common_export,PYTHONDONTWRITEBYTECODE=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[[ "$RETRY_FAILED" == "true" ]] && common_export="$common_export,DRPT_RETRY_FAILED=true"

submit_array() {
    local role="$1" spec="$2" dependency="$3" max_steps="$4"
    # A 3-step gate writes a genuinely complete run, and completed runs are
    # protected from overwrite — so a smoke run sharing the formal campaign id
    # would permanently block the 2000-step run for that cell. Give the gate its
    # own namespace, exactly as submit_general_loss_comparison.sh does with
    # "${CAMPAIGN_ID}-smoke-adamw".
    local campaign="$CAMPAIGN_ID"
    [[ "$role" == "smoke" ]] && campaign="${CAMPAIGN_ID}-smoke-global-axis"
    local export_args="$common_export,DRPT_CAMPAIGN_ID=$campaign"
    local time_limit="$TIME_LIMIT"
    # A 3-step gate is model load + 3 steps + two loss evals per cell. Asking
    # for the formal 7-day window would park a long reservation in front of the
    # queue this array is meant to clear quickly.
    [[ "$role" == "smoke" ]] && time_limit="08:00:00"
    [[ -n "$max_steps" ]] && export_args="$export_args,DRPT_MAX_STEPS=$max_steps"
    local cmd=(
        sbatch --parsable
        --array="$spec"
        --partition="$PARTITION"
        --gres=gpu:1
        --ntasks=8
        --mem="$MEMORY"
        --qos="$QOS"
        --time="$time_limit"
        --job-name="dolci-gax-$role"
        --output="$DRPT_LOGS_DIR/dolci32k_global_axis_${role}_%A_%a.out"
        --chdir="$JOB_REPO_ROOT"
        --export="$export_args"
    )
    [[ -n "$dependency" ]] && cmd+=(--dependency="$dependency")
    cmd+=(SFT/train/dolci32k_global_axis_job.sh)
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '[DRY-RUN]' >&2; printf ' %q' "${cmd[@]}" >&2; echo >&2
        echo "DRYRUN_${role}"
        return 0
    fi
    local job_id
    job_id="$("${cmd[@]}")"
    echo "${job_id%%;*}"
}

if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$DRPT_LOGS_DIR" "$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID"
    # Rotate any previous gate out of the way. A completed 3-step run is
    # "protected" by train.sh and would be skipped rather than re-run, so
    # reusing the namespace could pass the gate without ever exercising the
    # code being gated.
    smoke_root="$DRPT_RUNS_DIR/campaigns/${CAMPAIGN_ID}-smoke-global-axis"
    if [[ "$RUN_SMOKE" == "true" && -d "$smoke_root" ]]; then
        rotated="$smoke_root-$(date -u +%Y%m%dT%H%M%SZ)"
        echo "Rotating previous smoke namespace to $(basename "$rotated")"
        mv "$smoke_root" "$rotated"
    fi
fi

smoke_job_id=""
main_dependency=""
if [[ "$RUN_SMOKE" == "true" ]]; then
    smoke_job_id="$(submit_array smoke "${array_spec%%%*}%1" "" 3)"
    echo "Submitted 3-step smoke array: $smoke_job_id"
    main_dependency="afterok:$smoke_job_id"
fi
main_job_id="$(submit_array main "$array_spec" "$main_dependency" "")"
echo "Submitted formal array: $main_job_id"

if [[ -n "$HOLD_JOBS" && "$DRY_RUN" != "true" ]]; then
    IFS=',' read -r -a hold_ids <<< "$HOLD_JOBS"
    for hold_id in "${hold_ids[@]}"; do
        if scontrol update "jobid=$hold_id" "dependency=afterany:$main_job_id"; then
            echo "Deferred job $hold_id behind $main_job_id"
        else
            echo "WARNING: could not defer job $hold_id; check the queue by hand" >&2
        fi
    done
elif [[ -n "$HOLD_JOBS" ]]; then
    echo "[DRY-RUN] would run: scontrol update jobid={$HOLD_JOBS} dependency=afterany:<main>" >&2
fi

if [[ "$DRY_RUN" == "true" ]]; then
    exit 0
fi

manifest="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/global_axis_submission.tsv"
git_revision="$(git -C "$DRPT_REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
snapshot_hash="$(<"$snapshot_root/.complete")"
header="campaign_id	profile	family	role	seed	array	job_id	dependency	memory	max_concurrent	max_steps	retry_failed	snapshot_sha256	artifact_build_id	git_revision	model_profile"
if [[ ! -f "$manifest" ]]; then
    printf '%b\n' "$header" > "$manifest"
fi
{
    if [[ -n "$smoke_job_id" ]]; then
        printf '%s\n' "$CAMPAIGN_ID	dolci32k	adamw-global-axis	smoke	$SEED	${array_spec%%%*}%1	$smoke_job_id	none	$MEMORY	1	3	$RETRY_FAILED	$snapshot_hash	$ARTIFACT_BUILD_ID	$git_revision	$MODEL_PROFILE"
    fi
    printf '%s\n' "$CAMPAIGN_ID	dolci32k	adamw-global-axis	main	$SEED	$array_spec	$main_job_id	${main_dependency:-none}	$MEMORY	$MAX_CONCURRENT	full	$RETRY_FAILED	$snapshot_hash	$ARTIFACT_BUILD_ID	$git_revision	$MODEL_PROFILE"
} >> "$manifest"
echo "Submission manifest: $manifest"
