#!/bin/bash
# Submit legacy comparisons or the canonical Dolci32k campaign.
#
# AdamW: 4 settings x 5 methods = 20 runs.
# Muon:  4 settings x 8 methods = 32 runs. "Muon" here means official
# torch.optim.Muon for eligible matrices plus auxiliary AdamW for ineligible
# parameter shapes; the four surrogate scores use Muon-managed matrices only.
# Legacy families get independent reports. Dolci32k additionally enforces the
# AdamW-first dependency and creates a combined report after Muon terminates.

set -euo pipefail

_repo_root="${DRPT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$_repo_root/cluster_env.sh" \
    || { echo "ERROR: $_repo_root/cluster_env.sh not found."; exit 1; }
unset _repo_root
path_export_args="DRPT_REPO_ROOT=$DRPT_REPO_ROOT,DRPT_DATA_DIR=$DRPT_DATA_DIR,DRPT_RUNS_DIR=$DRPT_RUNS_DIR,DRPT_LOGS_DIR=$DRPT_LOGS_DIR,DRPT_REPORTS_DIR=$DRPT_REPORTS_DIR,DRPT_PYTHON=$DRPT_PYTHON"
CAMPAIGN_ID="baseline9-layerwise-s42"
COMPARISON_PROFILE="baseline9"
SEED=42
OPTIMIZER_FAMILY="all"
ARRAY_RANGE=""
MAX_CONCURRENT=5
MAX_CONCURRENT_SET=false
PARTITION="$DRPT_SLURM_PARTITION"
QOS="$DRPT_SLURM_QOS"
TIME_LIMIT="3-00:00:00"
TIME_LIMIT_SET=false
MAX_STEPS=""
WANDB_PROJECT="drpt_opus"
TF32="True"
DRY_RUN=false
SUBMIT_REPORT=true
RUN_SMOKE=true
RETRY_FAILED=false
MODEL_PROFILE="qwen3_1_7b"

usage() {
    cat <<'EOF'
Submit an AdamW/Muon general-validation-loss comparison campaign.

The default submits two independent arrays and two independent reports:
  AdamW: 4 tasks x 5 methods = 20 runs
  Muon:  4 tasks x 8 methods = 32 runs (four Muon-matrix-only surrogates)

The dolci32k profile submits AdamW first (5 settings x 5 = 25). After that
array reaches any terminal state, Muon (5 x 8 = 40) becomes eligible.

Options:
  --campaign-id ID       Isolated runs/campaigns/ID namespace
  --profile NAME         baseline9 (default), dolci32k, or loss52
  --model-profile NAME   dolci32k model: olmo3_7b, qwen3_1_7b, qwen3_4b, or qwen3_8b
  --family NAME          all, adamw, or muon (default: all)
  --seed N               Training seed (default: 42)
  --array RANGE          Override array IDs/ranges (comma-separated allowed)
                         for a single selected family
  --max-concurrent N|auto
                         Maximum running tasks, or let Slurm fill every
                         allocatable whole-GPU slot dynamically (default: 5)
  --max-steps N          Optional short-run override for smoke testing
  --partition NAME       Slurm partition (default: DRPT_SLURM_PARTITION)
  --qos NAME             Slurm QoS (default: DRPT_SLURM_QOS)
  --time LIMIT           Slurm time limit (default: 3-00:00:00)
  --wandb-project NAME   W&B project (default: drpt_opus)
  --tf32 BOOL            Use TF32 uniformly (default: True)
  --no-report            Do not submit optimizer-specific plotting jobs
  --no-smoke             Skip smoke only for --max-steps development arrays;
                         formal dolci32k arrays always require the 3-step gate
  --retry-failed         Permit dolci32k workers to retry failed (never completed) runs
  --dry-run              Print the resolved matrix and sbatch commands

Use --family adamw or --family muon to launch/replot one optimizer family
without waiting for the other family.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
        --profile) COMPARISON_PROFILE="$2"; shift 2 ;;
        --model-profile) MODEL_PROFILE="$2"; shift 2 ;;
        --family|--optimizer-family) OPTIMIZER_FAMILY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --array) ARRAY_RANGE="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; MAX_CONCURRENT_SET=true; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; TIME_LIMIT_SET=true; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --tf32) TF32="$2"; shift 2 ;;
        --no-report) SUBMIT_REPORT=false; shift ;;
        --no-smoke) RUN_SMOKE=false; shift ;;
        --retry-failed) RETRY_FAILED=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "ERROR: invalid campaign id: $CAMPAIGN_ID" >&2; exit 2;
}
[[ "$OPTIMIZER_FAMILY" == "all" || "$OPTIMIZER_FAMILY" == "adamw" || "$OPTIMIZER_FAMILY" == "muon" ]] || {
    echo "ERROR: family must be all, adamw, or muon" >&2; exit 2;
}
[[ "$COMPARISON_PROFILE" == "baseline9" || "$COMPARISON_PROFILE" == "loss52" \
   || "$COMPARISON_PROFILE" == "dolci32k" ]] || {
    echo "ERROR: profile must be baseline9, loss52, or dolci32k" >&2; exit 2;
}
if [[ "$COMPARISON_PROFILE" == "dolci32k" ]]; then
    case "$MODEL_PROFILE" in olmo3_7b|qwen3_1_7b|qwen3_4b|qwen3_8b) ;; *)
        echo "ERROR: invalid dolci32k model profile: $MODEL_PROFILE" >&2; exit 2 ;;
    esac
fi
IS_32K_PROFILE=false
[[ "$COMPARISON_PROFILE" == "dolci32k" ]] && IS_32K_PROFILE=true
CAMPAIGN_MODEL_PROFILE="$MODEL_PROFILE"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: seed must be non-negative" >&2; exit 2; }
[[ "$MAX_CONCURRENT" == "auto" || "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: max-concurrent must be a positive integer or auto" >&2; exit 2;
}
if [[ -n "$MAX_STEPS" && ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: max-steps must be positive" >&2
    exit 2
fi
[[ "$TF32" == "True" || "$TF32" == "False" ]] || {
    echo "ERROR: tf32 must be exactly True or False" >&2; exit 2;
}
if [[ -n "$ARRAY_RANGE" && ! "$ARRAY_RANGE" =~ ^([0-9]+(-[0-9]+)?)(,([0-9]+(-[0-9]+)?))*(%[1-9][0-9]*)?$ ]]; then
    echo "ERROR: unsupported array range: $ARRAY_RANGE" >&2; exit 2
fi
if [[ "$OPTIMIZER_FAMILY" == "all" && -n "$ARRAY_RANGE" ]]; then
    echo "ERROR: --array is ambiguous with --family all; select adamw or muon" >&2
    exit 2
fi

campaign_root="$DRPT_RUNS_DIR/campaigns/$CAMPAIGN_ID"
dolci_control_root="$DRPT_REPO_ROOT"
if [[ "$IS_32K_PROFILE" == "true" && -f "$campaign_root/source/.complete" ]]; then
    dolci_control_root="$campaign_root/source"
fi
run_dolci_prepare() {
    (
        cd "$dolci_control_root"
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$dolci_control_root:${PYTHONPATH:-}" \
            "$DRPT_PYTHON" "SFT/data/prepare_dolci32k.py" "$@"
    )
}

settings=(alpaca_samsum less_squad less_tydiqa triviaqa_nq)
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    if ! registry_payload="$(
        cd "$dolci_control_root" &&
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$dolci_control_root:${PYTHONPATH:-}" "$DRPT_PYTHON" -c '
from SFT.data.dolci32k.profile import ADAMW_METHODS, MUON_METHODS, SETTING_ORDER
print("settings\t" + "\t".join(SETTING_ORDER))
print("adamw\t" + "\t".join(ADAMW_METHODS))
print("muon\t" + "\t".join(MUON_METHODS))
'
    )"; then
        echo "ERROR: failed to load the canonical $COMPARISON_PROFILE registry" >&2
        exit 2
    fi
    settings=()
    adamw_methods=()
    muon_methods=()
    while IFS=$'\t' read -r -a registry_fields; do
        registry_kind="${registry_fields[0]:-}"
        registry_values=("${registry_fields[@]:1}")
        case "$registry_kind" in
            settings) settings=("${registry_values[@]}") ;;
            adamw) adamw_methods=("${registry_values[@]}") ;;
            muon) muon_methods=("${registry_values[@]}") ;;
            *) echo "ERROR: malformed $COMPARISON_PROFILE registry row: $registry_kind" >&2; exit 2 ;;
        esac
    done <<< "$registry_payload"
    if (( ${#settings[@]} != 5 || ${#adamw_methods[@]} != 5 || ${#muon_methods[@]} != 8 )); then
        echo "ERROR: $COMPARISON_PROFILE registry must define exactly 5 settings, 5 AdamW methods, and 8 Muon methods" >&2
        exit 2
    fi
    adamw_default_array="0-24"
    muon_default_array="0-39"
    soft_constraint="capped_k8_and_probability_simplex"
    [[ "$MAX_CONCURRENT_SET" == "true" ]] || MAX_CONCURRENT=4
    [[ "$TIME_LIMIT_SET" == "true" ]] || TIME_LIMIT="7-00:00:00"
    [[ "$TF32" == "True" ]] || {
        echo "ERROR: $COMPARISON_PROFILE fixes TF32=True; got --tf32 $TF32" >&2
        exit 2
    }
    if [[ "$RUN_SMOKE" != "true" && -z "$MAX_STEPS" ]]; then
        echo "ERROR: a formal $COMPARISON_PROFILE array cannot bypass the required 3-step smoke gate" >&2
        exit 2
    fi
    [[ "$MAX_CONCURRENT" != "auto" && "$MAX_CONCURRENT" -le 4 ]] || {
        echo "ERROR: $COMPARISON_PROFILE permits at most four concurrent single-GPU jobs" >&2
        exit 2
    }
    if [[ "$ARRAY_RANGE" == *%* ]]; then
        requested_array_concurrency="${ARRAY_RANGE##*%}"
        ARRAY_RANGE="${ARRAY_RANGE%%%*}"
        if (( requested_array_concurrency < MAX_CONCURRENT )); then
            MAX_CONCURRENT="$requested_array_concurrency"
        fi
    fi
    dolci_muon_light_concurrency="$MAX_CONCURRENT"
    dolci_muon_heavy_concurrency="$MAX_CONCURRENT"
    if (( dolci_muon_heavy_concurrency > 2 )); then
        dolci_muon_heavy_concurrency=2
    fi
    # Host-memory envelopes. Muon Soft/SoftP retain one CPU bf16 candidate
    # factor pair per layer for the whole logical window, so their footprint
    # scales with N; the 16-candidate window doubles what the 8-candidate
    # window needed. Everything else stays bounded by the microbatch.
    dolci_adamw_memory="${DRPT_DOLCI_ADAMW_MEM:-48G}"
    dolci_muon_light_memory="${DRPT_DOLCI_MUON_LIGHT_MEM:-48G}"
    dolci_muon_heavy_memory="${DRPT_DOLCI_MUON_HEAVY_MEM:-192G}"
elif [[ "$COMPARISON_PROFILE" == "baseline9" ]]; then
    adamw_methods=(FullTraining LayerwiseRaw LayerwiseSoft LayerwiseSoftP LayerwiseOptA)
    muon_methods=(
        FullTraining LayerwiseRaw LayerwiseSoft LayerwiseSoftP
        LayerwiseMuonSur LayerwiseMuonPSur
        LayerwiseMuonSatSur LayerwiseMuonSatPSur
    )
    adamw_default_array="0-19"
    muon_default_array="0-31"
    soft_constraint="capped_and_probability_simplex"
else
    adamw_methods=(FullTraining GlobalRaw LayerwiseRaw GlobalOptA LayerwiseOptA GlobalSoft LayerwiseSoft)
    muon_methods=(FullTraining GlobalRaw LayerwiseRaw LayerwiseMuonSur GlobalSoft LayerwiseSoft)
    adamw_default_array="0-27"
    muon_default_array="0-23"
    soft_constraint="capped_simplex_k4"
fi
log_prefix="$COMPARISON_PROFILE"

if [[ "$OPTIMIZER_FAMILY" == "all" ]]; then
    families=(adamw muon)
else
    families=("$OPTIMIZER_FAMILY")
fi

echo "Campaign: $CAMPAIGN_ID"
echo "Profile: $COMPARISON_PROFILE | seed: $SEED | partition: $PARTITION | family: $OPTIMIZER_FAMILY | TF32: $TF32"
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    echo "Soft constraints: LayerwiseSoft capped k=8 at batch_size=16; LayerwiseSoftP probability simplex"
elif [[ "$COMPARISON_PROFILE" == "baseline9" ]]; then
    echo "Soft constraints: LayerwiseSoft capped k=4; LayerwiseSoftP probability simplex"
else
    echo "Soft constraint: capped probability simplex, k=4 at batch_size=8"
fi
_adamw_runs=$((${#settings[@]} * ${#adamw_methods[@]}))
_muon_runs=$((${#settings[@]} * ${#muon_methods[@]}))
echo "Default matrix: AdamW $_adamw_runs + Muon $_muon_runs = $((_adamw_runs + _muon_runs)) runs"

if [[ "$DRY_RUN" == "true" ]]; then
    for family in "${families[@]}"; do
        if [[ "$family" == "adamw" ]]; then
            methods=("${adamw_methods[@]}")
            runtime_optimizer="adamw"
        else
            methods=("${muon_methods[@]}")
            runtime_optimizer="muon"
        fi
        for setting in "${settings[@]}"; do
            for method in "${methods[@]}"; do
                echo -e "$setting\t$family\t$runtime_optimizer\t$method"
            done
        done
    done
fi

if [[ -e "$campaign_root" ]]; then
    if [[ "$IS_32K_PROFILE" != "true" ]]; then
        echo "ERROR: campaign path already exists; refusing to overwrite: $campaign_root" >&2
        exit 3
    fi
    echo "Reusing $COMPARISON_PROFILE campaign root for a missing optimizer family: $campaign_root"
fi

dolci_artifact_build_id=""
dolci_artifact_audited=false
dolci_tokenizers_ready=false
dolci_artifact_pin="$campaign_root/${COMPARISON_PROFILE}_artifact_build_id.txt"
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    if [[ -f "$dolci_artifact_pin" ]]; then
        dolci_artifact_build_id="$(<"$dolci_artifact_pin")"
        [[ "$dolci_artifact_build_id" =~ ^[0-9a-f]+$ ]] || {
            echo "ERROR: invalid $COMPARISON_PROFILE artifact pin: $dolci_artifact_pin" >&2
            exit 3
        }
    elif [[ -d "$campaign_root" ]] && \
         find "$campaign_root" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        echo "ERROR: existing campaign lacks its immutable $COMPARISON_PROFILE artifact pin: $campaign_root" >&2
        exit 3
    fi
fi

if [[ "$DRY_RUN" != "true" ]]; then
    # Prepare public datasets once on the login/submission node. Array workers
    # validate this exact pinned build rather than following a mutable CURRENT.
    activate_env
    if [[ "$IS_32K_PROFILE" == "true" ]]; then
        if ! dolci_artifact_root="$(
            run_dolci_prepare --data-dir "$DRPT_DATA_DIR" --print-artifact-root
        )" || [[ "$dolci_artifact_root" != /* || "$dolci_artifact_root" == *$'\n'* ]]; then
            echo "ERROR: failed to resolve the canonical $COMPARISON_PROFILE artifact root" >&2
            exit 3
        fi
        if [[ -z "$dolci_artifact_build_id" ]]; then
            current_pointer="$dolci_artifact_root/CURRENT"
            if [[ -f "$current_pointer" ]]; then
                candidate_build_id="$(<"$current_pointer")"
                if [[ "$candidate_build_id" =~ ^[0-9a-f]+$ ]] && \
                   run_dolci_prepare --data-dir "$DRPT_DATA_DIR" --audit-only \
                       --build-id "$candidate_build_id"; then
                    dolci_artifact_build_id="$candidate_build_id"
                    dolci_artifact_audited=true
                    echo "Reusing audited $COMPARISON_PROFILE build: $dolci_artifact_build_id"
                fi
            fi
            if [[ -z "$dolci_artifact_build_id" ]]; then
                run_dolci_prepare --data-dir "$DRPT_DATA_DIR" --build \
                    --profile-tokenizers all
                dolci_tokenizers_ready=true
            fi
            [[ -f "$current_pointer" ]] || {
                echo "ERROR: $COMPARISON_PROFILE builder did not create $current_pointer" >&2
                exit 3
            }
            [[ -n "$dolci_artifact_build_id" ]] || dolci_artifact_build_id="$(<"$current_pointer")"
            [[ "$dolci_artifact_build_id" =~ ^[0-9a-f]+$ ]] || {
                echo "ERROR: $COMPARISON_PROFILE builder produced invalid build id: $dolci_artifact_build_id" >&2
                exit 3
            }
            mkdir -p "$campaign_root"
            pin_tmp="$dolci_artifact_pin.tmp.$$"
            printf '%s\n' "$dolci_artifact_build_id" > "$pin_tmp"
            mv "$pin_tmp" "$dolci_artifact_pin"
        fi
        if [[ "$dolci_artifact_audited" != "true" ]]; then
            run_dolci_prepare --data-dir "$DRPT_DATA_DIR" --audit-only \
                --build-id "$dolci_artifact_build_id"
        fi
        # Raw membership is model-independent, but Dolci training fails closed
        # unless the derived diagnostics for every supported tokenizer are
        # persisted for this exact build. Existing campaign pins and reused
        # CURRENT builds therefore receive the same preflight bundle as a new
        # build; matching Parquet caches are reused by the profiler.
        if [[ "$dolci_tokenizers_ready" != "true" ]]; then
            run_dolci_prepare --data-dir "$DRPT_DATA_DIR" --audit-only \
                --build-id "$dolci_artifact_build_id" \
                --profile-tokenizers all
            dolci_tokenizers_ready=true
        fi
    else
        "$DRPT_PYTHON" "$DRPT_REPO_ROOT/SFT/data/prepare_baseline9.py" \
            --data-dir "$DRPT_DATA_DIR"
    fi
elif [[ "$IS_32K_PROFILE" == "true" && -z "$dolci_artifact_build_id" ]]; then
    dolci_artifact_build_id="${DRPT_ARTIFACT_BUILD_ID:-CURRENT}"
fi

# dolci32k jobs run from an immutable source snapshot.  Generated datasets,
# checkpoints, logs, reports, caches, and git metadata remain outside it via
# the explicit DRPT_* paths exported below.
JOB_REPO_ROOT="$DRPT_REPO_ROOT"
if [[ "$IS_32K_PROFILE" == "true" && "$DRY_RUN" != "true" ]]; then
    snapshot_root="$campaign_root/source"
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
    if [[ ! -f "$snapshot_root/.complete" ]]; then
        mkdir -p "$snapshot_root"
        dolci_snapshot_excludes=()
        if [[ "$dolci_artifact_root" == "$DRPT_REPO_ROOT"/* ]]; then
            dolci_artifact_relative="${dolci_artifact_root#"$DRPT_REPO_ROOT"/}"
            dolci_snapshot_excludes+=(--exclude="/$dolci_artifact_relative/***")
        fi
        rsync -a --delete \
            --exclude='/.git/***' --exclude='/SFT/runs/***' \
            --exclude='/SFT/eval/reports/***' --exclude='/logs/***' \
            "${dolci_snapshot_excludes[@]}" \
            --include='*/' --include='*.py' --include='*.sh' \
            --include='*.yaml' --include='*.yml' --include='*.json' \
            --include='*.txt' --include='*.md' --exclude='*' \
            "$DRPT_REPO_ROOT/" "$snapshot_root/"
        tree_hash="$(compute_snapshot_hash "$snapshot_root")"
        printf '%s\n' "$tree_hash" > "$snapshot_root/.complete"
    else
        expected_tree_hash="$(<"$snapshot_root/.complete")"
        actual_tree_hash="$(compute_snapshot_hash "$snapshot_root")"
        if [[ "$actual_tree_hash" != "$expected_tree_hash" ]]; then
            echo "ERROR: immutable campaign snapshot was modified: $snapshot_root" >&2
            echo "expected=$expected_tree_hash actual=$actual_tree_hash" >&2
            exit 3
        fi
    fi
    JOB_REPO_ROOT="$snapshot_root"
fi
path_export_args="DRPT_REPO_ROOT=$JOB_REPO_ROOT,DRPT_DATA_DIR=$DRPT_DATA_DIR,DRPT_RUNS_DIR=$DRPT_RUNS_DIR,DRPT_LOGS_DIR=$DRPT_LOGS_DIR,DRPT_REPORTS_DIR=$DRPT_REPORTS_DIR,DRPT_PYTHON=$DRPT_PYTHON"
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    path_export_args="$path_export_args,DRPT_ARTIFACT_BUILD_ID=$dolci_artifact_build_id,PYTHONDONTWRITEBYTECODE=1"
    path_export_args="$path_export_args,DRPT_MODEL_PROFILE=$MODEL_PROFILE"
fi

declare -A train_job_ids=()
declare -A report_job_ids=()
declare -A submitted_arrays=()

# dolci32k has a deliberately different physical schedule from the legacy
# campaigns below. Keep this branch self-contained so the already published
# baseline9 array indices and resource requests cannot drift.
if [[ "$IS_32K_PROFILE" == "true" ]]; then
    dolci_manifest="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/submission.tsv"
    snapshot_hash=""
    [[ -f "$JOB_REPO_ROOT/.complete" ]] && snapshot_hash="$(<"$JOB_REPO_ROOT/.complete")"

    dolci_family_already_submitted() {
        local family="$1"
        [[ -f "$dolci_manifest" ]] || return 1
        awk -F '\t' -v wanted="$family" -v profile="$COMPARISON_PROFILE" '
            NR > 1 && $2 == profile && $3 == wanted && $4 ~ /^main/ { found=1 }
            END { exit(found ? 0 : 1) }
        ' "$dolci_manifest"
    }

    for family in "${families[@]}"; do
        if dolci_family_already_submitted "$family" && [[ "$RETRY_FAILED" != "true" ]]; then
            echo "ERROR: $COMPARISON_PROFILE family '$family' already has main-array submissions in $dolci_manifest" >&2
            echo "Completed runs are immutable; use --retry-failed only for failed run directories." >&2
            exit 3
        fi
    done

    dolci_manifest_header="campaign_id	profile	family	role	seed	array	job_id	dependency	memory	max_concurrent	max_steps	retry_failed	snapshot_sha256	artifact_build_id	git_revision	model_profile"
    if [[ "$DRY_RUN" != "true" ]]; then
        mkdir -p "$DRPT_LOGS_DIR" "$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID"
        if [[ ! -f "$dolci_manifest" ]]; then
            printf '%b\n' "$dolci_manifest_header" > "$dolci_manifest"
        elif [[ "$(head -n 1 "$dolci_manifest")" != "$(printf '%b' "$dolci_manifest_header")" ]]; then
            echo "ERROR: stale $COMPARISON_PROFILE submission manifest schema: $dolci_manifest" >&2
            exit 3
        fi
        if ! awk -F '\t' -v campaign="$CAMPAIGN_ID" -v seed="$SEED" \
            -v profile="$COMPARISON_PROFILE" -v snapshot="$snapshot_hash" \
            -v artifact="$dolci_artifact_build_id" -v model="$CAMPAIGN_MODEL_PROFILE" '
                NR > 1 && ($1 != campaign || $2 != profile || $5 != seed ||
                           $13 != snapshot || $14 != artifact || $16 != model) { bad=1 }
                END { exit(bad ? 1 : 0) }
            ' "$dolci_manifest"; then
            echo "ERROR: same-ID $COMPARISON_PROFILE continuation would mix campaign seed, artifact build, or source snapshot" >&2
            exit 3
        fi
    fi

    dolci_git_revision="unknown"
    if [[ -f "$dolci_manifest" ]]; then
        prior_git_revision="$(awk -F '\t' 'NR > 1 && $15 != "" {print $15; exit}' "$dolci_manifest")"
        if [[ -n "$prior_git_revision" ]]; then
            dolci_git_revision="$prior_git_revision"
        fi
    fi
    if [[ "$dolci_git_revision" == "unknown" ]] && git -C "$DRPT_REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
        dolci_git_revision="$(git -C "$DRPT_REPO_ROOT" rev-parse HEAD)"
    fi
    SUBMITTED_JOB_ID=""
    submit_dolci_array() {
        local role="$1" family="$2" raw_array="$3" concurrency="$4"
        local memory="$5" campaign_id="$6" dependency="$7" steps="$8"
        local array_spec="$raw_array"
        if [[ "$array_spec" != *%* && "$concurrency" != "auto" ]]; then
            array_spec="${array_spec}%${concurrency}"
        fi
        local export_args="ALL,$path_export_args,DRPT_CAMPAIGN_ID=$campaign_id,DRPT_COMPARISON_PROFILE=$COMPARISON_PROFILE,DRPT_OPTIMIZER_FAMILY=$family,DRPT_SEED=$SEED,DRPT_WANDB_PROJECT=$WANDB_PROJECT,DRPT_TF32=$TF32,DRPT_RETRY_FAILED=$RETRY_FAILED,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        [[ -n "$steps" ]] && export_args="$export_args,DRPT_MAX_STEPS=$steps"
        local command=(
            sbatch --parsable
            --array="$array_spec"
            --partition="$PARTITION"
            --gres=gpu:1
            --ntasks=8
            --mem="$memory"
            --qos="$QOS"
            --time="$TIME_LIMIT"
            --job-name="d32-${family:0:2}-${role:0:7}-${CAMPAIGN_ID:0:10}"
            --output="$DRPT_LOGS_DIR/${COMPARISON_PROFILE}_${family}_${role}_%A_%a.out"
            --chdir="$JOB_REPO_ROOT"
            --export="$export_args"
        )
        [[ -n "$dependency" ]] && command+=(--dependency="$dependency")
        command+=(SFT/train/general_loss_comparison_job.sh)

        if [[ "$DRY_RUN" == "true" ]]; then
            printf '[DRY-RUN]'; printf ' %q' "${command[@]}"; echo
            SUBMITTED_JOB_ID="<${family}-${role}-job>"
        else
            SUBMITTED_JOB_ID="$("${command[@]}")"
            SUBMITTED_JOB_ID="${SUBMITTED_JOB_ID%%;*}"
            echo "Submitted $COMPARISON_PROFILE $family $role array: $SUBMITTED_JOB_ID"
            echo -e "$CAMPAIGN_ID\t$COMPARISON_PROFILE\t$family\t$role\t$SEED\t$array_spec\t$SUBMITTED_JOB_ID\t${dependency:-none}\t$memory\t$concurrency\t${steps:-full}\t$RETRY_FAILED\t${snapshot_hash:-unknown}\t$dolci_artifact_build_id\t$dolci_git_revision\t$CAMPAIGN_MODEL_PROFILE" >> "$dolci_manifest"
        fi
    }

    DOLCI_REPORT_JOB_ID=""
    submit_dolci_report() {
        local family="$1" dependency_ids="$2"
        local report_profile="$COMPARISON_PROFILE-$family"
        [[ "$family" == "combined" ]] && report_profile="$COMPARISON_PROFILE"
        local command=(
            sbatch --parsable
            --partition="$PARTITION"
            --ntasks=1
            --cpus-per-task=2
            --qos="$QOS"
            --time=01:00:00
            --dependency="afterany:$dependency_ids"
            --job-name="plot-${family}-${CAMPAIGN_ID:0:14}"
            --output="$DRPT_LOGS_DIR/${COMPARISON_PROFILE}_${family}_plot_%j.out"
            --chdir="$JOB_REPO_ROOT"
            --export="ALL,$path_export_args,DRPT_CAMPAIGN_ID=$CAMPAIGN_ID,DRPT_REPORT_FAMILY=$report_profile,DRPT_COMPARISON_PROFILE=$COMPARISON_PROFILE,DRPT_SEED=$SEED"
            SFT/eval/plot_general_loss_campaign.sh
        )
        if [[ "$DRY_RUN" == "true" ]]; then
            printf '[DRY-RUN]'; printf ' %q' "${command[@]}"; echo
            DOLCI_REPORT_JOB_ID="<${family}-report-job>"
        else
            mkdir -p "$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/$family"
            DOLCI_REPORT_JOB_ID="$("${command[@]}")"
            DOLCI_REPORT_JOB_ID="${DOLCI_REPORT_JOB_ID%%;*}"
            echo "Submitted $COMPARISON_PROFILE $family afterany report: $DOLCI_REPORT_JOB_ID"
            echo -e "$CAMPAIGN_ID\t$COMPARISON_PROFILE\t$family\tplot\t$SEED\t-\t$DOLCI_REPORT_JOB_ID\tafterany:$dependency_ids\tcpu\t1\t-\tfalse\t${snapshot_hash:-unknown}\t$dolci_artifact_build_id\t$dolci_git_revision\t$CAMPAIGN_MODEL_PROFILE" >> "$dolci_manifest"
        fi
    }

    submit_dolci_downstream() {
        local family="$1" dependency_ids="$2"
        [[ "$SUBMIT_REPORT" == "true" && -z "$MAX_STEPS" ]] || return 0
        if [[ "$DRY_RUN" == "true" && "$dependency_ids" == *'<'* ]]; then
            echo "[DRY-RUN] downstream $family symbolic dependency: $dependency_ids"
            if [[ "$family" == "adamw" ]]; then
                dependency_ids="900001"
            else
                dependency_ids="900002:900003"
            fi
        fi
        local command=(
            bash "$JOB_REPO_ROOT/SFT/eval/submit_campaign_downstream.sh"
            --campaign-id "$CAMPAIGN_ID"
            --profile "$COMPARISON_PROFILE"
            --model-profile "$CAMPAIGN_MODEL_PROFILE"
            --family "$family"
            --dependency "$dependency_ids"
            --seed "$SEED"
            --runs-root "$DRPT_RUNS_DIR"
            --partition "$PARTITION"
            --qos "$QOS"
        )
        [[ "$DRY_RUN" == "true" ]] && command+=(--dry-run)
        "${command[@]}"
    }

    adamw_main_id=""
    adamw_post_submit=false
    muon_post_submit=false
    adamw_gate=""
    if [[ "$OPTIMIZER_FAMILY" == "all" || "$OPTIMIZER_FAMILY" == "adamw" ]]; then
        if [[ "$RUN_SMOKE" == "true" && -z "$MAX_STEPS" ]]; then
            submit_dolci_array smoke adamw "0-4" 1 "$dolci_adamw_memory" "${CAMPAIGN_ID}-smoke-adamw" "" 3
            adamw_gate="afterok:$SUBMITTED_JOB_ID"
        fi
        submit_dolci_array main adamw "${ARRAY_RANGE:-$adamw_default_array}" "$MAX_CONCURRENT" "$dolci_adamw_memory" "$CAMPAIGN_ID" "$adamw_gate" "$MAX_STEPS"
        adamw_main_id="$SUBMITTED_JOB_ID"
        train_job_ids[adamw]="$adamw_main_id"
        submitted_arrays[adamw]="${ARRAY_RANGE:-$adamw_default_array}"
        if [[ "$SUBMIT_REPORT" == "true" && -z "$MAX_STEPS" && -z "$ARRAY_RANGE" ]]; then
            adamw_post_submit=true
        fi
    fi

    # A later --family muon continuation inherits an active AdamW dependency
    # from the same campaign. Once AdamW has left squeue, no stale dependency is
    # attached (the prerequisite is already terminal).
    muon_after_adam=""
    if [[ -n "$adamw_main_id" ]]; then
        muon_after_adam="afterany:$adamw_main_id"
    elif [[ "$OPTIMIZER_FAMILY" == "muon" ]]; then
        prior_adamw_id=""
        if [[ -f "$dolci_manifest" ]]; then
            prior_adamw_id="$(awk -F '\t' -v profile="$COMPARISON_PROFILE" '
                $2==profile && $3=="adamw" && $4=="main" {
                    array=$6; sub(/%.*/, "", array)
                    if (array=="0-24") id=$7
                }
                END{print id}
            ' "$dolci_manifest")"
        fi
        if [[ -z "$prior_adamw_id" ]]; then
            echo "ERROR: Muon $COMPARISON_PROFILE submission requires a prior full AdamW 0-24 array in the same campaign." >&2
            echo "Submit --family adamw first, then continue with --family muon after that array is terminal." >&2
            exit 3
        fi
        if [[ "$DRY_RUN" != "true" ]] && squeue -h -j "$prior_adamw_id" 2>/dev/null | grep -q .; then
            muon_after_adam="afterany:$prior_adamw_id"
        fi
    fi

    if [[ "$OPTIMIZER_FAMILY" == "all" || "$OPTIMIZER_FAMILY" == "muon" ]]; then
        muon_gate="$muon_after_adam"
        if [[ "$RUN_SMOKE" == "true" && -z "$MAX_STEPS" ]]; then
            submit_dolci_array smoke muon "0-7" 1 "$dolci_muon_heavy_memory" "${CAMPAIGN_ID}-smoke-muon" "$muon_after_adam" 3
            muon_gate="afterok:$SUBMITTED_JOB_ID"
        fi

        muon_main_ids=()
        if [[ -n "$ARRAY_RANGE" ]]; then
            # A partial retry can mix methods, so use the conservative Soft/SoftP
            # resource envelope rather than guessing from a range expression.
            submit_dolci_array main-custom muon "$ARRAY_RANGE" "$dolci_muon_heavy_concurrency" "$dolci_muon_heavy_memory" "$CAMPAIGN_ID" "$muon_gate" "$MAX_STEPS"
            muon_main_ids+=("$SUBMITTED_JOB_ID")
        else
            # Canonical method-minor mapping: indices 2/3 in each group of eight
            # are Soft and SoftP. They receive the larger host-memory envelope.
            muon_light="0-1,4-9,12-17,20-25,28-33,36-39"
            muon_heavy="2-3,10-11,18-19,26-27,34-35"
            submit_dolci_array main-light muon "$muon_light" "$dolci_muon_light_concurrency" "$dolci_muon_light_memory" "$CAMPAIGN_ID" "$muon_gate" "$MAX_STEPS"
            muon_main_ids+=("$SUBMITTED_JOB_ID")
            submit_dolci_array main-heavy muon "$muon_heavy" "$dolci_muon_heavy_concurrency" "$dolci_muon_heavy_memory" "$CAMPAIGN_ID" "$muon_gate" "$MAX_STEPS"
            muon_main_ids+=("$SUBMITTED_JOB_ID")
        fi
        muon_dependency_ids="$(IFS=:; echo "${muon_main_ids[*]}")"
        train_job_ids[muon]="$muon_dependency_ids"
        submitted_arrays[muon]="${ARRAY_RANGE:-$muon_default_array}"
        if [[ "$SUBMIT_REPORT" == "true" && -z "$MAX_STEPS" && -z "$ARRAY_RANGE" ]]; then
            muon_post_submit=true
        fi
    fi

    # Optional consumers are submitted only after every requested training
    # array is safely in Slurm. AdamW downstream/report and Muon smoke both use
    # the AdamW main array as an afterany prerequisite and can then run in
    # parallel even when individual AdamW tasks failed.
    if [[ "$adamw_post_submit" == "true" ]]; then
        submit_dolci_report adamw "$adamw_main_id"
        report_job_ids[adamw]="$DOLCI_REPORT_JOB_ID"
        submit_dolci_downstream adamw "$adamw_main_id"
    fi
    if [[ "$muon_post_submit" == "true" ]]; then
        submit_dolci_report muon "$muon_dependency_ids"
        report_job_ids[muon]="$DOLCI_REPORT_JOB_ID"
        submit_dolci_downstream muon "$muon_dependency_ids"
        # By this point a full AdamW array is either terminal (family
        # continuation) or an ancestor of the Muon arrays (family=all), so the
        # Muon terminal dependency is sufficient for the integrated 13-method
        # report over both optimizer families.
        submit_dolci_report combined "$muon_dependency_ids"
        report_job_ids[combined]="$DOLCI_REPORT_JOB_ID"
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        echo "Submission manifest: $dolci_manifest"
        echo "Immutable source SHA256: $snapshot_hash"
        echo "Model profile: $CAMPAIGN_MODEL_PROFILE"
    fi
    exit 0
fi

for family in "${families[@]}"; do
    if [[ -n "$ARRAY_RANGE" ]]; then
        raw_array="$ARRAY_RANGE"
    elif [[ "$family" == "adamw" ]]; then
        raw_array="$adamw_default_array"
    else
        raw_array="$muon_default_array"
    fi
    array_spec="$raw_array"
    if [[ "$array_spec" != *%* && "$MAX_CONCURRENT" != "auto" ]]; then
        array_spec="${array_spec}%${MAX_CONCURRENT}"
    fi
    submitted_arrays["$family"]="$array_spec"

    export_args="ALL,$path_export_args,DRPT_CAMPAIGN_ID=$CAMPAIGN_ID,DRPT_COMPARISON_PROFILE=$COMPARISON_PROFILE,DRPT_OPTIMIZER_FAMILY=$family,DRPT_SEED=$SEED,DRPT_WANDB_PROJECT=$WANDB_PROJECT,DRPT_TF32=$TF32,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    [[ -n "$MAX_STEPS" ]] && export_args="$export_args,DRPT_MAX_STEPS=$MAX_STEPS"

    train_cmd=(
        sbatch --parsable
        --array="$array_spec"
        --partition="$PARTITION"
        --gres=gpu:1
        --ntasks=8
        --qos="$QOS"
        --time="$TIME_LIMIT"
        --job-name="${log_prefix:0:8}-${family}-${CAMPAIGN_ID:0:14}"
        --output="$DRPT_LOGS_DIR/${log_prefix}_${family}_%A_%a.out"
        --chdir="$DRPT_REPO_ROOT"
        --export="$export_args"
        SFT/train/general_loss_comparison_job.sh
    )

    full_array=false
    if [[ "$family" == "adamw" && "$raw_array" == "$adamw_default_array" ]] || \
       [[ "$family" == "muon" && "$raw_array" == "$muon_default_array" ]]; then
        full_array=true
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        printf '[DRY-RUN]'; printf ' %q' "${train_cmd[@]}"; echo
        if [[ "$SUBMIT_REPORT" == "true" && "$full_array" == "true" && -z "$MAX_STEPS" ]]; then
            echo "[DRY-RUN] report family=$COMPARISON_PROFILE-$family profile=$COMPARISON_PROFILE-$family dependency=afterany:<${family}-training-job-id> output=$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/$family"
        fi
        continue
    fi

    mkdir -p "$DRPT_LOGS_DIR" "$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/$family"
    train_job_id="$("${train_cmd[@]}")"
    train_job_id="${train_job_id%%;*}"
    train_job_ids["$family"]="$train_job_id"
    echo "Submitted $family training array: $train_job_id"

    if [[ "$SUBMIT_REPORT" == "true" && "$full_array" == "true" && -z "$MAX_STEPS" ]]; then
        report_job_id="$(sbatch --parsable \
            --partition="$PARTITION" \
            --ntasks=1 \
            --cpus-per-task=2 \
            --qos="$QOS" \
            --time=01:00:00 \
            --dependency="afterany:$train_job_id" \
            --job-name="plot-${family}-${CAMPAIGN_ID:0:14}" \
            --output="$DRPT_LOGS_DIR/${log_prefix}_${family}_plot_%j.out" \
            --chdir="$DRPT_REPO_ROOT" \
            --export="ALL,$path_export_args,DRPT_CAMPAIGN_ID=$CAMPAIGN_ID,DRPT_REPORT_FAMILY=$COMPARISON_PROFILE-$family,DRPT_SEED=$SEED" \
            SFT/eval/plot_general_loss_campaign.sh)"
        report_job_id="${report_job_id%%;*}"
        report_job_ids["$family"]="$report_job_id"
        echo "Submitted $family afterany report job: $report_job_id"
    else
        report_job_ids["$family"]=""
    fi
done

if [[ "$DRY_RUN" == "true" ]]; then
    exit 0
fi

manifest="$DRPT_REPORTS_DIR/campaigns/$CAMPAIGN_ID/submission.tsv"
git_revision="$(git -C "$DRPT_REPO_ROOT" rev-parse HEAD)"
{
    echo -e "campaign_id\tprofile\tfamily\tseed\tarray\ttraining_job_id\treport_job_id\tmethods\truntime_optimizer\tsoft_constraint\ttf32\tpartition\ttime_limit\tmax_steps\tstatus\tgit_revision"
    for family in "${families[@]}"; do
        if [[ "$family" == "adamw" ]]; then
            method_count=${#adamw_methods[@]}
            runtime_optimizer="adamw"
        else
            method_count=${#muon_methods[@]}
            runtime_optimizer="torch_muon_with_aux_adamw"
        fi
        echo -e "$CAMPAIGN_ID\t$COMPARISON_PROFILE\t$family\t$SEED\t${submitted_arrays[$family]}\t${train_job_ids[$family]}\t${report_job_ids[$family]}\t$method_count\t$runtime_optimizer\t$soft_constraint\t$TF32\t$PARTITION\t$TIME_LIMIT\t${MAX_STEPS:-full}\tsubmitted\t$git_revision"
    done
} > "$manifest"
echo "Submission manifest: $manifest"
