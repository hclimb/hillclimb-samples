#!/bin/bash
#SBATCH --job-name=loss52
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=normal
#SBATCH --time=3-00:00:00
#SBATCH --array=0-51%5
#SBATCH --output=logs/loss52_%A_%a.out

set -euo pipefail

REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
CAMPAIGN_ID="${DRPT_CAMPAIGN_ID:?DRPT_CAMPAIGN_ID must be exported by the launcher}"
SEED="${DRPT_SEED:-42}"
WANDB_PROJECT="${DRPT_WANDB_PROJECT:-drpt_opus}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"

settings=(alpaca_samsum less_squad less_tydiqa triviaqa_nq)
adamw_methods=(
    FullTraining
    GlobalRaw
    LayerwiseRaw
    GlobalOptA
    LayerwiseOptA
    GlobalSoft
    LayerwiseSoft
)
muon_methods=(
    FullTraining
    GlobalRaw
    LayerwiseRaw
    LayerwiseHybridMuonSur
    GlobalSoft
    LayerwiseSoft
)

# New campaigns use two independent arrays. ``muon`` denotes the requested
# Muon-matrix-score family: official torch.optim.Muon updates eligible matrices
# and auxiliary AdamW updates parameters that official Muon does not accept.
muon_matrix_methods=(
    FullTraining
    GlobalRaw
    LayerwiseRaw
    LayerwiseMuonSur
    GlobalSoft
    LayerwiseSoft
)

# Future/default focused campaign.  This is intentionally versioned separately
# from ``loss52`` because Slurm array jobs read this file when each task starts;
# already-submitted tasks without DRPT_COMPARISON_PROFILE must keep the old
# 7-method AdamW / 6-method Muon index mapping above.
baseline9_adamw_methods=(
    FullTraining
    LayerwiseRaw
    LayerwiseSoft
    LayerwiseSoftP
    LayerwiseOptA
)
baseline9_muon_methods=(
    FullTraining
    LayerwiseRaw
    LayerwiseSoft
    LayerwiseSoftP
    LayerwiseMuonSur
    LayerwiseMuonPSur
    LayerwiseMuonSatSur
    LayerwiseMuonSatPSur
)

dolci32k_settings=()
dolci32k_adamw_methods=()
dolci32k_muon_methods=()

optimizer_family="${DRPT_OPTIMIZER_FAMILY:-legacy-combined}"
comparison_profile="${DRPT_COMPARISON_PROFILE:-loss52}"

case "$comparison_profile" in
    loss52|baseline9|dolci32k) ;;
    *)
        echo "ERROR: DRPT_COMPARISON_PROFILE must be loss52, baseline9, or dolci32k, got $comparison_profile" >&2
        exit 2
        ;;
esac

if [[ "$comparison_profile" == "dolci32k" ]]; then
    registry_python="${DRPT_PYTHON:-python}"
    if ! registry_payload="$(
        cd "$REPO_ROOT" &&
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$registry_python" -c '
from SFT.data.dolci32k.profile import ADAMW_METHODS, MUON_METHODS, SETTING_ORDER
print("settings\t" + "\t".join(SETTING_ORDER))
print("adamw\t" + "\t".join(ADAMW_METHODS))
print("muon\t" + "\t".join(MUON_METHODS))
'
    )"; then
        echo "ERROR: failed to load the canonical $comparison_profile registry" >&2
        exit 2
    fi
    while IFS=$'\t' read -r -a registry_fields; do
        registry_kind="${registry_fields[0]:-}"
        registry_values=("${registry_fields[@]:1}")
        case "$registry_kind" in
            settings) dolci32k_settings=("${registry_values[@]}") ;;
            adamw) dolci32k_adamw_methods=("${registry_values[@]}") ;;
            muon) dolci32k_muon_methods=("${registry_values[@]}") ;;
            *) echo "ERROR: malformed dolci32k registry row: $registry_kind" >&2; exit 2 ;;
        esac
    done <<< "$registry_payload"
    if (( ${#dolci32k_settings[@]} != 5 || ${#dolci32k_adamw_methods[@]} != 5 || ${#dolci32k_muon_methods[@]} != 8 )); then
        echo "ERROR: dolci32k registry must define exactly 5 settings, 5 AdamW methods, and 8 Muon methods" >&2
        exit 2
    fi
fi

# The setting list is profile-scoped. loss52/baseline9 keep the original four so
# already-submitted arrays retain their exact index -> (setting, method) mapping.
active_settings=("${settings[@]}")
if [[ "$comparison_profile" == "dolci32k" ]]; then
    active_settings=("${dolci32k_settings[@]}")
fi

case "$optimizer_family" in
    adamw)
        optimizer_type="adamw"
        if [[ "$comparison_profile" == "dolci32k" ]]; then
            method_count=${#dolci32k_adamw_methods[@]}
            methods=("${dolci32k_adamw_methods[@]}")
        elif [[ "$comparison_profile" == "baseline9" ]]; then
            method_count=${#baseline9_adamw_methods[@]}
            methods=("${baseline9_adamw_methods[@]}")
        else
            method_count=${#adamw_methods[@]}
            methods=("${adamw_methods[@]}")
        fi
        task_count=$((${#active_settings[@]} * method_count))
        if (( TASK_ID < 0 || TASK_ID >= task_count )); then
            echo "ERROR: AdamW/$comparison_profile array task must be in [0, $((task_count - 1))], got $TASK_ID" >&2
            exit 2
        fi
        setting="${active_settings[$((TASK_ID / method_count))]}"
        method="${methods[$((TASK_ID % method_count))]}"
        ;;
    muon)
        optimizer_type="muon"
        if [[ "$comparison_profile" == "dolci32k" ]]; then
            method_count=${#dolci32k_muon_methods[@]}
            methods=("${dolci32k_muon_methods[@]}")
        elif [[ "$comparison_profile" == "baseline9" ]]; then
            method_count=${#baseline9_muon_methods[@]}
            methods=("${baseline9_muon_methods[@]}")
        else
            method_count=${#muon_matrix_methods[@]}
            methods=("${muon_matrix_methods[@]}")
        fi
        task_count=$((${#active_settings[@]} * method_count))
        if (( TASK_ID < 0 || TASK_ID >= task_count )); then
            echo "ERROR: Muon/$comparison_profile array task must be in [0, $((task_count - 1))], got $TASK_ID" >&2
            exit 2
        fi
        setting="${active_settings[$((TASK_ID / method_count))]}"
        method="${methods[$((TASK_ID % method_count))]}"
        ;;
    legacy-combined)
        # Compatibility path for already-submitted 0-51 arrays whose exported
        # environment predates DRPT_OPTIMIZER_FAMILY.
        if (( TASK_ID < 0 || TASK_ID >= 52 )); then
            echo "ERROR: legacy array task must be in [0, 51], got $TASK_ID" >&2
            exit 2
        fi
        setting_index=$((TASK_ID / 13))
        method_index=$((TASK_ID % 13))
        setting="${settings[$setting_index]}"
        if (( method_index < 7 )); then
            optimizer_type="adamw"
            method="${adamw_methods[$method_index]}"
        else
            optimizer_type="hybrid"
            method="${muon_methods[$((method_index - 7))]}"
        fi
        ;;
    *)
        echo "ERROR: DRPT_OPTIMIZER_FAMILY must be adamw or muon, got $optimizer_family" >&2
        exit 2
        ;;
esac

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TF32="${DRPT_TF32:-True}"
if [[ "$TF32" != "True" && "$TF32" != "False" ]]; then
    echo "ERROR: DRPT_TF32 must be exactly True or False, got $TF32" >&2
    exit 2
fi

config_path="configs/$setting"
[[ "$comparison_profile" == "dolci32k" ]] && config_path="configs/dolci32k/$setting"
args=(
    -c "$config_path"
    -m "$method"
    --seed "$SEED"
    --optimizer_type "$optimizer_type"
    --campaign_id "$CAMPAIGN_ID"
    --report_to wandb
    --wandb_project "$WANDB_PROJECT"
)
if [[ "$comparison_profile" == "dolci32k" ]]; then
    artifact_build_id="${DRPT_ARTIFACT_BUILD_ID:?immutable 32k worker requires DRPT_ARTIFACT_BUILD_ID from the campaign launcher}"
    args+=(--artifact-build-id "$artifact_build_id" --lr 1e-5)
fi
if [[ "$comparison_profile" == "dolci32k" ]]; then
    model_profile="${DRPT_MODEL_PROFILE:?dolci32k worker requires DRPT_MODEL_PROFILE}"
    args+=(--model-profile "$model_profile")
fi
if [[ "$comparison_profile" == "dolci32k" && "$optimizer_family" == "muon" ]]; then
    args+=(--muon-lr 3e-4 --aux-adamw-lr 1e-5)
fi
if [[ "$comparison_profile" == "dolci32k" && "${DRPT_RETRY_FAILED:-false}" == "true" ]]; then
    args+=(--retry-failed)
fi
if [[ -n "${DRPT_MAX_STEPS:-}" ]]; then
    args+=(--max_steps "$DRPT_MAX_STEPS")
fi
if [[ "${DRPT_DRY_RUN:-false}" == "true" ]]; then
    args+=(--dry-run)
fi

cd "$REPO_ROOT"
echo "[comparison] profile=$comparison_profile campaign=$CAMPAIGN_ID family=$optimizer_family array_task=$TASK_ID setting=$setting optimizer=$optimizer_type method=$method seed=$SEED TF32=$TF32"
if [[ "$comparison_profile" == "baseline9" ]]; then
    echo "[comparison] LayerwiseSoft uses capped-simplex sum(w)=4; LayerwiseSoftP uses probability-simplex sum(w)=1"
elif [[ "$comparison_profile" == "dolci32k" ]]; then
    # Logical window is fixed at N=16 candidates / T=2 target; the per-setting
    # compute chunk sizes (C) are printed by train.sh's own Window: line.
    echo "[comparison] dolci32k/$DRPT_MODEL_PROFILE: logical N=16, target T=2; hard/Soft mass=8, SoftP mass=1"
else
    echo "[comparison] soft constraint for Soft methods: 0<=w<=1, sum(w)=4 for a full batch of 8"
fi
exec bash SFT/train/train.sh "${args[@]}"
