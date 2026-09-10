#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-eval}" # train or eval
SETTING="${SETTING:-triviaqa_nq}"
SEED="${SEED:-42}"
N_TEST="${N_TEST:--1}" # 500
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"
DRY_RUN="${DRY_RUN:-false}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-both}"   # adamw, muon, hybrid, or adamw+muon
REPORT_TO="${REPORT_TO:-wandb}"           # wandb or none
WANDB_PROJECT="${WANDB_PROJECT:-drpt_opus}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

usage() {
  cat <<'EOF'
Usage:
  ./sft_drpt_opus.sh [setting] [train|eval] [--dry-run]
  SETTING=<setting> MODE=<train|eval> ./sft_drpt_opus.sh

Settings (baseline9, four small tasks):
  alpaca_samsum
  less_tydiqa
  triviaqa_nq
  less_squad

Environment:
  SEED=42
  N_TEST=500
  EVAL_BATCH_SIZE=8
  OPTIMIZER_TYPE=both          # adamw, muon, hybrid, or adamw+muon
  METHODS=sft10                # sft10, adamw-core, muon-core, solver-comparison, or CSV
  REPORT_TO=wandb              # set REPORT_TO=none to disable wandb
  WANDB_PROJECT=drpt_opus
  WANDB_GROUP=<setting>-<optimizer>-s<seed>
  WANDB_TAGS=sft,drpt_opus,<setting>,<optimizer>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    train|eval)
      MODE="$1"
      shift
      ;;
    alpaca_samsum|less_tydiqa|triviaqa_nq|less_squad)
      SETTING="$1"
      shift
      ;;
    --setting)
      SETTING="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --optimizer_type|--optimizer-type)
      OPTIMIZER_TYPE="$2"
      shift 2
      ;;
    --report_to|--report-to)
      REPORT_TO="$2"
      shift 2
      ;;
    --wandb_project|--wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_group|--wandb-group)
      WANDB_GROUP="$2"
      shift 2
      ;;
    --wandb_tags|--wandb-tags)
      WANDB_TAGS="$2"
      shift 2
      ;;
    --wandb_run_name|--wandb-run-name)
      WANDB_RUN_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

case "$SETTING" in
  alpaca_samsum)
    CONFIG_DIR="configs/alpaca_samsum"
    TRAIN="alpaca"
    TASK="samsum"
    ;;
  less_tydiqa)
    CONFIG_DIR="configs/less_tydiqa"
    TRAIN="less"
    TASK="tydiqa"
    ;;
  triviaqa_nq)
    CONFIG_DIR="configs/triviaqa_nq"
    TRAIN="triviaqa"
    TASK="nq_open"
    ;;
  less_squad)
    CONFIG_DIR="configs/less_squad"
    TRAIN="less"
    TASK="squad"
    ;;
  *)
    echo "Unknown setting: $SETTING"
    usage
    exit 1
    ;;
esac

sft10_methods=(
  FullTraining
  GlobalRaw
  LayerwiseRaw
  OptGroupRaw
  GlobalOptA
  LayerwiseOptA
  OptGroupOptA
  GlobalOptB
  LayerwiseOptB
  OptGroupOptB
)

opus_baselines=(
  FullTraining
  GlobalRaw
  GlobalOptA
  LayerwiseRaw
  LayerwiseOptA
  OptGroupRaw
  OptGroupOptA
)

new_solver_methods=(
  GlobalRandom
  LayerwiseRandom
  GlobalSoft
  LayerwiseSoft
  LayerwiseSoftP
  GlobalHybridMuonSur
  LayerwiseHybridMuonSur
  GlobalMuonSur
  LayerwiseMuonSur
  LayerwiseMuonPSur
  LayerwiseMuonSatSur
  LayerwiseMuonSatPSur
)

adamw_core_methods=(
  FullTraining
  GlobalRandom
  LayerwiseRandom
  GlobalRaw
  LayerwiseRaw
  GlobalOptA
  LayerwiseOptA
  GlobalSoft
  LayerwiseSoft
)

muon_core_methods=(
  FullTraining
  LayerwiseRaw
  LayerwiseSoft
  LayerwiseSoftP
  LayerwiseMuonSur
  LayerwiseMuonPSur
  LayerwiseMuonSatSur
  LayerwiseMuonSatPSur
)

solver_comparison_methods=("${muon_core_methods[@]}")

methods_spec="${METHODS:-sft10}"
focused_optimizer=""
case "$methods_spec" in
  sft10|optimizer-ablation-10|drpt-opus-10|baselines)
    methods=("${sft10_methods[@]}")
    ;;
  opus-baselines|drpt-opus-baselines)
    methods=("${opus_baselines[@]}")
    ;;
  new-solvers)
    methods=("${new_solver_methods[@]}")
    ;;
  adamw-core|focused-adamw)
    methods=("${adamw_core_methods[@]}")
    focused_optimizer="adamw"
    ;;
  muon-core|focused-muon)
    methods=("${muon_core_methods[@]}")
    focused_optimizer="muon"
    ;;
  solver-comparison|focused-solver-comparison)
    methods=("${solver_comparison_methods[@]}")
    ;;
  *)
    IFS=',' read -ra methods <<< "$methods_spec"
    ;;
esac

if [[ "$focused_optimizer" == "adamw" ]]; then
  case "$OPTIMIZER_TYPE" in
    adamw|adamw-only|adamw_only|both|all|adamw,muon|muon,adamw)
      optimizer_types=(adamw)
      ;;
    *)
      echo "ERROR: METHODS=$methods_spec is AdamW-only; got OPTIMIZER_TYPE=$OPTIMIZER_TYPE" >&2
      exit 1
      ;;
  esac
elif [[ "$focused_optimizer" == "muon" ]]; then
  case "$OPTIMIZER_TYPE" in
    muon|both|all|adamw,muon|muon,adamw)
      optimizer_types=(muon)
      ;;
    *)
      echo "ERROR: METHODS=$methods_spec requires optimizer_type=muon; got OPTIMIZER_TYPE=$OPTIMIZER_TYPE" >&2
      exit 1
      ;;
  esac
else
  case "$OPTIMIZER_TYPE" in
    both|all|adamw,muon|muon,adamw)
      optimizer_types=(adamw muon)
      ;;
    *)
      IFS=',' read -ra optimizer_types <<< "$OPTIMIZER_TYPE"
      ;;
  esac
fi

optimizer_label() {
  case "$1" in
    adamw|adamw-only|adamw_only) echo "adamw" ;;
    *) echo "$1" ;;
  esac
}

method_label() {
  case "$1" in
    FullTraining-Full|FullTraining|Full-Training) echo "FullTraining" ;;
    GlobalSubset-Full|GlobalRaw|Global-Raw) echo "GlobalRaw" ;;
    LayerWiseSubset-Full|LayerwiseRaw|LayerWiseRaw|Layerwise-Raw|LayerWise-Raw) echo "LayerwiseRaw" ;;
    OptimizerGroupWise-Full|OptGroupRaw|OptGroup-Raw) echo "OptGroupRaw" ;;
    OptimizerAwareGlobalSubset-Full|OptimizerAwareGlobalSubset-OptA-Full|GlobalOptA|Global-OptA) echo "GlobalOptA" ;;
    LayerWiseOptimizerAwareSubset-Full|LayerWiseOptimizerAwareSubset-OptA-Full|LayerwiseOptA|LayerWiseOptA|Layerwise-OptA|LayerWise-OptA) echo "LayerwiseOptA" ;;
    OptimizerAwareGroupWise-Full|OptimizerAwareGroupWise-OptA-Full|OptGroupOptA|OptGroup-OptA) echo "OptGroupOptA" ;;
    OptimizerAwareGlobalSubset-OptB-Full|GlobalOptB|Global-OptB) echo "GlobalOptB" ;;
    LayerWiseOptimizerAwareSubset-OptB-Full|LayerwiseOptB|LayerWiseOptB|Layerwise-OptB|LayerWise-OptB) echo "LayerwiseOptB" ;;
    OptimizerAwareGroupWise-OptB-Full|OptGroupOptB|OptGroup-OptB) echo "OptGroupOptB" ;;
    GlobalRandomSubset|GlobalRandomSubset-Full|GlobalRandom) echo "GlobalRandom" ;;
    LayerWiseRandomSubset|LayerWiseRandomSubset-Full|LayerwiseRandom|LayerWiseRandom) echo "LayerwiseRandom" ;;
    GlobalSoftWeighting|GlobalSoftWeighting-Full|GlobalSoft) echo "GlobalSoft" ;;
    LayerWiseSoftWeighting|LayerWiseSoftWeighting-Full|LayerwiseSoft|LayerWiseSoft) echo "LayerwiseSoft" ;;
    GlobalMuonSpectral|GlobalMuonSpectral-Full|GlobalHybridMuonSur) echo "GlobalHybridMuonSur" ;;
    LayerWiseMuonSpectral|LayerWiseMuonSpectral-Full|LayerwiseHybridMuonSur|LayerWiseHybridMuonSur) echo "LayerwiseHybridMuonSur" ;;
    GlobalHybridMuonMatrixSur) echo "GlobalHybridMuonMatrixSur" ;;
    LayerwiseHybridMuonMatrixSur|LayerWiseHybridMuonMatrixSur) echo "LayerwiseHybridMuonMatrixSur" ;;
    GlobalMuonMatrixSpectral|GlobalMuonMatrixSpectral-Full|GlobalMuonSur|GlobalMuonMatrixSur|GlobalMuonOnlySur) echo "GlobalMuonSur" ;;
    LayerWiseMuonMatrixSpectral|LayerWiseMuonMatrixSpectral-Full|LayerwiseMuonSur|LayerWiseMuonSur|LayerwiseMuonMatrixSur|LayerWiseMuonMatrixSur|LayerwiseMuonOnlySur|LayerWiseMuonOnlySur) echo "LayerwiseMuonSur" ;;
    LayerWiseMuonMatrixSpectralP|LayerWiseMuonMatrixSpectralP-Full|LayerwiseMuonPSur|LayerwiseMuonOnlyPSur) echo "LayerwiseMuonPSur" ;;
    LayerWiseMuonMatrixSpectralSat|LayerWiseMuonMatrixSpectralSat-Full|LayerwiseMuonSatSur|LayerwiseMuonOnlySatSur) echo "LayerwiseMuonSatSur" ;;
    LayerWiseMuonMatrixSpectralSatP|LayerWiseMuonMatrixSpectralSatP-Full|LayerwiseMuonSatPSur|LayerwiseMuonOnlySatPSur) echo "LayerwiseMuonSatPSur" ;;
    *) echo "$1" ;;
  esac
}

case "$MODE" in
  train)
    for opt_type in "${optimizer_types[@]}"; do
      opt_label="$(optimizer_label "$opt_type")"
      run_wandb_group="${WANDB_GROUP:-${SETTING}-${opt_label}-s${SEED}}"
      for method in "${methods[@]}"; do
        method_run_label="$(method_label "$method")"
        case "$method_run_label" in
          LayerwiseOptA)
            if [[ "$opt_label" != "adamw" ]]; then
              echo "Skipping $method_run_label for $opt_label (AdamW-only baseline)."
              continue
            fi
            ;;
          GlobalHybridMuonSur|LayerwiseHybridMuonSur|GlobalHybridMuonMatrixSur|LayerwiseHybridMuonMatrixSur)
            if [[ "$opt_label" != "hybrid" ]]; then
              echo "Skipping $method_run_label for $opt_label (legacy hybrid-score ablation)."
              continue
            fi
            ;;
          GlobalMuonSur|LayerwiseMuonSur|LayerwiseMuonPSur|LayerwiseMuonSatSur|LayerwiseMuonSatPSur)
            if [[ "$opt_label" != "muon" ]]; then
              echo "Skipping $method_run_label for $opt_label (Muon-only baseline)."
              continue
            fi
            ;;
        esac
        run_wandb_tags="${WANDB_TAGS:-sft,drpt_opus,${SETTING},${opt_label},${method_run_label}}"
        if [[ "$DRY_RUN" == "true" ]]; then
          cmd=(bash SFT/train/train.sh \
            -c "$CONFIG_DIR" \
            -m "$method" \
            --seed "$SEED")
          cmd+=(--optimizer_type "$opt_type")
          cmd+=(--report_to "$REPORT_TO")
          cmd+=(--wandb_project "$WANDB_PROJECT")
          cmd+=(--wandb_group "$run_wandb_group")
          cmd+=(--wandb_tags "$run_wandb_tags")
          [[ -n "$WANDB_RUN_NAME" ]] && cmd+=(--wandb_run_name "$WANDB_RUN_NAME")
          cmd+=(--dry-run)
          "${cmd[@]}"
        else
          train_args=(
            SFT/train/train.sh
            -c "$CONFIG_DIR"
            -m "$method"
            --seed "$SEED"
          )
          train_args+=(--optimizer_type "$opt_type")
          train_args+=(--report_to "$REPORT_TO")
          train_args+=(--wandb_project "$WANDB_PROJECT")
          train_args+=(--wandb_group "$run_wandb_group")
          train_args+=(--wandb_tags "$run_wandb_tags")
          [[ -n "$WANDB_RUN_NAME" ]] && train_args+=(--wandb_run_name "$WANDB_RUN_NAME")
          JOB_NAME="sft-${TRAIN}-${TASK}-${method_run_label}-${opt_label}-s${SEED}" \
          ./submit.sh "${train_args[@]}"
        fi
      done
    done
    ;;

  eval)
    for opt_type in "${optimizer_types[@]}"; do
      opt_label="$(optimizer_label "$opt_type")"
      for method in "${methods[@]}"; do
        method_run_label="$(method_label "$method")"
        case "$method_run_label" in
          LayerwiseOptA)
            if [[ "$opt_label" != "adamw" ]]; then
              echo "Skipping $method_run_label for $opt_label (AdamW-only baseline)."
              continue
            fi
            ;;
          GlobalHybridMuonSur|LayerwiseHybridMuonSur|GlobalHybridMuonMatrixSur|LayerwiseHybridMuonMatrixSur)
            if [[ "$opt_label" != "hybrid" ]]; then
              echo "Skipping $method_run_label for $opt_label (legacy hybrid-score ablation)."
              continue
            fi
            ;;
          GlobalMuonSur|LayerwiseMuonSur|LayerwiseMuonPSur|LayerwiseMuonSatSur|LayerwiseMuonSatPSur)
            if [[ "$opt_label" != "muon" ]]; then
              echo "Skipping $method_run_label for $opt_label (Muon-only baseline)."
              continue
            fi
            ;;
        esac
        if [[ "$DRY_RUN" == "true" ]]; then
          bash SFT/eval/eval.sh \
            --train "$TRAIN" \
            --task "$TASK" \
            --method "$method" \
            --optimizer_type "$opt_type" \
            --n_test "$N_TEST" \
            --batch_size "$EVAL_BATCH_SIZE" \
            --seed "$SEED" \
            --dry-run
        else
          JOB_NAME="eval-${TASK}-${method_run_label}-${opt_label}-s${SEED}" \
          ./submit.sh SFT/eval/eval.sh \
            --train "$TRAIN" \
            --task "$TASK" \
            --method "$method" \
            --optimizer_type "$opt_type" \
            --n_test "$N_TEST" \
            --batch_size "$EVAL_BATCH_SIZE" \
            --seed "$SEED"
        fi
      done
    done
    ;;

  *)
    usage
    exit 1
    ;;
esac
