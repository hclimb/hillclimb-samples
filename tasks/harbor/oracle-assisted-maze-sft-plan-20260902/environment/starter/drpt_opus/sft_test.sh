#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-eval}"
SETTING="${SETTING:-less_squad}"
SEED="${SEED:-42}"
N_TEST="${N_TEST:-500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
DRY_RUN="${DRY_RUN:-false}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-}"

usage() {
  cat <<'EOF'
Usage:
  ./sft_test.sh [setting] [train|eval] [--dry-run]
  SETTING=<setting> MODE=<train|eval> ./sft_test.sh

Settings:
  alpaca_samsum
  less_tydiqa
  triviaqa_nq
  less_squad

Environment:
  SEED=42
  N_TEST=500
  EVAL_BATCH_SIZE=8
  OPTIMIZER_TYPE=hybrid
  METHODS=FullTraining-LoRA,LayerWiseSubset-LoRA,...
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

# Edit this list to choose which methods to run/evaluate.
if [[ -n "${METHODS:-}" ]]; then
  IFS=',' read -ra methods <<< "$METHODS"
else
  methods=(
    FullTraining-LoRA
    LayerWiseSubset-LoRA
    GlobalSubset-LoRA
    FullTraining-MeSO
    LayerWiseSubset-MeSO
    GlobalSubset-MeSO
  )
fi

case "$MODE" in
  train)
    for method in "${methods[@]}"; do
      if [[ "$DRY_RUN" == "true" ]]; then
        cmd=(bash SFT/train/train.sh \
          -c "$CONFIG_DIR" \
          -m "$method" \
          --seed "$SEED")
        [[ -n "$OPTIMIZER_TYPE" ]] && cmd+=(--optimizer_type "$OPTIMIZER_TYPE")
        cmd+=(--dry-run)
        "${cmd[@]}"
      else
        train_args=(
          SFT/train/train.sh
          -c "$CONFIG_DIR"
          -m "$method"
          --seed "$SEED"
        )
        [[ -n "$OPTIMIZER_TYPE" ]] && train_args+=(--optimizer_type "$OPTIMIZER_TYPE")
        JOB_NAME="sft-${TRAIN}-${TASK}-${method}-s${SEED}" \
        ./submit.sh "${train_args[@]}"
      fi
    done
    ;;

  eval)
    for method in "${methods[@]}"; do
      if [[ "$DRY_RUN" == "true" ]]; then
        bash SFT/eval/eval.sh \
          --train "$TRAIN" \
          --task "$TASK" \
          --method "$method" \
          --n_test "$N_TEST" \
          --batch_size "$EVAL_BATCH_SIZE" \
          --seed "$SEED" \
          --dry-run
      else
        JOB_NAME="eval-${TASK}-${method}-s${SEED}" \
        ./submit.sh SFT/eval/eval.sh \
          --train "$TRAIN" \
          --task "$TASK" \
          --method "$method" \
          --n_test "$N_TEST" \
          --batch_size "$EVAL_BATCH_SIZE" \
          --seed "$SEED"
      fi
    done
    ;;

  *)
    usage
    exit 1
    ;;
esac
