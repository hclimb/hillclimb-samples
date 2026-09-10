#!/bin/bash
#SBATCH -J submod_test
#SBATCH -p gpu01
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --qos=deadline
#SBATCH -o logs/%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
source "$_repo_root/cluster_env.sh"
REPO_ROOT="$DRPT_REPO_ROOT"
unset _repo_root
cd "$REPO_ROOT"
mkdir -p "$DRPT_LOGS_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  SEEDS="${SEEDS:-42 43 77 101 123 202 3407 2024}"
  PARTITION="${PARTITION:-gpu01}"
  GPUS="${GPUS:-1}"
  NTASKS="${NTASKS:-8}"
  QOS="${QOS:-deadline}"

  echo "[LAUNCH] Submitting submodularity sweep with seeds: $SEEDS"
  for seed in $SEEDS; do
    sbatch \
      --job-name="submod_s${seed}" \
      --partition="$PARTITION" \
      --gres="gpu:${GPUS}" \
      --ntasks="$NTASKS" \
      --qos="$QOS" \
      --output="$DRPT_LOGS_DIR/submod_s${seed}_%j.out" \
      --export="ALL,DRPT_REPO_ROOT=$DRPT_REPO_ROOT,DRPT_DATA_DIR=$DRPT_DATA_DIR,DRPT_RUNS_DIR=$DRPT_RUNS_DIR,DRPT_REPORTS_DIR=$DRPT_REPORTS_DIR,DRPT_LOGS_DIR=$DRPT_LOGS_DIR,DRPT_PYTHON=$DRPT_PYTHON,SEED=${seed}" \
      "$0"
  done
  exit 0
fi

activate_env

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_DATASETS_CACHE"

PYTHON="${PYTHON:-$DRPT_PYTHON}"
FILE="$REPO_ROOT/submodularity.py"

# MODE=toy: synthetic sanity checks, including a counterexample.
# MODE=real: small real SFT dataset gradients from train/validation examples.
# MODE=both: run toy first, then real.
MODE="${MODE:-real}"

# Small real-data default. Override these at sbatch time if needed:
#   MODE=real MODEL=meta-llama/Llama-3.2-1B-Instruct TRAIN_N=16 VAL_N=4 sbatch test.sh
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}" # meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3-0.6B
DATA_DIR="${DATA_DIR:-$DRPT_DATA_DIR}"
TASK="${TASK:-samsum}"
TRAIN_DATASETS="${TRAIN_DATASETS:-alpaca}"
TRAIN_N="${TRAIN_N:-32}"
VAL_N="${VAL_N:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
TRIALS="${TRIALS:-5000}"
K="${K:-8}"
SEED="${SEED:-42}"
REAL_LAYER_NAME="${REAL_LAYER_NAME:-}"
MATRIX_CROP_ROWS="${MATRIX_CROP_ROWS:-128}"
MATRIX_CROP_COLS="${MATRIX_CROP_COLS:-128}"

run_toy() {
  "$PYTHON" "$FILE" \
    --toy-suite \
    --n 12 \
    --rows 8 \
    --cols 8 \
    --trials "$TRIALS" \
    --k "$K" \
    --max-l-size 3 \
    --seed "$SEED" \
    --json-out "$DRPT_LOGS_DIR/submod_toy_s${SEED}.json"
}

run_real() {
  layer_args=()
  if [[ -n "$REAL_LAYER_NAME" ]]; then
    layer_args=(--real-layer-name "$REAL_LAYER_NAME")
  fi

  "$PYTHON" "$FILE" \
    --real-sft \
    --model-name-or-path "$MODEL" \
    --data-dir "$DATA_DIR" \
    --analysis-dataset "$TASK" \
    --train-dataset-names "$TRAIN_DATASETS" \
    --eval-split validation \
    --train-n "$TRAIN_N" \
    --val-n "$VAL_N" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --trials "$TRIALS" \
    --k "$K" \
    --seed "$SEED" \
    --matrix-crop "$MATRIX_CROP_ROWS" "$MATRIX_CROP_COLS" \
    --save-real-gradients "$DRPT_LOGS_DIR/submod_${TASK}_real_grads_s${SEED}.pt" \
    --json-out "$DRPT_LOGS_DIR/submod_${TASK}_real_s${SEED}.json" \
    "${layer_args[@]}"
}

echo "[INFO] MODE=$MODE"
echo "[INFO] MODEL=$MODEL TASK=$TASK TRAIN_DATASETS=$TRAIN_DATASETS TRAIN_N=$TRAIN_N VAL_N=$VAL_N K=$K TRIALS=$TRIALS CROP=${MATRIX_CROP_ROWS}x${MATRIX_CROP_COLS}"

case "$MODE" in
  toy)
    run_toy
    ;;
  real)
    run_real
    ;;
  both)
    run_toy
    run_real
    ;;
  *)
    echo "Unknown MODE=$MODE. Use toy, real, or both." >&2
    exit 1
    ;;
esac
