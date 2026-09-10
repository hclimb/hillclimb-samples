#!/bin/bash

# DRPT_REPO_ROOT is exported by launchers because SLURM may execute a spooled
# script copy. SLURM_SUBMIT_DIR remains a compatibility fallback.
SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_REPO_ROOT}}"
source "$REPO_ROOT/cluster_env.sh" \
    || { echo "ERROR: $REPO_ROOT/cluster_env.sh not found."; exit 1; }
REPO_ROOT="$DRPT_REPO_ROOT"
activate_env || { echo "ERROR: failed to activate $DRPT_CONDA_ENV" >&2; exit 1; }

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

set -e

# Default values
models_dir="$DRPT_RUNS_DIR"
data_dir="$DRPT_DATA_DIR"
python_bin="${DRPT_EVAL_PYTHON:-$DRPT_PYTHON}"
model_path=""
train=""
task=""
subject=""
method=""
optimizer_type=""
n_test=-1
batch_size=1
max_new_tokens=""
seed=42
dry_run=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models_dir)
            models_dir="$2"
            shift 2
            ;;
        --data_dir)
            data_dir="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --train)
            train="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --subject)
            subject="$2"
            shift 2
            ;;
        --method)
            method="$2"
            shift 2
            ;;
        --optimizer_type|--optimizer-type)
            optimizer_type="$2"
            shift 2
            ;;
        --n_test)
            n_test="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --max_new_tokens)
            max_new_tokens="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --models_dir DIR     Models directory (default: \$DRPT_RUNS_DIR)"
            echo "  --data_dir DIR       Data directory (default: \$DRPT_DATA_DIR)"
            echo "  --train NAME         Filter by training dataset (alpaca, less, tulu3, wizardlm)"
            echo "  --task NAME          Override task (includes ifeval, ifbench, math500, mbpp_plus)"
            echo "  --subject NAME       MMLU subject or BBH task to evaluate on (default: all)"
            echo "  --method NAME        Filter by method (e.g., FullTraining-MeSO, LayerWiseSubset-Full)"
            echo "  --optimizer_type TYPE Filter by optimizer suffix (adamw, muon, or hybrid)"
            echo "  --n_test N           Number of test examples (-1 for all)"
            echo "  --batch_size N       Batch size for generation (default: 1)"
            echo "  --max_new_tokens N   Max tokens to generate (default: task policy)"
            echo "  --seed N             Random seed for reproducibility (default: 42)"
            echo "  --dry-run            Print command without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================================"
echo "  SFT Evaluation"
echo "========================================================"
echo "Models dir:      $models_dir"
echo "Data dir:        $data_dir"
echo ""
echo "Filters:"
echo "  Train:         ${train:-all}"
echo "  Task:          ${task:-auto-detect}"
echo "  Subject:       ${subject:-all}"
echo "  Method:        ${method:-all}"
echo "  Optimizer:     ${optimizer_type:-all}"
echo ""
echo "Generation:"
echo "  Batch size:    $batch_size"
echo "  Max new tokens: ${max_new_tokens:-task policy}"
echo "  N test:        $n_test (-1 = all)"
echo "  Seed:          $seed"
echo "========================================================"

# Build command
cmd="$python_bin -m SFT.eval.eval"
if [[ -n "$model_path" ]]; then
    cmd="$cmd --model_path $model_path"
else
    cmd="$cmd --models_dir $models_dir"
    # Fully specified launcher jobs must resolve to one trained run. Keep the
    # historical generic batch-eval behavior when any discovery key is absent.
    if [[ -n "$train" && -n "$task" && -n "$method" && -n "$optimizer_type" ]]; then
        cmd="$cmd --require_single_match"
    fi
fi
cmd="$cmd --data_dir $data_dir"
cmd="$cmd --n_test $n_test"
cmd="$cmd --batch_size $batch_size"
if [[ -n "$max_new_tokens" ]]; then
    cmd="$cmd --max_new_tokens $max_new_tokens"
fi
cmd="$cmd --seed $seed"

if [[ -n "$train" ]]; then
    cmd="$cmd --train $train"
fi

if [[ -n "$task" ]]; then
    cmd="$cmd --task $task"
fi

if [[ -n "$subject" ]]; then
    cmd="$cmd --subject $subject"
fi

if [[ -n "$method" ]]; then
    cmd="$cmd --method $method"
fi

if [[ -n "$optimizer_type" ]]; then
    cmd="$cmd --optimizer_type $optimizer_type"
fi

if [[ "$dry_run" == true ]]; then
    echo "Dry run: $cmd"
else
    eval "$cmd"
fi
