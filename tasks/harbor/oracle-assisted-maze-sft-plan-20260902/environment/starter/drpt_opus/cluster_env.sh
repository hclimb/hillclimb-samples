#!/bin/bash

# Central path configuration for the current checkout. Every value can be
# overridden by exporting it before this file is sourced.
_drpt_cluster_env_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRPT_REPO_ROOT="${DRPT_REPO_ROOT:-$_drpt_cluster_env_dir}"
export DRPT_DATA_DIR="${DRPT_DATA_DIR:-$DRPT_REPO_ROOT/SFT/data}"
export DRPT_RUNS_DIR="${DRPT_RUNS_DIR:-$DRPT_REPO_ROOT/SFT/runs}"
export DRPT_REPORTS_DIR="${DRPT_REPORTS_DIR:-$DRPT_REPO_ROOT/SFT/eval/reports}"
export DRPT_LOGS_DIR="${DRPT_LOGS_DIR:-$DRPT_REPO_ROOT/logs}"
export DRPT_SLURM_PARTITION="${DRPT_SLURM_PARTITION:-standard}"
export DRPT_SLURM_QOS="${DRPT_SLURM_QOS:-normal}"
# Official IFBench evaluator checkout, required by inst_if/mixed_if downstream
# evaluation. The evaluator verifies this checkout's commit before scoring, so
# the pin lives in SFT/data/dolci32k/profile.py, not here.
export DRPT_IFBENCH_REPO="${DRPT_IFBENCH_REPO:-$(cd -- "$DRPT_REPO_ROOT/../.." && pwd)/IFBench}"
unset _drpt_cluster_env_dir

export DRPT_CUDA_HOME="${DRPT_CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_HOME="$DRPT_CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
case ":${PATH:-}:" in
  *":$CUDA_HOME/bin:"*) ;;
  *) export PATH="$CUDA_HOME/bin:${PATH:-}" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
  *":$CUDA_HOME/lib64:"*) ;;
  *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

export DRPT_CONDA_ROOT="${DRPT_CONDA_ROOT:-${HOME}/miniconda3}"
export DRPT_CONDA_ENV="${DRPT_CONDA_ENV:-drpt-next}"
export DRPT_PYTHON="${DRPT_PYTHON:-$DRPT_CONDA_ROOT/envs/$DRPT_CONDA_ENV/bin/python}"

# Legacy aliases remain available for older scripts. New SFT code should use
# the explicit DRPT_* directories above instead of appending repository names.
export CODE_DIR="${CODE_DIR:-$DRPT_REPO_ROOT}"
export SCRATCH_DIR="${SCRATCH_DIR:-$DRPT_REPO_ROOT}"

activate_env() {
  local conda_sh="$DRPT_CONDA_ROOT/etc/profile.d/conda.sh"
  if [[ -r "$conda_sh" ]]; then
    # shellcheck disable=SC1090
    source "$conda_sh"
    conda activate "$DRPT_CONDA_ENV"
  elif [[ -x "$DRPT_PYTHON" ]]; then
    export PATH="$(dirname -- "$DRPT_PYTHON"):${PATH:-}"
  else
    echo "ERROR: cannot activate '$DRPT_CONDA_ENV'; missing $conda_sh and $DRPT_PYTHON" >&2
    return 1
  fi

  if [[ ! -x "$DRPT_PYTHON" ]]; then
    DRPT_PYTHON="$(command -v python)" || return 1
    export DRPT_PYTHON
  fi
}
