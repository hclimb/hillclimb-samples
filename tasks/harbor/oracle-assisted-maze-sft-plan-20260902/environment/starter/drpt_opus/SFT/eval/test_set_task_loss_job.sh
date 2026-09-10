#!/bin/bash
# Teacher-forced loss on MATH500 / MBPP+ reference solutions for the AdamW campaign.
set -euo pipefail
REPO_ROOT="${DRPT_REPO_ROOT:?}"
cd "$REPO_ROOT"
source cluster_env.sh
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0
EXTRA=()
for s in reason_math reason_code mixed_math; do
  d="$REPO_ROOT/SFT/runs/target_only/$s"
  [[ -f "$d/_SUCCESS" ]] && EXTRA+=(--extra_model "$s:TargetOnly=$d")
done
"$DRPT_PYTHON" -m SFT.eval.test_set_task_loss \
    --out "$REPO_ROOT/SFT/eval/reports/temp_inst_if_adamw" "${EXTRA[@]}"
