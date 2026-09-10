#!/bin/bash
# Target-only sanity check: SFT on D* (64 target-gradient rows), no curation.
set -euo pipefail
REPO_ROOT="${DRPT_REPO_ROOT:?}"
cd "$REPO_ROOT"
source cluster_env.sh
IFS=':' read -r -a settings <<< "${DRPT_TO_SETTINGS:-reason_math:reason_code:inst_if}"
setting="${settings[${SLURM_ARRAY_TASK_ID:?}]}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
OUT="$REPO_ROOT/SFT/runs/target_only/$setting"
echo "[target-only] setting=$setting out=$OUT epochs=${DRPT_TO_EPOCHS:-15} micro=${DRPT_TO_MICRO:-2}"
"$DRPT_PYTHON" -m SFT.train.train_target_only \
    --setting "$setting" --artifact_build_id "${DRPT_ARTIFACT_BUILD_ID:?}" \
    --data_dir "$REPO_ROOT/SFT/data" --epochs "${DRPT_TO_EPOCHS:-15}" \
    --micro_batch_size "${DRPT_TO_MICRO:-2}" --output_dir "$OUT"
