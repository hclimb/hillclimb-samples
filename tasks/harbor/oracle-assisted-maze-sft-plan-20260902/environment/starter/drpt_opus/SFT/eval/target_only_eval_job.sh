#!/bin/bash
# Downstream generation eval for the target-only (D*-64) checkpoints.
set -euo pipefail
REPO_ROOT="${DRPT_REPO_ROOT:?}"
cd "$REPO_ROOT"
source cluster_env.sh
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export DRPT_IFBENCH_REPO="${DRPT_IFBENCH_REPO:-/home/nakyungl/drpt-next/IFBench}"
# array index -> "setting:task:max_new_tokens"
cells=(
  "reason_math:math500:4096"
  "reason_code:mbpp_plus:2048"
  "inst_if:ifeval:2048"
  "inst_if:ifbench:2048"
)
IFS=':' read -r setting task mnt <<< "${cells[${SLURM_ARRAY_TASK_ID:?}]}"
RUN="$REPO_ROOT/SFT/runs/target_only/$setting"
[[ -f "$RUN/_SUCCESS" ]] || { echo "ERROR: no trained model at $RUN" >&2; exit 2; }

# Target-only checkpoints were trained from an immutable Dolci32k artifact
# build. Reuse that exact build for downstream benchmarks instead of falling
# back to the legacy, unpinned SFT/data/eval/<task> path.
metadata_path="$RUN/target_only_metadata.json"
[[ -f "$metadata_path" ]] || {
    echo "ERROR: target-only metadata not found: $metadata_path" >&2
    exit 2
}
artifact_build_id="$(
    "$DRPT_PYTHON" - "$metadata_path" "$setting" <<'PY'
import json
import sys

metadata_path, expected_setting = sys.argv[1:]
with open(metadata_path, encoding="utf-8") as handle:
    metadata = json.load(handle)

actual_setting = metadata.get("setting")
if actual_setting != expected_setting:
    raise SystemExit(
        f"target-only metadata setting mismatch: "
        f"expected {expected_setting!r}, got {actual_setting!r}"
    )

build_id = metadata.get("artifact_build_id")
if (
    not isinstance(build_id, str)
    or len(build_id) != 64
    or any(char not in "0123456789abcdef" for char in build_id)
):
    raise SystemExit(f"invalid target-only artifact_build_id: {build_id!r}")
print(build_id)
PY
)"
export DRPT_DOWNSTREAM_PROFILE=dolci32k
export DRPT_ARTIFACT_BUILD_ID="$artifact_build_id"

echo "[target-only-eval] setting=$setting task=$task model=$RUN artifact_build_id=$artifact_build_id"
eval_args=(
    --model_path "$RUN"
    --task "$task"
    --n_test -1
    --batch_size 1
    --max_new_tokens "$mnt"
    --seed 42
)
[[ "${DRPT_JOB_DRY_RUN:-false}" == "true" ]] && eval_args+=(--dry-run)
bash "$REPO_ROOT/SFT/eval/eval.sh" "${eval_args[@]}"
