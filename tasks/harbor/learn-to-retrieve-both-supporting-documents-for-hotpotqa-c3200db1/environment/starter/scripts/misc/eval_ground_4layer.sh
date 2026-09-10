#!/bin/bash
# One-off: re-evaluate the 4-layer grounding checkpoint through the FIXED eval path
# (apply_checkpoint_model_cfg -> checkpoint's mem_layers=[9,14,20,27] is authoritative, not the
# eval default [14]). Runs the same 3 corpus tasks the current hard-neg run used: msmarco / hotpotqa
# / musique @ c512, n=128, reporting llm_judge_accuracy + lexical_grounding + mem_pos_weight_mass.
#
# Also runs the unit test first so we prove the resolver before spending ~1.5h of TPU on evals, and
# greps each run for the "[eval] checkpoint-authoritative: memory.mem_layers=..." line — that line
# appearing is the on-box proof that (a) the old path WOULD have clobbered to [14], (b) the fix now
# builds all 4 layers.
#
#   CKPT=gs://memory-layers-training/ground_s1_zeroinit_4layer-2026-07-04-09-42-55/qwen3_mem_embed/38000 \
#     bash scripts/misc/eval_ground_4layer.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

CKPT="${CKPT:?set CKPT=gs://.../qwen3_mem_embed/<step>}"
SAMPLES="${SAMPLES:-128}"
DST_PREFIX="${DST_PREFIX:-gs://memory-layers-training/ground_4layer_refix/$(basename "$(dirname "$(dirname "$CKPT")")")/step$(basename "$CKPT")}"

echo "################ UNIT TEST: apply_checkpoint_model_cfg ################"
uv run python tests/test_checkpoint_model_authoritative.py || { echo "UNIT TEST FAILED — aborting"; exit 1; }

# alias=task_config for the three corpus evals (same configs the hard-neg run used).
declare -A TASKS=(
  [msmarco_c512]=gen_large_mem_msmarco_c512
  [hotpotqa_c512]=gen_large_mem_hotpotqa_c512
  [musique_c512]=gen_large_mem_musique_c512
)

for alias in msmarco_c512 hotpotqa_c512 musique_c512; do
  task="${TASKS[$alias]}"
  out="$HOME/evalbox/ground4l_${alias}"
  rm -rf "$out"
  echo
  echo "################ EVAL $alias ($task) on $CKPT ################"
  # Free any lingering vLLM judge holding the TPU from a previous task.
  uv run python -c "from evals.vllm import VLLMInference; VLLMInference.free_tpu_devices()" 2>/dev/null || true
  sleep 5
  uv run python eval.py \
      "checkpoint_dir=$CKPT" \
      "~eval_set@evals=pretraining" \
      "+eval/tasks@evals.${alias}=${task}" \
      "evals.${alias}.eval.num_samples=${SAMPLES}" \
      "+aux_losses.mem_pos_weight_mass.enabled=true" "+aux_losses.mem_pos_weight_mass.weight=0.0" \
      "+aux_losses.doc_access_acc.enabled=true" "+aux_losses.doc_access_acc.weight=0.0" \
      tp_devices=1 use_wandb=false "hydra.run.dir=$out" 2>&1 \
    | grep -aE "checkpoint-authoritative|explicit model overrides|llm_judge_accuracy|lexical_grounding|mem_pos_weight_mass|doc_hit_rate|Traceback|Error|not finite" \
    | tail -25

  res=$(ls "$out"/eval_results/step_*/"$alias"/outputs/*.json 2>/dev/null | head -1)
  if [ -n "$res" ]; then
    uv run python -c "import json,sys; d=json.load(open('$res')); m=d.get('metrics',d); print('  FINAL $alias:', {k:round(v,4) for k,v in m.items() if isinstance(v,(int,float)) and any(s in k for s in ['judge','ground','pos_weight','hit_rate'])})"
    uv run python -c "
import os; from utils import setup_gcs_credentials; setup_gcs_credentials()
import gcsfs; gcsfs.GCSFileSystem().put('$res', '${DST_PREFIX#gs://}/$alias.json')" 2>/dev/null \
      && echo "  uploaded -> $DST_PREFIX/$alias.json"
  else
    echo "  !! no result JSON for $alias"
  fi
done
echo
echo "################ DONE ################"
