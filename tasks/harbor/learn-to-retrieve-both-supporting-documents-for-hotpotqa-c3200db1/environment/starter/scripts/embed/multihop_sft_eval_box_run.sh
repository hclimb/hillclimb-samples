#!/bin/bash
# Runs ON the SECOND box, in parallel with scripts/embed/train_multihop_sft_midtrain.sh.
# Watches that run's GCS checkpoints and evaluates each milestone, logging eval/* INTO the
# training run's wandb curve.
#
# Launch both together — multi-tpu-box-run.sh mints ONE RUN_START_TIME and forwards it to every
# box, which is what lets the eval box compose the same run-dir without waiting for train.py to
# print it:
#     bash scripts/infrastructure/multi-tpu-box-run.sh \
#         <train-box>=scripts/embed/train_multihop_sft_midtrain.sh \
#         <eval-box>=scripts/embed/multihop_sft_eval_box_run.sh
set -uo pipefail
set -a; . $HOME/.env; set +a
export PATH=$HOME/.local/bin:$PATH
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?GCS_USER_EMAIL not set in ~/.env}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT

# MUST match the training script's CKPT_BUCKET. The training run writes to the us-central1 bucket
# (in-region with the box); .env's GCS_BUCKET is the EUROPE-WEST4 one, so leaving it unset would
# point this box at a run-dir that does not exist and it would poll forever finding nothing.
export GCS_BUCKET="${CKPT_BUCKET:-memory-layers-training-usc1}"

export HARD_NEG_EVAL_SAMPLES=${HARD_NEG_EVAL_SAMPLES:-128}
# Match the training script's trainer.checkpoint_interval=1000. A milestone finer than the
# checkpoint cadence just spins on steps that will never have a checkpoint; coarser leaves gaps.
export HARD_NEG_EVAL_MILESTONE=${HARD_NEG_EVAL_MILESTONE:-1000}
export HARD_NEG_EVAL_SCAN_S=${HARD_NEG_EVAL_SCAN_S:-300}
# MuSiQue c512 ONLY. Note this is an OUT-OF-DOMAIN eval for this run — the model is midtraining on
# ragrawal36/multihop_qa_sft, not on MuSiQue — so unlike the MuSiQue midtraining run these numbers
# measure transfer, and are directly comparable to that run's in-domain curve and to the
# base checkpoint's 0.141 (wiki/experiments/2026-07-18-musique-midtraining-vs-rag.md).
export HARD_NEG_EVAL_DATASETS=${HARD_NEG_EVAL_DATASETS:-musique}
export HARD_NEG_EVAL_WANDB=${HARD_NEG_EVAL_WANDB:-1}

# Judge sizing for a 4-chip box. The metric defaults (evals/metrics/llm_judge.py) are Qwen3-8B at
# tensor_parallel_size=8, a v6e-8 assumption; TP must divide the local chip count, so on a v5p-4
# the judge's vLLM never starts and every task dies AFTER paying for generation. {alias} is
# substituted per task by the driver.
TP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]*$'); [ "$TP" -gt 0 ] 2>/dev/null || TP=4
# NOT written as ${VAR:-default}: bash ends that expansion at the first unescaped '}', which is
# the one in {alias} — the placeholder silently truncates to "{alias" and Hydra rejects the
# override AFTER the eval has loaded the checkpoint. Plain if/then keeps the braces intact.
if [ -z "${HARD_NEG_EVAL_EXTRA_OVERRIDES:-}" ]; then
  HARD_NEG_EVAL_EXTRA_OVERRIDES="+evals.{alias}.eval.metrics.llm_judge_accuracy.model_id=Qwen/Qwen3-4B +evals.{alias}.eval.metrics.llm_judge_accuracy.tensor_parallel_size=$TP"
fi
export HARD_NEG_EVAL_EXTRA_OVERRIDES

# Compose the run-dir from the identity multi-tpu-box-run.sh forwarded. No default run-dir on
# purpose: a stale one silently scores the wrong model into the wrong wandb run.
RUN_NAME=${RUN_NAME:-multihop_sft_midtrain_topk64_seq512_chunks16_bs32}
if [ -z "${RUN_DIR:-}" ] && [ -n "${RUN_START_TIME:-}" ]; then
  RUN_DIR="${RUN_NAME}-${RUN_START_TIME}"
  echo "[eval-box] composed RUN_DIR=$RUN_DIR from RUN_NAME + RUN_START_TIME"
fi
if [ -z "${RUN_DIR:-}" ]; then
  echo "ERROR: need RUN_DIR, or RUN_START_TIME (+ RUN_NAME) to compose it." >&2
  return 1 2>/dev/null || exit 1
fi
echo "[eval-box] bucket=$GCS_BUCKET  milestone=$HARD_NEG_EVAL_MILESTONE  tasks=$HARD_NEG_EVAL_DATASETS  judge_tp=$TP"

cd $HOME/memory-layers
. .venv/bin/activate
exec env PATH="$PATH" PYTHONPATH=. .venv/bin/python scripts/misc/hard_neg_eval_box.py "$RUN_DIR"
