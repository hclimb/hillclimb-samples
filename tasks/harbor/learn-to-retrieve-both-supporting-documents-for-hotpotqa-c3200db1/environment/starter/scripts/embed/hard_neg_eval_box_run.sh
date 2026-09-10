#!/bin/bash
# Runs ON a rohun* eval box (NOT the training box). Watches ONE training run-dir's GCS checkpoints
# and evaluates each milestone, logging eval/* into that run's TRAINING wandb run.
#
# RUN_DIR is the run-dir BASENAME (<run_name>-<YYYY-MM-DD>-<HH-MM-SS>), NOT a run_name. train.py
# prints it at startup beside the wandb id. This is required, not cosmetic: a run_name is not
# unique (this one already had a complete April run at steps 40k-100k), and a name-scoped box
# would evaluate that older model and log it into this run's curve.
#
# The training run must have been launched with trainer.wandb_run_id=auto — the box derives the
# same wandb id from the SAME run-dir (utils.wandb_run_id_from_run_dir) and attaches as a
# shared-mode secondary writer. See wiki/evaluation/eval-boxes.md.
#
# Shard across boxes by exporting HARD_NEG_EVAL_DATASETS (e.g. box A=science,msmarco B=hotpotqa,musique);
# unset => all 4 on one box. GCS-result idempotency keeps boxes off each other's work.
set -a; . $HOME/.env; set +a
export PATH=$HOME/.local/bin:$PATH
# GCS identity comes from .env (GCS_USER_EMAIL) — never hardcode it here, or this box silently
# disagrees with what setup_gcs_credentials() (utils.py) picks for train.py/eval.py. `:?` fails
# loudly rather than building a legacy_credentials//adc.json path that 404s much later.
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?GCS_USER_EMAIL not set in ~/.env}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT

export HARD_NEG_EVAL_SAMPLES=${HARD_NEG_EVAL_SAMPLES:-128}
# Match trainer.checkpoint_interval: a milestone with no checkpoint is skipped, and orbax rotates
# old checkpoints away, so a milestone finer than the checkpoint cadence just spins.
export HARD_NEG_EVAL_MILESTONE=${HARD_NEG_EVAL_MILESTONE:-10000}
export HARD_NEG_EVAL_SCAN_S=${HARD_NEG_EVAL_SCAN_S:-900}
export HARD_NEG_EVAL_DATASETS=${HARD_NEG_EVAL_DATASETS:-science,msmarco,hotpotqa,musique}
export HARD_NEG_EVAL_WANDB=${HARD_NEG_EVAL_WANDB:-1}

# Target ONE run-dir. Two ways to say which:
#   RUN_DIR=<run_name>-<YYYY-MM-DD>-<HH-MM-SS>   explicit (copy it from train.py's startup line)
#   RUN_NAME=... RUN_START_TIME=...              composed — how multi-tpu-box-run.sh does it, since
#                                                it mints one RUN_START_TIME for every box and the
#                                                training script owns the run_name
# No default on purpose: a wrong/stale run-dir silently scores the wrong model into the wrong
# wandb run — the failure this whole design exists to prevent.
RUN_NAME=${RUN_NAME:-qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16}
if [ -z "${RUN_DIR:-}" ] && [ -n "${RUN_START_TIME:-}" ]; then
  RUN_DIR="${RUN_NAME}-${RUN_START_TIME}"
  echo "[eval-box] composed RUN_DIR=$RUN_DIR from RUN_NAME + RUN_START_TIME"
fi
if [ -z "${RUN_DIR:-}" ]; then
  echo "ERROR: need RUN_DIR, or RUN_START_TIME (+ RUN_NAME) to compose it." >&2
  echo "  RUN_DIR=${RUN_NAME}-2026-07-16-15-14-02" >&2
  echo "  train.py prints the run-dir at startup next to the wandb id;" >&2
  echo "  multi-tpu-box-run.sh sets RUN_START_TIME for every box automatically." >&2
  return 1 2>/dev/null || exit 1
fi

cd $HOME/memory-layers
. .venv/bin/activate
exec env PATH="$PATH" PYTHONPATH=. .venv/bin/python scripts/misc/hard_neg_eval_box.py "$RUN_DIR"
