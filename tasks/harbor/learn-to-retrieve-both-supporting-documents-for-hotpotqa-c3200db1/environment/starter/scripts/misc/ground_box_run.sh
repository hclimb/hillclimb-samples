#!/bin/bash
# Runs ON a rohun* eval box. Shard the 3 datasets across 3 boxes by exporting
# GROUND_EVAL_DATASETS before invoking (box A=msmarco, B=hotpotqa, C=musique);
# unset => all 3 on one box. All boxes watch all 4 runs; GCS-result idempotency keeps
# them from redoing each other's work.
set -a; . $HOME/.env; set +a
export PATH=$HOME/.local/bin:$PATH
# GCS identity from .env (GCS_USER_EMAIL); `:?` fails loudly if it's unset.
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?GCS_USER_EMAIL not set in ~/.env}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT
# Dense curve: evaluate every 2000-step checkpoint. Lower SAMPLES keeps each box inside the
# orbax rotation window now that there are 6 tasks/run. GROUND_EVAL_WANDB=1 => also log to wandb.
export SIM_EVAL_SAMPLES=${SIM_EVAL_SAMPLES:-64}
export SIM_EVAL_MILESTONE=${SIM_EVAL_MILESTONE:-2000}
export SIM_EVAL_SCAN_S=${SIM_EVAL_SCAN_S:-900}
export GROUND_EVAL_DATASETS=${GROUND_EVAL_DATASETS:-msmarco,hotpotqa,musique}
export GROUND_EVAL_WANDB=${GROUND_EVAL_WANDB:-1}
cd $HOME/memory-layers
. .venv/bin/activate
exec env PATH="$PATH" PYTHONPATH=. .venv/bin/python scripts/misc/ground_eval_box.py \
  ground_control ground_s1_zeroinit_4layer ground_s2_kv_split ground_s3_span ground_s2_copyinit \
  ground_s2_copyinit_topk4 4B_msmarco_triplets_topk32_per_query_isolation
