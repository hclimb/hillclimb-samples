#!/bin/bash
set -a; . $HOME/.env; set +a
export PATH=$HOME/.local/bin:$PATH
# GCS identity from .env (GCS_USER_EMAIL); `:?` fails loudly if it's unset.
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?GCS_USER_EMAIL not set in ~/.env}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT
export SIM_EVAL_SAMPLES=64 SIM_EVAL_MILESTONE=8000 SIM_EVAL_SCAN_S=900
cd $HOME/memory-layers
. .venv/bin/activate
exec env PATH="$PATH" PYTHONPATH=. .venv/bin/python scripts/misc/sim_eval_box.py \
  simpair_ctrl_v3 simpair_sim40_v3 simpair_sim40nce_v3 simpair_sim15nce_v3
