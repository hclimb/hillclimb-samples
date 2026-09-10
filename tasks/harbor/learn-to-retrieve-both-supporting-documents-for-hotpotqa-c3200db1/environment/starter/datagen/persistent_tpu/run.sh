export TPU_NAME=machine-43
export ZONE=europe-west4-a
export PROJECT_ID=memory-layers-484918

python orchestrator.py \
 --project $PROJECT_ID \
 --zone $ZONE \
 --tpu-name $TPU_NAME \
 --tpu-type v6e-8 \
 --env-file /Users/suhas/Desktop/Repos/gcloud/memory-layers/.env \
 --parquets "24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46" \
 --chunk-size 23 \
 --state-file orchestrator_43_state.json \
 --setup-script setup_memory_layers.sh \
 --run-command "uv run datagen/persistent_tpu/generate_cot.py --parquet-numbers '{CHUNKS}'"
