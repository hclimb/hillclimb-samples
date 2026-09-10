export TPU_NAME=scienceqa-datagen
export ZONE=us-east5-c
export PROJECT_ID=memory-layers-484918

python orchestrator.py \
  --project $PROJECT_ID \
  --zone $ZONE \
  --tpu-name $TPU_NAME \
  --tpu-type v6e-8 \
  --env-file /home/mihiragarwal/memory-layers-new/memory-layers/.env \
  --parquets "0-49" \
  --chunk-size 1 \
  --state-file scienceqa_state.json \
  --setup-script setup_scienceqa.sh \
  --run-command "uv run datagen/generate_scienceqa_sft.py --parquet-numbers '{CHUNKS}' --tp-size 8"
