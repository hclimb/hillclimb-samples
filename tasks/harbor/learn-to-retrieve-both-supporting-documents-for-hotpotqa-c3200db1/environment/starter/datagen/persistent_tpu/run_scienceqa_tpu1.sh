export TPU_NAME=scienceqa-tpu-1
export ZONE=us-east1-d
export PROJECT_ID=memorylayers

python /home/mihiragarwal/memory-layers-new/memory-layers/datagen/persistent_tpu/orchestrator.py \
  --project $PROJECT_ID \
  --zone $ZONE \
  --tpu-name $TPU_NAME \
  --tpu-type v6e-8 \
  --env-file /home/mihiragarwal/memory-layers-new/memory-layers/.env \
  --parquets "0-27" \
  --chunk-size 1 \
  --state-file scienceqa_state_tpu1.json \
  --setup-script setup_scienceqa.sh \
  --run-command "uv run datagen/generate_scienceqa_sft.py --parquet-numbers '{CHUNKS}' --tp-size 8"
