export TPU_NAME=dqa1-tpu-4
export ZONE=us-east1-d
export PROJECT_ID=memorylayers

python /home/mihiragarwal/memory-layers-new/memory-layers/datagen/persistent_tpu/orchestrator.py \
  --project $PROJECT_ID \
  --zone $ZONE \
  --tpu-name $TPU_NAME \
  --tpu-type v6e-8 \
  --env-file /home/mihiragarwal/memory-layers-new/memory-layers/.env \
  --parquets "135-179" \
  --chunk-size 1 \
  --state-file /home/mihiragarwal/memory-layers-new/memory-layers/datagen/persistent_tpu/dqa1_state_tpu4.json \
  --setup-script /home/mihiragarwal/memory-layers-new/memory-layers/datagen/persistent_tpu/setup_scienceqa.sh \
  --run-command "uv run datagen/generate_dqa1_sft.py --parquet-numbers '{CHUNKS}' --tp-size 8"
