export TPU_NAME=scienceqa-cleanup
export ZONE=europe-west4-a
export PROJECT_ID=memorylayers

python /home/mihiragarwal/memory-layers-new/memory-layers/datagen/persistent_tpu/orchestrator.py \
  --project $PROJECT_ID \
  --zone $ZONE \
  --tpu-name $TPU_NAME \
  --tpu-type v6e-8 \
  --env-file /home/mihiragarwal/memory-layers-new/memory-layers/.env \
  --parquets "0-111" \
  --chunk-size 1 \
  --state-file /home/mihiragarwal/memory-layers-new/memory-layers/datagen/persistent_tpu/scienceqa_state_cleanup.json \
  --setup-script /home/mihiragarwal/memory-layers-new/memory-layers/datagen/persistent_tpu/setup_scienceqa.sh \
  --run-command "uv run datagen/generate_scienceqa_sft.py --parquet-numbers '{CHUNKS}' --tp-size 8"
