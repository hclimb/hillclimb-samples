cd ~/memory-layers
HF_HUB_OFFLINE=0 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
uv run python datagen/join_multihop_cot.py ${JOIN_ARGS:-}
