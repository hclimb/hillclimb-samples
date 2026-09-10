cd ~/memory-layers
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" JAX_PLATFORMS=cpu \
uv run python scripts/debug/smoke_test_cot_ablation.py \
    ${SMOKE_TEST_DATASET:+--dataset "$SMOKE_TEST_DATASET"}
