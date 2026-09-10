#!/bin/bash
# End-to-end verification of the orbax native sharded-save fix (utils.py::save_checkpoint,
# 2026-07-24). Launches the dummy_token spec64 config with checkpoint_interval=100 and a
# short step count so the first save fires quickly. Passes if it saves cleanly at step 100
# (previously OOM'd at first save on Rohun's tp=2 smoke; would have OOM'd on our tp=1 at
# step 10000). Evals off — save is the thing under test.
GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed_spec64 \
    trainer=staged \
    dataset=qa_hard_neg_think_sft4b \
    dataset.num_workers=0 \
    trainer.tp_devices=1 \
    trainer.steps=300 \
    trainer.checkpoint_interval=100 \
    trainer.max_to_keep=4 \
    trainer.log_interval=10 \
    trainer.eval_interval=0 \
    'eval_set@trainer.evals=none' \
    ~trainer.training_stages \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="verify_sharded_save"
