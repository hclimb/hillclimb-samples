uv run train.py \
    model=qwen3_mem_embed \
    dataset=pretraining \
    eval_set@trainer.evals=pretraining \
    +trainer.run_name="4B_pretraining"