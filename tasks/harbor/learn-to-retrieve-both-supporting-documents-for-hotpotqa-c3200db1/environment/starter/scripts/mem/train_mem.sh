uv run train.py \
    model=qwen3_mem \
    model.main_model.model_id=Qwen/Qwen3-1.7B \
    model.memory.mem_use_product_keys=true \
    dataset=bior \
    dataset.provide_docs=false \
    dataset@eval_dataset=bior \
    eval_dataset.provide_docs=false
    
    