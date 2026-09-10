uv run eval.py \
    checkpoint_dir=/home/suhas/memory-layers/outputs/bior_qwen3_mem_16k_top8_product_keys_bio_interval1000_numqa6 \
    eval=generation \
    dataset=bior \
    dataset.limit=100 \
    dataset.bio_interval=1000 \
    dataset.num_qa_per_bio=6 \
    dataset.provide_docs=false \
    dataset.chat_template=true \
    dataset.shuffle=false \
    eval.batch_size=16 \
    eval.output_file=outputs/eval_bior_qwen3_mem_16k_top8_product_keys_bio_interval1000_numqa6.json