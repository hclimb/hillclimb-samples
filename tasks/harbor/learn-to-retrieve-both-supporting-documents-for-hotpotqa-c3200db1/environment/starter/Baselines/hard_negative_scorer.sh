
ulimit -s unlimited

uv run python hard_negative_scorer.py \
    --input_dataset  "vm2825/nemotron-cc-v21-Parsed-QA4-hard-negatives-2M" \
    --input_split    "train" \
    --model_name     "Qwen/Qwen3-Reranker-0.6B" \
    --hf_ckpt_dir    "~/weights/huggingface" \
    --tp_devices     1 \
    --max_length     1536 \
    --score_batch_size 512 \
    --task           "Given a web search query, retrieve relevant passages that answer the query" \
    --output_dir     ./scored_negatives \
    --hf_output_dataset "vm2825/nemotron-cc-v21-Parsed-QA4-hard-negatives-scored-2M" \
    --hf_token       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 \
    --n_queries 2000
