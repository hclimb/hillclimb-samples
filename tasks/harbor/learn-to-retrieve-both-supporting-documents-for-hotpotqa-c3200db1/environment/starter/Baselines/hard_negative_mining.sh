
ulimit -s unlimited

uv run python hard_negative_mining.py \
    --doc_dataset "vm2825/nemotron-cc-v2-Parsed-DQA1-answered-1.7B-combined-docs" \
    --doc_split "train" \
    --doc_column "pos_doc" \
    --max_docs 5000000 \
    --query_dataset "vm2825/nemotron-cc-v2-Parsed-DQA1-answered-1.7B-combined" \
    --query_split "train" \
    --query_column "question" \
    --pos_doc_column "pos_doc" \
    --answer_column "original_answer" \
    --think_column "think" \
    --synthetic_answer_column "synthetic_answer" \
    --max_queries 15000000 \
    --model_name "Qwen/Qwen3-Embedding-0.6B" \
    --hf_ckpt_dir "~/weights/huggingface" \
    --tp_devices 1 \
    --doc_prefix "" \
    --query_task "Given a web search query, retrieve relevant passages that answer the query" \
    --max_doc_length 1024 \
    --max_query_length 128 \
    --encode_batch_size 2048 \
    --search_batch_size 2048 \
    --top_k 6 \
    --embeddings_dir ./embeddings_cache \
    --output_dir ./hard_negatives \
    --hf_output_dataset "vm2825/nemotron-cc-v2-Parsed-DQA1-answered-1.7B-combined-hard-negatives" \
    --hf_token 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
