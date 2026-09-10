
# uv run python single_embedding_retrieval.py \
#     --doc_dataset "vm2825/nemotron-cc-v21-Parsed-QA4-filtered-1.7B-evensplit-RQ-8B-val-docs" \
#     --doc_split "train" \
#     --doc_column "pos_doc" \
#     --max_docs 37920 \
#     --query_dataset "vm2825/nemotron-cc-v21-Parsed-QA4-filtered-1.7B-evensplit-RQ-8B" \
#     --query_split "validation" \
#     --query_column "synthetic_question" \
#     --query_gt_column "pos_doc" \
#     --query_answer_column "original_answer" \
#     --model_name "Qwen/Qwen3-Embedding-0.6B" \
#     --hf_ckpt_dir "~/weights/huggingface" \
#     --tp_devices 1 \
#     --doc_prefix "" \
#     --query_task "Given a web search query, retrieve relevant passages that answer the query" \
#     --max_doc_length 1024 \
#     --max_query_length 128 \
#     --encode_batch_size 2048 \
#     --search_batch_size 2048 \
#     --top_k 10 \
#     --num_queries 5000 \
#     --embeddings_dir ./embeddings_cache \
#     --output results.json


# uv run generator.py \
#     --input results_results.json \
#     --output gen_results.json \
#     --num_questions 500 \
#     --start_server \
#     --temperature 0 \
#     --tensor_parallel_size 8 \
#     --max_model_len 16384 \
#     --download_dir /tmp


uv run /home/suhas/memory-layers/llmeval/run_eval.py \
    --input /home/suhas/memory-layers/Baselines/gen_results.json \
    --input_type json \
    --output_file /home/suhas/memory-layers/Baselines/eval_results.json \
    --summary_file /home/suhas/memory-layers/Baselines/eval_summary.json \
    --start_server   
