uv run generator.py \
--input results_results.json \
--output gen_results.json \
--num_questions 500 \
--start_server \
--temperature 0 \
--tensor_parallel_size 8 \
--max_model_len 16384 \
--download_dir /tmp