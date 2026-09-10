# CPU-only correctness checks for the mem_lookup_batched bf16/mem_scores changes and the new
# doc_access_per_query_loss (no TPU touched) — see
# wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md.
cd ~/memory-layers
echo "=== regression: test_mem_lookup_batched.py ==="
JAX_PLATFORMS=cpu uv run python tests/test_mem_lookup_batched.py
echo
echo "=== new: test_doc_access_per_query_loss.py ==="
JAX_PLATFORMS=cpu uv run python tests/test_doc_access_per_query_loss.py
