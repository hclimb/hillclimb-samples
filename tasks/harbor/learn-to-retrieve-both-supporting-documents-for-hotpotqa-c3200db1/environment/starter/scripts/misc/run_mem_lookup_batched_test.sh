# CPU-only correctness check for mem_lookup_batched (no TPU touched) — see
# wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md.
cd ~/memory-layers
JAX_PLATFORMS=cpu uv run python tests/test_mem_lookup_batched.py
