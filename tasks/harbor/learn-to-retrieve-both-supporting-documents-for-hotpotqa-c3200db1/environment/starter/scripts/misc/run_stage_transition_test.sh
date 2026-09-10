# CPU-only correctness check for the proposed stage-transition opt_state fix (no TPU touched) —
# see wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md.
cd ~/memory-layers
JAX_PLATFORMS=cpu uv run python tests/test_stage_transition_opt_state.py
