#!/bin/bash
# Run the opt_state sharding probe. Multi-host: launch on every worker simultaneously
# via multi-tpu-box-run.sh — init_jax_distributed blocks until both peers show up.
# PROBE_TP_DEVICES defaults to 1; override with RUN_ENV="PROBE_TP_DEVICES=2".
uv run python scripts/misc/_probe_opt_state_shape.py
