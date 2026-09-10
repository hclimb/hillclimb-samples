#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 training.py "$@"
