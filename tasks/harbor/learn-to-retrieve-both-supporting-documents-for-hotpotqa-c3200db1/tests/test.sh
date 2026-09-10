#!/usr/bin/env bash
set -euo pipefail
cd /environment/starter
export PYTHONPATH="/tests:/environment/starter${PYTHONPATH:+:$PYTHONPATH}"
export OPTIFINE_STARTER="/environment/starter"
python -I /tests/verifiers/verify.py
