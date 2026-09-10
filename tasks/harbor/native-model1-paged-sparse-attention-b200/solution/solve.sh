#!/bin/bash
set -euo pipefail
solution_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
cd /environment/starter
patch --forward -p1 < "$solution_dir/direct_schedule.patch"
bash /environment/starter/solve.sh
python /environment/starter/optifine_public_tests/verifiers/check_timing.py \
  --output /logs/agent/attention-timing.json
python /environment/starter/optifine_public_tests/verifiers/benchmark.py \
  --checkout /environment/starter --output /logs/agent/attention-public
python -c 'import json; from pathlib import Path; result = json.loads(Path("/logs/agent/attention-public/reward.json").read_text()); assert result["valid"] == 1, result'
