#!/bin/bash
set -euo pipefail
cd /environment/starter
python optifine_public_tests/verifiers/build.py --checkout /environment/starter
