#!/usr/bin/env bash
set -euo pipefail
solution_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd /environment/starter
tar -xzf "$solution_dir/reference.tar.gz" -C /environment/starter
