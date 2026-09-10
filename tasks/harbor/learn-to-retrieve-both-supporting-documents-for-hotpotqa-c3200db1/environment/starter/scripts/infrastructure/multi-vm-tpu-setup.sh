#!/usr/bin/env bash
#
# Provision a TPU box for a run. Idempotent: each expensive step is skipped when it's already
# done, so re-running on a pre-existing box only re-syncs the code (seconds) instead of
# rebuilding the venv and reinstalling deps (minutes). Run from the home dir where
# multi-vm-tpu-run.sh dropped .env + memory_layers_sync.tar.gz.
set -uo pipefail

# ── uv: install only if missing ─────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1090,SC1091
. "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

# ── git identity from .env ─────────────────────────────────────────────────────────────────
if [ -f .env ]; then
  # shellcheck disable=SC1091
  . .env
else
  echo "[setup] WARNING: .env not found (expected WANDB_API_KEY, GIT_USER_EMAIL, GIT_USER_NAME)."
fi
git config --global user.email "${GIT_USER_EMAIL:-}"
git config --global user.name  "${GIT_USER_NAME:-}"

# ── code sync: extract fresh code but PRESERVE the venv (and .git) so we don't rebuild ──────
if [ -f memory_layers_sync.tar.gz ]; then
  if [ -d memory-layers ]; then
    echo "[setup] refreshing code (preserving .venv)..."
    find memory-layers -mindepth 1 -maxdepth 1 ! -name .venv ! -name .git -exec rm -rf {} +
  else
    echo "[setup] extracting code (fresh)..."
    mkdir memory-layers
  fi
  tar -xzf memory_layers_sync.tar.gz -C memory-layers
  rm -f memory_layers_sync.tar.gz
elif [ ! -d memory-layers ]; then
  echo "[setup] no tarball; cloning..."
  git clone "https://${GIT_PAT}@github.com/rohunagrawal/memory-layers"
  [ -n "${1:-}" ] && ( cd memory-layers && git checkout "$1" )
fi

cd memory-layers || { echo "[setup] ERROR: no memory-layers dir" >&2; exit 1; }

# ── venv: create only if missing ───────────────────────────────────────────────────────────
if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating venv..."
  uv venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

# ── deps: (re)install only when pyproject/lock changed (hash marker in the venv) ────────────
dep_files="pyproject.toml"; [ -f uv.lock ] && dep_files="$dep_files uv.lock"
# shellcheck disable=SC2086
DEP_HASH="$(cat $dep_files | sha256sum | cut -d' ' -f1)"
MARKER=".venv/.deps-sha256"
if [ "$(cat "$MARKER" 2>/dev/null || true)" != "$DEP_HASH" ]; then
  echo "[setup] installing deps (fresh venv or pyproject changed)..."
  uv pip install -e . && uv pip install jax==0.8.0 && echo "$DEP_HASH" > "$MARKER"
else
  echo "[setup] deps up to date — skipping install"
fi

# ── wandb: log in only if not already authenticated ────────────────────────────────────────
if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "[setup] wandb already logged in — skipping"
elif [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login "${WANDB_API_KEY}"
else
  echo "[setup] no WANDB_API_KEY — skipping wandb login"
fi

echo "[setup] done"
