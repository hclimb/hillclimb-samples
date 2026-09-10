#!/bin/bash

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
. $HOME/.local/bin/env

# install claude code
curl -fsSL https://claude.ai/install.sh | bash

# Get the root directory of the repository (this script lives in scripts/infrastructure/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# configure git
# . .env if it exists
if [ -f "$REPO_ROOT/.env" ]; then
    . "$REPO_ROOT/.env"
else
    echo "WARNING: $REPO_ROOT/.env file not found. Please create one with WANDB_API_KEY, GIT_USER_EMAIL, and GIT_USER_NAME."
fi

# create virtual environment
cd "$REPO_ROOT"
activate

# install repo
uv pip install -e .
uv pip install jax==0.8.0

# set git attributes
git config --global user.email "${GIT_USER_EMAIL}"
git config --global user.name "${GIT_USER_NAME}"

# wandb login
uv run wandb login ${WANDB_API_KEY}

# login to gcloud
gcloud auth login

