#!/bin/bash
# Install system basics
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libopenblas-dev golang tmux

# Install uv package manager
if [ ! -d "$HOME/.local/bin" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source $HOME/.local/bin/env

# Git setup
git config --global user.email "${GIT_USER_EMAIL}"
git config --global user.name "${GIT_USER_NAME}"

if [ ! -d "memory-layers" ]; then
    git clone https://${GIT_PAT}@github.com/rohunagrawal/memory-layers
    cd memory-layers
    git checkout main
    uv venv
    source .venv/bin/activate
    uv pip install -e .
else
    cd memory-layers
    git remote set-url origin https://${GIT_PAT}@github.com/rohunagrawal/memory-layers
    git reset --hard
    git checkout main
    git pull origin main
    source .venv/bin/activate
fi
