#!/bin/bash
# Install system basics
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libopenblas-dev golang tmux gcsfuse

# Install uv package manager securely
if [ ! -d "$HOME/.local/bin" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source $HOME/.local/bin/env

# Git setup
git config --global user.email "${GIT_USER_EMAIL}"
git config --global user.name "${GIT_USER_NAME}"

GIT_BRANCH=${GIT_BRANCH:-FlashRAG}

if [ ! -d "memory-layers" ]; then
    git clone https://${GIT_PAT}@github.com/rohunagrawal/memory-layers
    cd memory-layers
    git checkout ${GIT_BRANCH}
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    uv pip install vllm==0.12.0 vllm-tpu==0.12.0 openai
else
    cd memory-layers
    git remote set-url origin https://${GIT_PAT}@github.com/rohunagrawal/memory-layers
    git reset --hard
    git checkout ${GIT_BRANCH}
    git pull origin ${GIT_BRANCH}
    source .venv/bin/activate
    uv pip install vllm==0.12.0 vllm-tpu==0.12.0 openai
fi

# Mount GCS outputs bucket (no-op if already mounted)
mkdir -p outputs
gcsfuse --implicit-dirs memorylayers outputs 2>/dev/null || true
