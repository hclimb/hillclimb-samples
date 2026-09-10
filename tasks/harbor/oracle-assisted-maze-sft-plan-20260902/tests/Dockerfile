FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /environment/starter
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ninja-build ripgrep rsync zstd && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pytest torch==2.6.0 numpy==2.1.2
COPY . /tests/
