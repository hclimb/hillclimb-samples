FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
WORKDIR /environment/starter
# Task dependencies belong to this disposable image, not the host Python.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ninja-build ripgrep rsync zstd && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pytest torch==2.9.1 transformers==4.57.6 tokenizers==0.22.2 numpy==2.2.6 safetensors==0.7.0 huggingface_hub==0.36.0
COPY . /tests/
