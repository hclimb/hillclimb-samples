FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel@sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89
WORKDIR /environment/starter
# Task dependencies belong to this disposable image, not the host Python.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ninja-build ripgrep rsync zstd tmux asciinema && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pytest torch==2.11.0 numpy==2.2.6 packaging==25.0 setuptools==81.0.0 wheel==0.45.1 ninja==1.13.0 filelock==3.32.5 typing-extensions==4.16.0 sympy==1.14.0 networkx==3.6.1 jinja2==3.1.6 fsspec==2026.7.0 mpmath==1.3.0 markupsafe==3.0.3
COPY checkpoint.sh /usr/local/bin/checkpoint.sh
RUN chmod 0555 /usr/local/bin/checkpoint.sh
COPY stop_agent_processes.sh /usr/local/libexec/stop_agent_processes.sh
RUN chmod 0555 /usr/local/libexec/stop_agent_processes.sh
COPY starter/ /environment/starter/
