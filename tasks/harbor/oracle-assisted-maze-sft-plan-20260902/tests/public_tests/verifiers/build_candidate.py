#!/usr/bin/env python3
"""Resource-bounded candidate subprocess; emits untrusted JSON selections."""

import importlib.util
import json
from pathlib import Path
import resource
import sys

import numpy  # noqa: F401 -- Check candidate dependencies during preflight.
import torch


def main():
    inputs, output = map(Path, sys.argv[1:3])
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8_000_000, 8_000_000))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    torch.set_num_threads(1)
    if sys.argv[3:] == ["--check"]:
        print("builder-ready")
        return
    # These inputs are written by the parent; no private evaluation groups are included.
    groups = torch.load(inputs / "groups.pt", map_location="cpu", weights_only=False)
    spec = importlib.util.spec_from_file_location("maze_candidate", inputs / "candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_examples", None)):
        raise ValueError("candidate.py must define callable build_examples")
    selections = module.build_examples(groups)
    (output / "selections.json").write_text(json.dumps(selections))


if __name__ == "__main__":
    main()
