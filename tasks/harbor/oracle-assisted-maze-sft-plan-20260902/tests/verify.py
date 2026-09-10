#!/usr/bin/env python3
"""Protected private evaluator for oracle-assisted maze SFT."""

import json
import os
from pathlib import Path
import sys
import traceback

import torch

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from utils.contract import CandidateError
from utils.runner import compare

LOG_DIR = Path("/logs/verifier")
STARTER = Path(os.environ.get("STARTER_ROOT", "/environment/starter"))


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("infrastructure error: CUDA GPU is required")
    try:
        result = compare(STARTER, "full", 2_000_000, torch.device("cuda"), private=True)
    except CandidateError as error:
        (LOG_DIR / "diagnostics.json").write_text(
            json.dumps({"candidate_error": str(error)}, indent=2)
        )
        (LOG_DIR / "reward.json").write_text(json.dumps({"valid": 0, "reward": 0.0}))
        return
    except Exception as error:
        excerpt = "".join(traceback.format_exception(error))[-8000:]
        (LOG_DIR / "diagnostics.json").write_text(json.dumps({
            "infrastructure_error": str(error), "traceback_excerpt": excerpt
        }, indent=2))
        raise
    (LOG_DIR / "diagnostics.json").write_text(json.dumps(result, indent=2))
    (LOG_DIR / "reward.json").write_text(
        json.dumps({"valid": 1, "reward": result["reward"]})
    )


if __name__ == "__main__":
    main()
