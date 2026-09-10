#!/usr/bin/env python3
"""Public contract check and paired comparisons over disjoint maze panels."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
from utils.runner import CONFIGS, PUBLIC_FOLDS, compare, construct_examples, seed_sets  # noqa: E402
from utils.contract import load_examples  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "quick", "full"), default="quick")
    parser.add_argument("--validation-fold", type=int, choices=PUBLIC_FOLDS,
                        help="select three disjoint public panels (full mode only; default 0)")
    args = parser.parse_args()
    if args.validation_fold is not None and args.mode != "full":
        parser.error("--validation-fold requires --mode full")
    starter = os.environ.get("STARTER_ROOT", "/environment/starter")
    if args.mode == "contract":
        with tempfile.TemporaryDirectory(prefix="maze-contract-") as temporary:
            artifact = Path(temporary) / "examples.npz"
            construct_examples(starter, seed_sets("quick", 1_000_000)[0][0][:4], artifact)
            print(json.dumps({"valid": 1, "examples": len(load_examples(artifact)[1]),
                              "note": "Contract check only; no training or quality score."}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("public experiments require a CUDA GPU")
    note = ("Smoke test only; do not use quick scores to select the final candidate."
            if args.mode == "quick" else
            "Final-step mean quality (0-1) over three panels and two seeds per panel; "
            "public validation is not a private reward.")
    print(f"[maze] {note}", file=sys.stderr, flush=True)
    result = compare(starter, args.mode, 1_000_000, torch.device("cuda"),
                     public_fold=args.validation_fold)
    result.update(mode=args.mode, validation_fold=args.validation_fold,
                  settings=CONFIGS[args.mode], metric="mean_generated_path_quality", note=note)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
