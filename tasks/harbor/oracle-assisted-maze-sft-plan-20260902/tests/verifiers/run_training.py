#!/usr/bin/env python3
"""Trusted clean-process baseline or candidate training worker."""

import json
from pathlib import Path
import sys

import torch

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
from utils.contract import CandidateError, load_examples
from utils.maze import build_groups
from utils.model import initialized_model
from utils.runner import CONFIGS, canonical_arrays, evaluate, run_schedule, train_model, validate_model


def main():
    method, mode, training_path, output_path, evaluation_path, device_name, *artifact = sys.argv[1:]
    settings = CONFIGS[mode]
    training_runs = json.loads(Path(training_path).read_text())
    evaluation_runs = json.loads(Path(evaluation_path).read_text())
    if len(training_runs) not in (3, 6) or len(evaluation_runs) != len(training_runs):
        raise RuntimeError("Training and evaluation panel counts must match the full comparison")
    runs = []
    device = torch.device(device_name)
    prepared_panel = None
    for panel, run_seed in run_schedule(mode, len(training_runs)):
        if panel != prepared_panel:
            arrays = (load_examples(artifact[panel]) if method == "candidate" else
                      canonical_arrays(build_groups(training_runs[panel], include_catalog=False)))
            evaluation = build_groups(evaluation_runs[panel], include_catalog=False)
            prepared_panel = panel
        torch.manual_seed(run_seed)
        model = initialized_model()
        initial_keys = tuple(model.state_dict())
        trained = train_model(model, arrays, {**settings, "seed": run_seed}, device)
        validate_model(trained, initial_keys)
        metrics = evaluate(trained, evaluation, device)
        runs.append({"panel": panel, "seed": run_seed, **metrics})
    Path(output_path).write_text(json.dumps(runs))


if __name__ == "__main__":
    try:
        main()
    except CandidateError as error:
        print(f"CANDIDATE_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
