"""Measure candidate and frozen baseline throughput on the same GPU."""

import secrets
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiers.benchmark import evaluate
from utils.protocol import REPETITIONS


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    seeds = secrets.SystemRandom().sample(range(2 ** 31), REPETITIONS)
    evaluate(Path('/environment/starter'), seeds,
             Path('/logs/verifier'), build_candidate=True)
