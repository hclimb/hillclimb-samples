"""B200 auxiliary-stream clock diagnostic, independent of validity and scoring."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from verifiers.worker import time_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if 'B200' not in torch.cuda.get_device_name():
        raise RuntimeError('Run this qualification diagnostic on the target B200')
    auxiliary = torch.cuda.Stream()

    def auxiliary_work():
        with torch.cuda.stream(auxiliary):
            torch.cuda._sleep(50000000)

    auxiliary_work()
    torch.cuda.synchronize()
    samples = []
    for repetition in range(3):
        started = time.perf_counter()
        _, seconds_per_call = time_call(auxiliary_work, calls=8)
        wall_seconds = time.perf_counter() - started
        measured_seconds = seconds_per_call * 8
        samples.append(dict(repetition=repetition, calls=8, measured_seconds=measured_seconds,
                            wall_seconds=wall_seconds, measured_over_wall=measured_seconds / wall_seconds))
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, samples=samples, diagnostic_only=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
