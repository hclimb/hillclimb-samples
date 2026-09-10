"""Run only public data; this image has no private panel or legacy assets."""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    workspace = Path('/environment/starter')
    output = Path('/qualification-output')
    output.mkdir(exist_ok=True)
    evidence = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version',
                                        '--format=csv'], text=True)
    (output / 'hardware.txt').write_text(evidence)
    print(evidence, flush=True)
    started = time.monotonic()
    subprocess.run([sys.executable, '/environment/starter/optifine_public_tests/run.py',
                    '--submission', '/environment/starter/train_retriever.sh',
                    '--output', '/qualification-output/smoke', '--smoke'], cwd=workspace, check=True)
    measurements = []
    for recipe, seeds in [('control', [1729]), ('starter', [1729, 2027, 4093]),
                          ('reference', [1729, 2027, 4093])]:
        if recipe == 'reference':
            subprocess.run(['bash', '/reference/solve.sh'], cwd=workspace, check=True)
        (workspace / 'train_retriever.sh').write_text(
            '#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'
            f'exec torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 training.py --recipe {recipe} "$@"\n')
        for seed in seeds:
            destination = output / f'{recipe}-{seed}'
            command = [sys.executable, '/environment/starter/optifine_public_tests/run.py',
                       '--submission', '/environment/starter/train_retriever.sh', '--output', str(destination),
                       '--seed', str(seed)]
            with (output / f'{recipe}-{seed}.log').open('w') as stream:
                status = subprocess.run(command, cwd=workspace, stdout=stream, stderr=subprocess.STDOUT, timeout=1600)
            detail_path = destination / 'candidate/diagnostics.json'
            detail = json.loads(detail_path.read_text()) if detail_path.exists() else dict(valid=0)
            detail.pop('cases', None)
            record = dict(recipe=recipe, seed=seed, exit_code=status.returncode, **detail)
            measurements.append(record)
            print(json.dumps(record), flush=True)
            (output / 'measurements.json').write_text(json.dumps(measurements, indent=2))
            if status.returncode:
                raise RuntimeError('Public command failed; inspect logs before continuing')
    starter = [record['ndcg_at10'] for record in measurements if record['recipe'] == 'starter']
    reference = [record['ndcg_at10'] for record in measurements if record['recipe'] == 'reference']
    differences = [right - left for left, right in zip(starter, reference)]
    report = dict(seconds=time.monotonic() - started, starter_mean=statistics.mean(starter),
                  reference_mean=statistics.mean(reference), paired_differences=differences,
                  paired_difference_mean=statistics.mean(differences),
                  paired_difference_stdev=statistics.stdev(differences),
                  consistent_gain=all(value > 0 for value in differences),
                  solver_trials='Not yet run; required separately')
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
