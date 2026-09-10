import argparse
import json
from pathlib import Path

from verifiers.runner import isolated_command, run
from utils.process import phase
from utils.public_checkpoint import PublicRun


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=Path, default=Path('train_retriever.sh'))
    parser.add_argument('--output', type=Path, default=Path('runs/public'))
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--check-weights', type=Path)
    args = parser.parse_args()
    trusted = Path(__file__).resolve().parents[1]
    if args.paired and args.smoke:
        parser.error('--smoke is a validity check, not a paired quality measurement')
    with PublicRun(args.output) as saved:
        if args.check_weights:
            saved.checkpoint(Path.cwd())
            weights = saved.keep(args.check_weights.resolve(), 'model.safetensors')
            arguments = ['--models', str(trusted / 'assets/models/bert-tiny'), '--weights', str(weights),
                         '--corpus', '/unused', '--queries', '/unused', '--output', str(saved.results / 'validity.json'), '--validity']
            result = phase(isolated_command(trusted, 'verifiers.inference', arguments), trusted, saved.results, 'validity', 60)
        else:
            result = run(Path.cwd(), args.submission, trusted / 'assets', trusted / 'public_panel', saved.results,
                         trusted, args.paired, args.seed, args.smoke, public_checkpoint=saved)
    print(json.dumps(result, indent=2))
    if result.get('valid', 1) == 0:
        raise SystemExit(1)
