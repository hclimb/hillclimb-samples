import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.metrics import aggregate
from utils.process import ROOT, execute, repetitions
from utils.protocol import PUBLIC_SEEDS, REPETITIONS
from utils.public_checkpoint import PublicRun


def evaluate(checkout, seeds, output, build_candidate=False, case=None, implementation=None):
    output.mkdir(parents=True, exist_ok=True)
    reward_path = output / 'reward.json'
    reward_path.unlink(missing_ok=True)
    diagnostics = dict(candidate=[], phase='setup')
    reward = dict(valid=0, reward=0)
    try:
        manifest = json.loads((ROOT / 'utils/manifest.json').read_text())
        if len(seeds) != REPETITIONS or len(set(seeds)) != REPETITIONS:
            raise ValueError(f'Expected {REPETITIONS} distinct seeds')
        cases = [item for item in manifest['cases'] if case is None or item['name'] == case]
        if not cases:
            raise ValueError('Unknown case name')
        if build_candidate:
            diagnostics['phase'] = 'build'
            diagnostics['build_output'] = execute(['bash', 'solve.sh'], checkout, 1800)
        for run_index, seed in enumerate(seeds):
            diagnostics['run_index'] = run_index
            diagnostics['phase'] = 'candidate'
            repetitions(implementation or checkout / '.flashmla-build/site', [seed], case, diagnostics['candidate'])
        reward, metrics = aggregate(diagnostics['candidate'], cases, diagnostic_only=case is not None)
        diagnostics.update(metrics)
        diagnostics['subset_diagnostic_only'] = case is not None
        diagnostics['phase'] = 'complete'
    except Exception as exception:
        diagnostics['error'] = str(exception)[-14000:]
        diagnostics['evaluator_failure'] = (diagnostics['phase'] not in ('build', 'candidate') or
                                            isinstance(exception, OSError))
    (output / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    if diagnostics.get('error'):
        print(diagnostics['error'], file=sys.stderr)
    if diagnostics.get('evaluator_failure'):
        raise RuntimeError('Evaluator infrastructure failure, not candidate invalidity')
    reward_path.write_text(json.dumps(reward, indent=2, allow_nan=False))
    print(json.dumps(reward), flush=True)
    return reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--output', type=Path, default=Path('attention-results'))
    arguments = parser.parse_args()
    if arguments.case:
        manifest = json.loads((ROOT / 'utils/manifest.json').read_text())
        if arguments.case not in {case['name'] for case in manifest['cases']}:
            parser.error('Unknown case name')
    with PublicRun(arguments.output) as saved, tempfile.TemporaryDirectory(prefix='attention-public-') as temporary:
        frozen = Path(temporary) / 'submission'
        shutil.copytree(arguments.checkout.resolve(), frozen,
                        ignore=shutil.ignore_patterns('.git', 'build', '__pycache__'))
        saved.checkpoint(frozen)
        implementation = saved.keep(frozen / '.flashmla-build/site', 'kernel')
        evaluate(frozen, PUBLIC_SEEDS, saved.results, case=arguments.case, implementation=implementation)


if __name__ == '__main__':
    main()
