import argparse
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path


def main():
    if not sys.flags.isolated or os.geteuid() != 0:
        raise RuntimeError('Use verifiers/launch.sh with Linux root isolation')
    trusted = Path(__file__).resolve().parents[1]
    trusted.chmod(0o700)
    sys.path.insert(0, str(trusted))
    from verifiers.runner import run
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('/environment/starter'))
    parser.add_argument('--submission', type=Path, default=Path('/environment/starter/train_retriever.sh'))
    parser.add_argument('--output', type=Path, default=Path('/logs/verifier'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o700)
    (args.output / 'reward.json').unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='tiny-public-assets-') as temporary:
            root = Path(temporary)
            root.chmod(0o755)
            shutil.copytree(trusted / 'public_tests/assets', root / 'assets')
            result = run(args.workspace, args.submission, root / 'assets', trusted / 'private_panel',
                         args.output, trusted / 'public_tests', paired=True)
        numbers = dict(valid=result['valid'], reward=result['R'],
                       baseline_ndcg=result['B'], ndcg_at10=result['C'])
        (args.output / 'reward.json').write_text(json.dumps(numbers, allow_nan=False))
    except Exception as error:
        (args.output / 'diagnostics.json').write_text(json.dumps(dict(evaluator_error=str(error))))
        raise


if __name__ == '__main__':
    main()
