import json
import errno
import math
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

from utils.oracle import reward, score
from utils.process import PhaseFailure, phase, writable


EXCLUDED = {'.git', '.venv', 'venv', 'env', 'assets', 'runs', 'optifine_public_tests',
            'hotpot_benchmark', '__pycache__', '.cache', '.pytest_cache', '.ruff_cache'}


class InvalidSubmission(ValueError):
    pass


def read_rows(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def snapshot(source, destination):
    def ignore(directory, names):
        return [name for name in names if name in EXCLUDED]
    shutil.copytree(source, destination, ignore=ignore, symlinks=True)
    for path in destination.rglob('*'):
        if path.is_symlink():
            raise InvalidSubmission(f'Submitted source symlinks are unsupported: {path.name}')
        path.chmod(0o755 if path.is_dir() or path.suffix == '.sh' else 0o644)
        if os.geteuid() == 0:
            os.chown(path, 65534, 65534)
    writable(destination)


def isolated_command(root, module, arguments):
    code = 'import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module(sys.argv.pop(1),run_name="__main__")'
    return [sys.executable, '-I', '-c', code, str(root), module, *arguments]


def check_environment(assets, output, trusted):
    for relative in ('hotpotqa/train/questions.jsonl', 'hotpotqa/train/corpus.jsonl',
                     'models/bert-tiny/config.json', 'models/bert-tiny/vocab.txt',
                     'models/bert-tiny/model.safetensors'):
        if not (assets / relative).is_file():
            raise FileNotFoundError(f'Missing packaged asset: {relative}')
    code = ('import torch; assert torch.cuda.device_count() >= 2, "Two CUDA GPUs are required"; '
            '[torch.zeros(1, device=f"cuda:{index}") for index in range(2)]; torch.cuda.synchronize()')
    phase([sys.executable, '-I', '-c', code], trusted, output, 'environment', 30)


def run_one(tree, entry, assets, panel, output, trusted, seed=1729, smoke=False, snapshot_seconds=0,
            public_checkpoint=None):
    output.mkdir(parents=True, exist_ok=True)
    detail = {}
    with tempfile.TemporaryDirectory(prefix='tiny-phase-') as temporary:
        root = Path(temporary)
        root.chmod(0o755)
        train_output = writable(root / 'training')
        artifacts = writable(train_output / 'artifacts')
        try:
            command = ['bash', str(tree / entry), '--train_dir', str(assets / 'hotpotqa/train'),
                       '--models_dir', str(assets / 'models'), '--output_dir', str(artifacts), '--seed', str(seed)]
            if smoke:
                command += ['--steps', '2']
            limit = (60 if smoke else 900) - snapshot_seconds
            if limit <= 0:
                raise InvalidSubmission('Source snapshot exhausted the training budget')
            detail['training'] = phase(command, tree, train_output, 'training', limit, True)
            detail['training']['seconds'] += snapshot_seconds
            detail['training']['snapshot_seconds'] = snapshot_seconds
            shutil.copyfile(train_output / 'training.log', output / 'training.log')
            started = time.monotonic()
            inference = root / 'inference'
            inference.mkdir(mode=0o700)
            checkpoint = artifacts / 'model.safetensors'
            try:
                descriptor = os.open(checkpoint, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except OSError as error:
                if error.errno not in (errno.ENOENT, errno.ELOOP):
                    raise
                raise InvalidSubmission('Missing or symlinked model.safetensors') from error
            with os.fdopen(descriptor, 'rb') as stream:
                status = os.fstat(stream.fileno())
                if not stat.S_ISREG(status.st_mode) or status.st_size > 32_000_000:
                    raise InvalidSubmission('Expected regular safetensors checkpoint below 32 MB')
                (inference / 'model.safetensors').write_bytes(stream.read(32_000_001))
            arguments = ['--models', str(assets / 'models/bert-tiny'), '--weights', str(inference / 'model.safetensors'),
                         '--corpus', str(panel / 'corpus.jsonl'), '--queries', str(panel / 'queries.jsonl'),
                         '--output', str(inference / 'retrieval_results.json')]
            if smoke:
                arguments += ['--validity']
            preparation = time.monotonic() - started
            if public_checkpoint is not None:
                # Archive the exact trusted copy that inference will read. This
                # archival I/O is outside the existing training/retrieval budgets.
                public_checkpoint.keep(inference / 'model.safetensors', 'model.safetensors')
            try:
                detail['retrieval'] = phase(isolated_command(trusted, 'verifiers.inference', arguments),
                                            trusted, inference, 'retrieval', (60 if smoke else 600) - preparation)
            except PhaseFailure as error:
                if not error.detail['timeout'] and error.detail['exit_code'] != 2:
                    raise RuntimeError(f'Trusted retrieval failed: {error.detail}') from error
                raise
            detail['retrieval']['seconds'] += preparation
            shutil.copyfile(inference / 'retrieval.log', output / 'retrieval.log')
            if not smoke:
                detail.update(score(json.loads((inference / 'retrieval_results.json').read_text()),
                                    read_rows(panel / 'queries.jsonl'), read_rows(panel / 'gold.jsonl'),
                                    read_rows(panel / 'slices.jsonl'), 60000))
                shutil.copyfile(inference / 'retrieval_results.json', output / 'retrieval_results.json')
            detail['valid'] = 1
        except (InvalidSubmission, PhaseFailure) as error:
            detail.update(valid=0, error=str(error))
            if isinstance(error, PhaseFailure):
                detail['failure'] = error.detail
    (output / 'diagnostics.json').write_text(json.dumps(detail, indent=2, allow_nan=False))
    return detail


def run(workspace, entry, assets, panel, output, trusted, paired=False, seed=1729, smoke=False,
        public_checkpoint=None):
    paired = paired or not smoke  # Quality scores always compare against the measured starter.
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('score.json', 'reward.json', 'diagnostics.json'):
        (output / filename).unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix='tiny-snapshot-') as temporary:
        root = Path(temporary)
        root.chmod(0o755)
        started = time.monotonic()
        preparation_error = None
        try:
            entry = entry.resolve().relative_to(workspace.resolve())
            snapshot(workspace, root / 'candidate')
        except ValueError as error:
            preparation_error = str(error)
        candidate_snapshot = time.monotonic() - started
        if public_checkpoint is not None:
            public_checkpoint.checkpoint(root / 'candidate' if preparation_error is None else workspace)
        check_environment(assets, output, trusted)
        baseline = None
        if paired:
            started = time.monotonic()
            snapshot(trusted / 'incumbent', root / 'baseline')
            baseline_snapshot = time.monotonic() - started
            baseline = run_one(root / 'baseline', Path('train_retriever.sh'), assets, panel,
                               output / 'baseline', trusted, seed, smoke, baseline_snapshot)
            baseline_ndcg = baseline.get('ndcg_at10', float('nan'))
            if not baseline['valid'] or not math.isfinite(baseline_ndcg) or not 0 <= baseline_ndcg <= 1:
                raise RuntimeError(f'Baseline failure: {baseline}')
        if preparation_error is None:
            candidate = run_one(root / 'candidate', entry, assets, panel, output / 'candidate', trusted,
                                seed, smoke, candidate_snapshot, public_checkpoint)
        else:
            candidate = dict(valid=0, error=preparation_error, snapshot_seconds=candidate_snapshot)
            (output / 'candidate').mkdir(exist_ok=True)
            (output / 'candidate/diagnostics.json').write_text(json.dumps(candidate))
    result = dict(valid=candidate['valid'], diagnostic_only=int(smoke))
    if not smoke:
        result['C'] = candidate.get('ndcg_at10', 0.0)
        result['R'] = reward(result['C'], baseline['ndcg_at10']) if candidate['valid'] else 0.0
    if paired:
        result['B'] = baseline['ndcg_at10']
    (output / 'score.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (output / 'diagnostics.json').write_text(json.dumps(dict(baseline=baseline, candidate=candidate), indent=2))
    return result
