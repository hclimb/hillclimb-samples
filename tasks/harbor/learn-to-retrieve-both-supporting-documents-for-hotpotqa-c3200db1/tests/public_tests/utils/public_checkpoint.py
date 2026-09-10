"""Retain the submission, tested artifacts and results of each public invocation."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ARCHIVE_ROOT = Path('/logs/artifacts/public-verifier')
CHECKPOINT_SCRIPT = Path('/usr/local/bin/checkpoint.sh')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PublicRun:
    def __init__(self, output):
        self.output = Path(output).resolve()
        root = ARCHIVE_ROOT.resolve()
        if self.output == root or self.output in root.parents or root in self.output.parents:
            raise ValueError('Public output must be outside the managed checkpoint archive')
        for name in ('score.json', 'reward.json', 'diagnostics.json', 'validity.json', 'checkpoint.json'):
            (self.output / name).unlink(missing_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')
        self.path = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        self.results = self.path / 'results'
        self.results.mkdir()
        self.metadata = dict(schema_version=1, run_id=self.path.name, status='running',
                             command=sys.argv, archive=str(self.path), artifacts={})
        self._write()
        print(f'Public run archive: {self.path}', flush=True)

    def _write(self):
        content = json.dumps(self.metadata, indent=2)
        (self.results / 'checkpoint.json').write_text(content)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'checkpoint.json').write_text(content)

    def checkpoint(self, source):
        result = subprocess.run(
            ['bash', str(CHECKPOINT_SCRIPT), '--source', str(Path(source).resolve()),
             '--label', 'public-verifier', '--note', f'Public run: {self.path}'],
            check=True, capture_output=True, text=True, timeout=300,
        )
        print(result.stdout, end='', flush=True)
        prefix = 'Checkpoint saved: '
        paths = [line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)]
        if len(paths) != 1:
            raise RuntimeError('Checkpoint command did not report one saved checkpoint')
        path = Path(paths[0])
        metadata = dict(line.split('=', 1) for line in (path / 'metadata.txt').read_text().splitlines())
        self.metadata['source_checkpoint'] = dict(path=str(path), checkpoint_id=metadata['checkpoint_id'],
                                                  snapshot_sha256=metadata['snapshot_sha256'])
        self._write()

    def keep(self, source, name):
        source, target = Path(source), self.path / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            hashes = {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
        else:
            shutil.copyfile(source, target)
            hashes = {target.name: digest(target)}
        self.metadata['artifacts'][name] = dict(path=str(target), sha256=hashes)
        self._write()
        return target

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.metadata['status'] = 'error' if exception_type else 'finished'
        if exception_type:
            self.metadata['error'] = f'{exception_type.__name__}: {exception}'
        self._write()
        # Each run's authoritative reports remain in its unique archive. The
        # requested output is a convenient copy, so reusing it cannot erase history.
        self.output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.results, self.output, dirs_exist_ok=True)
        print(f'Public run saved: {self.path}', flush=True)
        return False
