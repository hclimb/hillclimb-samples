"""Adapt the compiler-generated launcher before it is packaged or tested."""
import shlex
from pathlib import Path


def prepare(launcher):
    lines = launcher.read_text().splitlines(keepends=True)
    commands = [['python', '/tests/verifiers/verify.py'],
                ['python', '-I', '/tests/verifiers/verify.py']]
    matches = [index for index, line in enumerate(lines) if shlex.split(line) in commands]
    if len(matches) != 1:
        raise RuntimeError('Unsupported generated verifier launcher; isolated startup cannot be established')
    lines[matches[0]] = 'python -I /tests/verifiers/verify.py\n'
    launcher.write_text(''.join(lines))


if __name__ == '__main__':
    prepare(Path(__file__).resolve().parents[1] / 'test.sh')
