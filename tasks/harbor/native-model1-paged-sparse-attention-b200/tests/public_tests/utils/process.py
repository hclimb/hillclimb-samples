import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def execute(command, cwd, timeout):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment['PYTHONNOUSERSITE'] = '1'
    with tempfile.TemporaryFile(mode='w+b') as output:
        child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True, text=True)
        timed_out = False
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        output.seek(0, os.SEEK_END)
        output.seek(max(0, output.tell() - 12000))
        excerpt = output.read().decode('utf-8', errors='replace')
    if timed_out or child.returncode:
        raise RuntimeError(f'Command {command} failed; timeout={timed_out}; '
                           f'exit={child.returncode}\n{excerpt}')
    return excerpt


def repetitions(implementation, seeds, case=None, reports=None):
    if reports is None:
        reports = []
    for seed in seeds:
        with tempfile.TemporaryDirectory(prefix='attention-run-') as temporary:
            report_path = Path(temporary) / 'report.json'
            command = [sys.executable, str(ROOT / 'verifiers/worker.py'),
                       '--implementation', str(implementation), '--seed', str(seed),
                       '--output', str(report_path)]
            if case:
                command += ['--case', case]
            execute(command, temporary, 180)
            reports.append(json.loads(report_path.read_text()))
    return reports
