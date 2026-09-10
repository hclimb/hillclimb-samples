import os
import signal
import subprocess
import time
from pathlib import Path


class PhaseFailure(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(str(detail))


def writable(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o755)
    if os.geteuid() == 0:
        os.chown(path, 65534, 65534)
    return path


def phase(command, cwd, output, name, seconds, untrusted=False):
    env = {key: os.environ[key] for key in ('PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES')
           if key in os.environ}
    env.update(HOME=str(writable(output / ('home-' + name))), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    options = dict(user=65534, group=65534, extra_groups=[]) if untrusted and os.geteuid() == 0 else {}
    log = output / (name + '.log')
    started, timed_out = time.monotonic(), False
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, **options)
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    elapsed = time.monotonic() - started
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 8000))
        excerpt = stream.read().decode(errors='replace')
    detail = dict(seconds=elapsed, timeout=timed_out, exit_code=process.returncode, output_excerpt=excerpt)
    if timed_out or process.returncode or elapsed > seconds:
        raise PhaseFailure(detail)
    return detail
