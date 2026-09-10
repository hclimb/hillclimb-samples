import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELATIVE_SOURCE = Path('native_paged_sparse_prefill/upstreams/FlashMLA')


def run(command, directory, environment, log_path):
    try:
        result = subprocess.run(command, cwd=directory, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    except subprocess.TimeoutExpired as exception:
        excerpt = exception.stdout or b''
        if isinstance(excerpt, bytes):
            excerpt = excerpt.decode('utf-8', errors='replace')
        log_path.write_text(excerpt)
        raise RuntimeError(f'Build timeout:\n{excerpt[-12000:]}') from exception
    log_path.write_text(result.stdout)
    if result.returncode:
        errors = '\n'.join(line for line in result.stdout.splitlines()
                           if any(marker in line.lower() for marker in
                                  ['error', 'failed:', 'killed', 'undefined reference']))
        raise RuntimeError(f'Build failed ({result.returncode}); complete log: {log_path}\n'
                           f'{errors[:8000]}\n{result.stdout[-4000:]}')
    print(result.stdout[-3000:])


def unpack_pinned(name, destination):
    lock = json.loads((ROOT / 'utils/artifacts.json').read_text())
    archive = ROOT / 'utils' / name
    if hashlib.sha256(archive.read_bytes()).hexdigest() != lock[name]['sha256']:
        raise RuntimeError(f'Corrupt pinned artifact: {name}')
    with tarfile.open(archive) as source:
        source.extractall(destination, filter='data')


def compile_wheel(source, destination):
    import torch
    if not hasattr(torch, 'float8_e8m0fnu') or not torch.version.cuda:
        raise RuntimeError('Requires CUDA Torch with float8_e8m0fnu')
    if not (source / 'csrc/cutlass/include/cutlass/cutlass.h').exists():
        unpack_pinned('cutlass.tar.gz', source / 'csrc/cutlass')
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_HOME='/usr/local/cuda', MAX_JOBS='8', NVCC_THREADS='2',
                       FLASH_MLA_DISABLE_SM90='1', PIP_NO_INDEX='1')
    cccl = Path(environment['CUDA_HOME']) / 'include/cccl'
    if cccl.is_dir():
        environment['CPATH'] = os.pathsep.join(filter(None, [str(cccl), environment.get('CPATH')]))
    # One setup invocation keeps the archive's timestamp-based package version stable.
    run([sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir', str(destination)],
        source, environment, destination / 'build.log')
    wheels = sorted(destination.glob('flash_mla-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(f'Expected one wheel, got {wheels}')
    wheel = wheels[0]
    site = destination / 'site'
    if site.exists():
        shutil.rmtree(site)
    with zipfile.ZipFile(wheel) as package:
        package.extractall(site)
    record = dict(wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True))
    (destination / 'build.json').write_text(json.dumps(record, indent=2))
    return site


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, default=Path('/environment/starter'))
    parser.add_argument('--incumbent', type=Path)
    arguments = parser.parse_args()
    if arguments.incumbent:
        destination = arguments.incumbent.resolve()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'FlashMLA'
            unpack_pinned('native.tar.gz', source)
            compile_wheel(source, destination)
    else:
        checkout = arguments.checkout.resolve()
        destination = checkout / '.flashmla-build'
        if destination.exists():
            shutil.rmtree(destination)
        compile_wheel(checkout / RELATIVE_SOURCE, destination)


if __name__ == '__main__':
    main()
