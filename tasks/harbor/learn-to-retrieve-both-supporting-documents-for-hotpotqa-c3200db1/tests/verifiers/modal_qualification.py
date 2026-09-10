"""Build an offline solver-only candidate image and record owned GPU execution."""
import argparse
import base64
import json
import hashlib
import os
import time
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--mode', choices=['public', 'boundary'], default='public')
    parser.add_argument('--compiled-task', type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    plan = json.loads((build / 'plan.json').read_text())
    driver = 'gpu_qualification.py' if args.mode == 'public' else 'boundary_qualification.py'
    if args.mode == 'boundary' and args.compiled_task is None:
        parser.error('--mode boundary requires --compiled-task with its adapted generated launcher')
    if args.compiled_task:
        compiled = args.compiled_task.resolve()
        image = modal.Image.from_dockerfile(compiled / 'environment/Dockerfile',
                                           context_dir=compiled / 'environment')
    else:
        image = (modal.Image.from_registry(plan['resources']['container_image'], add_python='3.11')
                 .pip_install(*plan['required_python_packages'])
                 .add_local_dir(build / 'starter-overlay', '/environment/starter', copy=True)
                 .add_local_dir(build / 'evaluator/public_tests', '/environment/starter/optifine_public_tests', copy=True))
    image = (image.add_local_file(build / 'evaluator/verifiers' / driver, '/qualification.py', copy=True)
             .add_local_dir(build / 'solution', '/reference', copy=True)
             .run_commands('chmod 700 /reference')
             .env({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
             .workdir('/environment/starter'))
    if args.mode == 'boundary':
        image = (image.add_local_dir(compiled / 'tests', '/tests', copy=True,
                                     ignore=['evidence/**', '**/__pycache__/**'])
                 .run_commands('chmod 700 /tests', 'chown -R 65534:65534 /environment/starter',
                               'mkdir -p /logs/artifacts && chmod 777 /logs/artifacts'))
    started = time.time()
    output = build / 'evaluator/evidence' / str(int(started))
    output.mkdir(parents=True, exist_ok=True)
    identities = {str(path.relative_to(build)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for folder in ('starter-overlay', 'solution', 'evaluator/utils', 'evaluator/verifiers')
                  for path in (build / folder).rglob('*') if path.is_file() and path.suffix in ('.py', '.sh')}
    (output / 'code_identity.json').write_text(json.dumps(identities, indent=2))
    sandbox = None
    process = None
    app = modal.App('hotpot-bert-tiny-candidate-qualification')
    with modal.enable_output(), app.run():
        try:
            sandbox = modal.Sandbox.create('sleep', 'infinity', image=image, app=app,
                                           gpu='H200:2', cpu=32, memory=262144, timeout=14400,
                                           block_network=True)
            (output / 'sandbox.json').write_text(json.dumps(dict(id=sandbox.object_id, started=started,
                                                                gpu='H200:2', block_network=True,
                                                                mode=args.mode, image=image.object_id), indent=2))
            process = sandbox.exec('python', '/qualification.py')
            with (output / 'qualification.log').open('w') as stream:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
            process.wait()
            (output / 'qualification.stderr').write_text(process.stderr.read())
            print('Qualification exit:', process.returncode, flush=True)
            if process.returncode:
                raise RuntimeError(f'Qualification infrastructure/workload failed: exit {process.returncode}')
        finally:
            if sandbox is not None:
                try:
                    if process is not None:
                        process = sandbox.exec('tar', 'czf', '/reports.tar.gz', '-C', '/qualification-output', '.')
                        process.wait()
                        if process.returncode == 0:
                            encoded = sandbox.exec('base64', '-w0', '/reports.tar.gz').stdout.read()
                            (output / 'reports.tar.gz').write_bytes(base64.b64decode(encoded))
                finally:
                    sandbox.terminate()
                    (output / 'cleanup.json').write_text(json.dumps(dict(id=sandbox.object_id, terminated=True,
                                                                         wall_seconds=time.time() - started), indent=2))


if __name__ == '__main__':
    main()
