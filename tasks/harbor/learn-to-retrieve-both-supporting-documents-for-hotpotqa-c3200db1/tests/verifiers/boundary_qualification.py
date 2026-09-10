"""Final fixed-code integration run; private results must not guide development."""
import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    workspace = Path('/environment/starter')
    output = Path('/qualification-output')
    output.mkdir(exist_ok=True)
    checks = {}
    starter_source = (workspace / 'training.py').read_bytes()
    (output / 'hardware.txt').write_text(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv'], text=True))

    def execute(name, command, cwd=workspace, public=False):
        options = dict(user=65534, group=65534, extra_groups=[]) if public else {}
        with (output / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=3500, **options)
        checks[name] = result.returncode
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see its log')

    execute('self_test', ['bash', '-c', "python3 -I verifiers/prepare_launcher.py && "
                         "python3 -m unittest discover -s verifiers -p 'test_*.py' -v"], Path('/tests'))
    probe = subprocess.run([sys.executable, '-I', '-c',
                            'from pathlib import Path; Path("/tests/private_panel/gold.jsonl").read_text()'],
                           user=65534, group=65534, extra_groups=[], capture_output=True, text=True)
    assert probe.returncode != 0 and 'PermissionError' in probe.stderr
    checks['private_labels_denied'] = True
    manual = workspace / 'runs/manual'
    manual.mkdir(parents=True)
    os.chown(workspace / 'runs', 65534, 65534)
    os.chown(manual, 65534, 65534)
    execute('control_weights', ['bash', str(workspace / 'train_retriever.sh'), '--recipe', 'control',
                               '--train_dir', str(workspace / 'optifine_public_tests/assets/hotpotqa/train'),
                               '--models_dir', str(workspace / 'optifine_public_tests/assets/models'),
                               '--output_dir', str(manual), '--seed', '1729'], public=True)
    tool = str(workspace / 'optifine_public_tests/run.py')
    execute('public_help', ['python', tool, '--help'], public=True)
    execute('check_weights', ['python', tool, '--check-weights', str(manual / 'model.safetensors'),
                             '--output', str(workspace / 'runs/validity')], public=True)
    execute('smoke', ['python', tool, '--submission', str(workspace / 'train_retriever.sh'),
                     '--output', str(workspace / 'runs/smoke'), '--smoke'], public=True)
    execute('public_candidate_only', ['python', tool, '--submission', str(workspace / 'train_retriever.sh'),
                                     '--output', str(workspace / 'runs/public')], public=True)
    execute('install_reference', ['bash', '/reference/solve.sh'])
    execute('public_paired', ['python', tool, '--submission', str(workspace / 'train_retriever.sh'),
                             '--output', str(workspace / 'runs/paired'), '--paired'], public=True)
    execute('private_paired', ['bash', '/tests/test.sh'])
    shutil.copytree(workspace / 'runs/paired', output / 'public_paired')
    shutil.copytree(workspace / 'runs/public', output / 'public_candidate_only')
    shutil.copytree('/logs/verifier', output / 'private_paired')
    public = json.loads((output / 'public_paired/score.json').read_text())
    private = json.loads((output / 'private_paired/reward.json').read_text())
    assert public['valid'] == private['valid'] == 1
    assert all(type(value) in (int, float) for value in private.values())
    assert public['R'] == (public['C'] - 0.446) / (1 - 0.446)
    assert private['reward'] == (private['ndcg_at10'] - 0.446) / (1 - 0.446)
    for name in ('validity', 'smoke', 'public', 'paired'):
        metadata = json.loads((workspace / 'runs' / name / 'checkpoint.json').read_text())
        assert metadata['status'] == 'finished'
        artifact = metadata['artifacts']['model.safetensors']
        assert hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest() == artifact['sha256']['model.safetensors']
        destination = output / ('restored-' + name)
        execute('restore_' + name, ['bash', '/logs/artifacts/progress/checkpoint.sh', '--restore',
                                    metadata['source_checkpoint']['path'], '--destination', str(destination)])
        assert (destination / 'training.py').is_file()
        if name != 'paired':
            assert (destination / 'training.py').read_bytes() == starter_source
        assert not (destination / 'optifine_public_tests').exists()
        assert not (destination / 'runs').exists()
    assert len(list(Path('/logs/artifacts/public-verifier').iterdir())) == 4
    checks['public_source_and_weights_retained'] = True
    shutil.copytree('/logs/artifacts', output / 'public-artifacts')
    (output / 'boundary.json').write_text(json.dumps(dict(checks=checks, public=public, private=private), indent=2))
    print(json.dumps(dict(checks=checks, public=public, private=private)), flush=True)


if __name__ == '__main__':
    main()
