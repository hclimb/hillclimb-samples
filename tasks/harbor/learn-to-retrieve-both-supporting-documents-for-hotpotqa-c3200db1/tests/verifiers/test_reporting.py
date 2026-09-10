import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.process import PhaseFailure
from verifiers import verify
from verifiers.runner import run, run_one


class ReportingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires the compiled tensor libraries')
    def test_public_command_from_public_only_file_view(self):
        from safetensors.torch import save_file
        from transformers import BertModel

        public = Path(__file__).resolve().parents[1] / 'public_tests'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            starter = root / 'environment/starter'
            tools = starter / 'optifine_public_tests'
            shutil.copytree(public, tools)
            # This checks public startup and tensor loading without requiring the
            # runtime's /logs mount. The Linux boundary probe tests real snapshots.
            helper = tools / 'utils/public_checkpoint.py'
            helper.write_text(helper.read_text() + '\nARCHIVE_ROOT = Path(' + repr(str(root / 'archive')) +
                              ')\nPublicRun.checkpoint = lambda self, source: None\n')
            self.assertEqual({path.name for path in tools.iterdir()},
                             {'run.py', 'assets', 'public_panel', 'incumbent', 'utils', 'verifiers'})
            self.assertEqual({path.name for path in (tools / 'assets').iterdir()}, {'models', 'hotpotqa'})
            model = BertModel.from_pretrained(tools / 'assets/models/bert-tiny',
                                              add_pooling_layer=False, local_files_only=True)
            checkpoint = starter / 'runs/manual/model.safetensors'
            checkpoint.parent.mkdir(parents=True)
            save_file(model.state_dict(), str(checkpoint))
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1')
            command = [sys.executable, str(tools / 'run.py')]
            help_result = subprocess.run(command + ['--help'], cwd=starter, env=env,
                                         capture_output=True, text=True, timeout=30)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            output = starter / 'runs/validity'
            result = subprocess.run(command + ['--check-weights', str(checkpoint), '--output', str(output)],
                                    cwd=starter, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((output / 'validity.json').read_text())['valid'], 1)
            self.assertFalse((output / 'reward.json').exists())
            saved = json.loads((output / 'checkpoint.json').read_text())
            retained = Path(saved['artifacts']['model.safetensors']['path'])
            self.assertEqual(retained.read_bytes(), checkpoint.read_bytes())
            self.assertEqual(saved['status'], 'finished')

    def test_launcher_isolation_and_argument_forwarding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'verifiers').mkdir()
            shutil.copyfile(Path(__file__).with_name('launch.sh'), root / 'verifiers/launch.sh')
            (root / 'verifiers/verify.py').write_text(
                'import sys, json\nfrom pathlib import Path\n'
                'assert sys.flags.isolated\n'
                '(Path(sys.argv[1]) / "startup.json").write_text(json.dumps(sys.argv[2:]))\n')
            hostile = root / 'hostile'
            hostile.mkdir()
            (hostile / 'sitecustomize.py').write_text('raise RuntimeError("untrusted startup")')
            (root / 'python').symlink_to(sys.executable)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], PYTHONPATH=str(hostile))
            result = subprocess.run(['bash', str(root / 'verifiers/launch.sh'), str(root), '--paired'],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / 'startup.json').read_text()), ['--paired'])
            (root / 'verifiers/verify.py').write_text('raise RuntimeError("infrastructure unavailable")\n')
            result = subprocess.run(['bash', str(root / 'verifiers/launch.sh'), str(root)],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('infrastructure unavailable', result.stderr)
            self.assertFalse((root / 'reward.json').exists())

    def test_unisolated_entrypoint_fails_before_running_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run([sys.executable, str(Path(verify.__file__).resolve()),
                                     '--output', str(root)],
                                    cwd=root, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Use verifiers/launch.sh with Linux root isolation', result.stderr)
            self.assertFalse((root / 'reward.json').exists())
            self.assertFalse((root / 'candidate').exists())

    def test_verifier_numeric_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for result in (dict(valid=1, R=-0.5, B=0.5, C=0.169), dict(valid=0, R=0, B=0.5, C=0)):
                with self.subTest(valid=result['valid']), \
                        patch.object(sys, 'argv', ['verify', '--output', str(root)]), \
                        patch.object(sys, 'flags') as flags, patch('os.geteuid', return_value=0), \
                        patch.object(Path, 'chmod'), patch('verifiers.verify.shutil.copytree'), \
                        patch('verifiers.runner.run', return_value=result) as run:
                    flags.isolated = True
                    verify.main()
                self.assertEqual(json.loads((root / 'reward.json').read_text()),
                                 dict(valid=result['valid'], reward=result['R'],
                                      baseline_ndcg=result['B'], ndcg_at10=result['C']))
                trusted = Path(__file__).resolve().parents[1]
                self.assertEqual(run.call_args.args[3], trusted / 'private_panel')
                self.assertEqual(run.call_args.args[5], trusted / 'public_tests')
                self.assertEqual(run.call_args.kwargs, dict(paired=True))

    def test_trusted_failure_is_not_submission_invalidity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def training(command, *args):
                artifacts = Path(command[command.index('--output_dir') + 1])
                (artifacts / 'model.safetensors').write_bytes(b'checked in child')
                (artifacts.parent / 'training.log').write_text('training done')
                return dict(seconds=0.1)

            for exit_code, timed_out, invalid in [(1, False, False), (2, False, True), (-9, True, True)]:
                failure = PhaseFailure(dict(exit_code=exit_code, timeout=timed_out,
                                            seconds=0.1, output_excerpt='underlying failure'))
                with self.subTest(exit_code=exit_code), patch('verifiers.runner.phase') as phase:
                    phase.side_effect = lambda *args: training(*args) if phase.call_count == 1 else self.fail_phase(failure)
                    arguments = (root, Path('train.sh'), root, root, root / 'output', root)
                    if invalid:
                        result = run_one(*arguments)
                        self.assertEqual(result['valid'], 0)
                        self.assertEqual(result['failure']['output_excerpt'], 'underlying failure')
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'underlying failure'):
                            run_one(*arguments)

    @staticmethod
    def fail_phase(failure):
        raise failure

    def test_snapshot_io_error_propagates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('verifiers.runner.snapshot', side_effect=OSError('disk unavailable')):
                with self.assertRaisesRegex(OSError, 'disk unavailable'):
                    run(root, root / 'train.sh', root, root, root / 'output', root)
            self.assertFalse((root / 'output/score.json').exists())

    def test_environment_failure_precedes_candidate_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure = PhaseFailure(dict(exit_code=1, timeout=False, output_excerpt='Two CUDA GPUs are required'))
            with patch('verifiers.runner.snapshot'), \
                    patch('verifiers.runner.check_environment', side_effect=failure), \
                    patch('verifiers.runner.run_one') as candidate:
                with self.assertRaisesRegex(PhaseFailure, 'Two CUDA GPUs'):
                    run(root, root / 'train.sh', root, root, root / 'output', root)
                candidate.assert_not_called()
            self.assertFalse((root / 'output/score.json').exists())

    def test_verifier_diagnostics_without_zero_reward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'reward.json').write_text('{"valid": 1, "reward": 1}')
            with patch.object(sys, 'argv', ['verify', '--output', str(root)]), \
                    patch.object(sys, 'flags') as flags, patch('os.geteuid', return_value=0), \
                    patch.object(Path, 'chmod'), patch('verifiers.verify.shutil.copytree'), \
                    patch('verifiers.runner.run', side_effect=RuntimeError('CUDA unavailable')):
                flags.isolated = True
                with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                    verify.main()
            self.assertFalse((root / 'reward.json').exists())
            self.assertEqual(json.loads((root / 'diagnostics.json').read_text()),
                             dict(evaluator_error='CUDA unavailable'))
