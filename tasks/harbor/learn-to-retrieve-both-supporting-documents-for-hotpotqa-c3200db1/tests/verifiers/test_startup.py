import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from verifiers.prepare_launcher import prepare


class CompiledStartupTests(unittest.TestCase):
    def test_adapter_is_idempotent_and_rejects_unknown_launchers(self):
        with tempfile.TemporaryDirectory() as temporary:
            launcher = Path(temporary) / 'test.sh'
            launcher.write_text('#!/bin/bash\npython /tests/verifiers/verify.py\n')
            prepare(launcher)
            prepared = launcher.read_bytes()
            prepare(launcher)
            self.assertEqual(launcher.read_bytes(), prepared)
            launcher.write_text('python /tests/unknown.py\n')
            with self.assertRaisesRegex(RuntimeError, 'Unsupported generated verifier launcher'):
                prepare(launcher)
            self.assertEqual(launcher.read_text(), 'python /tests/unknown.py\n')

    def test_generated_launcher_reaches_numeric_writer_without_candidate_imports(self):
        trusted = Path(__file__).resolve().parents[1]
        launcher = trusted / 'test.sh'
        self.assertTrue(launcher.is_file(), 'Run the declared self-test in the compiled evaluator tree')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tests = root / 'tests'
            starter = root / 'environment/starter'
            output = root / 'logs/verifier'
            for directory in (tests / 'verifiers', tests / 'public_tests/assets', starter, output):
                directory.mkdir(parents=True)
            source = (trusted / 'verifiers/verify.py').read_text()
            source = source.replace('/environment/starter', str(starter)).replace('/logs/verifier', str(output))
            if os.geteuid() != 0:
                source = 'import os\nos.geteuid = lambda: 0\n' + source
            (tests / 'verifiers/verify.py').write_text(source)
            (tests / 'verifiers/runner.py').write_text(
                'import os\nimport sys\n'
                'def run(workspace, submission, assets, panel, output, trusted, paired):\n'
                '    assert sys.flags.isolated\n'
                '    assert paired and panel.name == "private_panel"\n'
                '    assert trusted.name == "public_tests" and assets.is_dir()\n'
                '    assert submission == workspace / "train_retriever.sh"\n'
                '    (output / "workload-reached").touch()\n'
                '    if os.environ["STARTUP_CASE"] == "infrastructure":\n'
                '        raise RuntimeError("infrastructure unavailable")\n'
                '    if os.environ["STARTUP_CASE"] == "invalid":\n'
                '        return dict(valid=0, R=0, B=0.5, C=0)\n'
                '    return dict(valid=1, R=-0.5, B=0.5, C=0.169)\n')
            locations = {'/environment/starter': str(starter), '/tests': str(tests)}
            relocated = re.sub('/environment/starter|/tests', lambda match: locations[match[0]], launcher.read_text())
            (tests / 'test.sh').write_text(relocated)
            binaries = root / 'bin'
            binaries.mkdir()
            (binaries / 'python').symlink_to(sys.executable)
            marker = root / 'candidate-imported'
            hostile = f'import os\nopen({str(marker)!r}, "w").close()\nos._exit(91)\n'
            for filename in ('sitecustomize.py', 'usercustomize.py', 'json.py'):
                (starter / filename).write_text(hostile)
            env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'],
                       PYTHONPATH=str(starter), PYTHONHOME=str(root / 'invalid-python-home'))
            for scenario in ('valid', 'invalid', 'infrastructure'):
                with self.subTest(scenario=scenario):
                    shutil.rmtree(output)
                    output.mkdir()
                    (output / 'reward.json').write_text('{"valid": 1, "reward": 99}')
                    result = subprocess.run(['bash', str(tests / 'test.sh')], cwd=starter,
                                            env=dict(env, STARTUP_CASE=scenario),
                                            capture_output=True, text=True, timeout=10)
                    self.assertFalse(marker.exists(), result.stderr)
                    self.assertTrue((output / 'workload-reached').exists(), result.stderr)
                    if scenario == 'infrastructure':
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('infrastructure unavailable', result.stderr)
                        self.assertFalse((output / 'reward.json').exists())
                        self.assertEqual(json.loads((output / 'diagnostics.json').read_text()),
                                         dict(evaluator_error='infrastructure unavailable'))
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        expected = (dict(valid=1, reward=-0.5, baseline_ndcg=0.5, ndcg_at10=0.169)
                                    if scenario == 'valid' else
                                    dict(valid=0, reward=0, baseline_ndcg=0.5, ndcg_at10=0))
                        self.assertEqual(json.loads((output / 'reward.json').read_text()), expected)
