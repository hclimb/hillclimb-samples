import gzip
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicDeploymentTests(unittest.TestCase):
    def _wrapper(self):
        deployed = Path("/environment/starter/maze_task/public_test.sh")
        if deployed.is_file():
            return deployed
        packaged = ROOT.parent / "environment" / "starter" / "maze_task" / "public_test.sh"
        if packaged.is_file():
            return packaged
        construction = ROOT.parent / "starter-overlay" / "maze_task" / "public_test.sh"
        return construction if construction.is_file() else None

    def test_public_bundle_is_self_contained_and_public_only(self):
        bundle = ROOT / "public_tests"
        required = (
            "verifiers/run_public.py",
            "verifiers/build_candidate.py",
            "verifiers/run_training.py",
            "utils/contract.py",
            "utils/maze.py",
            "utils/model.py",
            "utils/runner.py",
            "fixtures/manifest.json",
            "fixtures/maze_policy_init.pt",
            "fixtures/frozen_failures.json.gz",
            "fixtures/public_train.jsonl.gz",
            "fixtures/public_eval.jsonl.gz",
        )
        for relative in required:
            self.assertTrue((bundle / relative).is_file(), relative)
        self.assertFalse(any(bundle.rglob("private_*")))
        manifest = json.loads((bundle / "fixtures" / "manifest.json").read_text())
        self.assertEqual(set(manifest["partitions"]), {"public_train", "public_eval"})
        failures = json.loads(gzip.decompress(
            (bundle / "fixtures" / "frozen_failures.json.gz").read_bytes()))
        self.assertEqual(
            {int(key) for key in failures["actions_by_prompt_id"]},
            set(manifest["partitions"]["public_train"]["prompt_ids"]),
        )

        wrapper = self._wrapper()
        expected = "/environment/starter/optifine_public_tests/verifiers/run_public.py"
        if wrapper is not None:
            with tempfile.TemporaryDirectory() as temporary:
                capture = Path(temporary) / "python"
                arguments = Path(temporary) / "arguments.json"
                capture.write_text(
                    "#!/bin/sh\nprintf '%s\\n' \"$@\" | python3 -c "
                    "'import json,sys; json.dump(sys.stdin.read().splitlines(), open(sys.argv[1], \"w\"))' "
                    f"{shlex.quote(str(arguments))}\n"
                )
                capture.chmod(0o755)
                wrapper_result = subprocess.run(
                    [str(wrapper), "--help"],
                    text=True,
                    capture_output=True,
                    env={"PATH": f"{temporary}:{os.environ['PATH']}", "PYTHONNOUSERSITE": "1"},
                )
                self.assertEqual(
                    wrapper_result.returncode,
                    0,
                    wrapper_result.stdout + wrapper_result.stderr,
                )
                invoked = json.loads(arguments.read_text())
            self.assertEqual(invoked, ["-I", "-B", expected, "--help"])

        result = subprocess.run(
            [sys.executable, str(bundle / "verifiers" / "run_public.py"), "--help"],
            cwd=bundle,
            text=True,
            capture_output=True,
            env={"PATH": os.environ["PATH"], "PYTHONNOUSERSITE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--mode", result.stdout)
        self.assertIn("--validation-fold", result.stdout)

    def test_public_wrapper_tests_saved_code_and_keeps_linked_results(self):
        wrapper = self._wrapper()
        if wrapper is None:
            self.skipTest("public wrapper is not installed")
        cases = (
            ([], 0, 0, 0, False),
            (["--mode", "contract"], 0, 9, 8, False),
            (["--mode", "quick"], 0, 0, 0, False),
            (["--mode=contract"], 0, 0, 0, False),
            (["--mode=quick"], 7, 0, 0, False),
            (["--mode", "full", "--validation-fold", "1"], 0, 0, 0, True),
            (["--mode=full"], 0, 0, 0, True),
            (["--mo", "full"], 0, 0, 0, True),
            (["--m=full"], 0, 0, 0, True),
            (["--mode", "full", "--mode=quick"], 0, 0, 0, False),
            (["--mode=quick", "--mode", "full"], 0, 0, 0, True),
            (["--help"], 0, 0, 0, False),
            (["-h"], 0, 0, 0, False),
            (["--mode", "full", "--help"], 0, 0, 0, False),
            (["--mode", "full"], 7, 0, 0, True),
            (["--invalid"], 2, 0, 0, False),
            (["--mode", "full"], 0, 9, 0, True),
            (["--mode", "full"], 0, 0, 8, True),
        )
        for arguments, test_status, save_status, restore_status, should_save in cases:
            with self.subTest(arguments=arguments, test_status=test_status,
                              save_status=save_status, restore_status=restore_status), \
                    tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                events = root / "events"
                saved_arguments = root / "saved-arguments"
                live = root / "live" / "maze_task" / "candidate.py"
                live.parent.mkdir(parents=True)
                live.write_text("original")
                checkpoint_dir = root / "progress" / "checkpoint"
                preamble = "#!/bin/sh\nset -eu\ncase_root=" + shlex.quote(str(root)) + "\n"
                python = root / "python"
                python.write_text(
                    preamble
                    + 'printf "%s\\n" test >> "$case_root/events"\n'
                    + 'printf "%s\\n" "$@" > "$case_root/test-arguments"\n'
                    + 'for argument in "$@"; do\n'
                    + '  case "$argument" in --help|-h) printf "{}\\n"; exit 0 ;; esac\n'
                    + 'done\n'
                    + 'source_root=${STARTER_ROOT:-"$case_root/live"}\n'
                    + 'printf "%s\\n" "$source_root" > "$case_root/test-root"\n'
                    + 'before=$(cat "$source_root/maze_task/candidate.py")\n'
                    # Simulate an agent edit between two public panels.
                    + 'printf modified > "$case_root/live/maze_task/candidate.py"\n'
                    + 'after=$(cat "$source_root/maze_task/candidate.py")\n'
                    + 'printf \'{"before":"%s","after":"%s"}\\n\' "$before" "$after"\n'
                    + 'printf "public diagnostic\\n" >&2\n'
                    f"exit {test_status}\n"
                )
                checkpoint = root / "checkpoint.sh"
                checkpoint.write_text(
                    preamble
                    + 'checkpoint="$case_root/progress/checkpoint"\n'
                    + 'if [ "$1" = --restore ]; then\n'
                    + '  printf "%s\\n" restore >> "$case_root/events"\n'
                    + f'  [ {restore_status} -eq 0 ] || exit {restore_status}\n'
                    + '  [ "$2" = "$checkpoint" ] && [ "$3" = --destination ]\n'
                    + '  cp -R "$checkpoint/source/." "$4/"\n'
                    + '  printf "Checkpoint restored: %s\\n" "$checkpoint"\n'
                    + '  exit 0\n'
                    + 'fi\n'
                    + 'printf "%s\\n" save >> "$case_root/events"\n'
                    + 'printf "%s\\n" "$@" > "$case_root/saved-arguments"\n'
                    + f'[ {save_status} -eq 0 ] || exit {save_status}\n'
                    + 'mkdir -p "$checkpoint"\n'
                    + 'cp -R "$case_root/live" "$checkpoint/source"\n'
                    + 'printf "checkpoint_id=checkpoint\\nsnapshot_sha256=fixture\\n" > "$checkpoint/metadata.txt"\n'
                    + 'printf "Checkpoint saved: %s\\n" "$checkpoint"\n'
                    + 'printf "Restore with: checkpoint.sh --restore checkpoint --destination final_app\\n"\n'
                )
                python.chmod(0o755)
                checkpoint.chmod(0o755)
                result = subprocess.run(
                    [str(wrapper), *arguments], text=True, capture_output=True,
                    env={"PATH": f"{temporary}:{os.environ['PATH']}"},
                )
                expected_status = (save_status or restore_status or test_status
                                   if should_save else test_status)
                self.assertEqual(result.returncode, expected_status,
                                 result.stdout + result.stderr)
                help_request = any(argument in ("--help", "-h") for argument in arguments)
                expected_events = ["save"] if should_save else ["test"]
                if should_save and not save_status:
                    expected_events.append("restore")
                    if not restore_status:
                        expected_events.append("test")
                self.assertEqual(events.read_text().splitlines(), expected_events)
                if should_save:
                    self.assertEqual(saved_arguments.read_text().splitlines(), [
                        "--label", "public-test", "--note",
                        "Public test input: " + " ".join(arguments),
                    ])
                else:
                    self.assertFalse(saved_arguments.exists())
                    self.assertFalse(checkpoint_dir.exists())
                    self.assertFalse((root / "public-tests").exists())
                    self.assertEqual(json.loads(result.stdout), {} if help_request else
                                     {"before": "original", "after": "modified"})
                    self.assertEqual(live.read_text(), "original" if help_request else "modified")
                    continue
                if help_request or save_status or restore_status:
                    self.assertEqual(live.read_text(), "original")
                    continue
                expected_result = {"before": "original", "after": "original"}
                self.assertEqual(json.loads(result.stdout), expected_result)
                self.assertEqual(live.read_text(), "modified")
                self.assertEqual((checkpoint_dir / "source/maze_task/candidate.py").read_text(),
                                 "original")
                self.assertFalse(Path((root / "test-root").read_text().strip()).exists())
                records = root / "public-tests" / checkpoint_dir.name
                self.assertEqual(json.loads((records / "result.json").read_text()), expected_result)
                self.assertEqual((records / "exit-code.txt").read_text(), f"{test_status}\n")
                self.assertEqual((records / "stderr.log").read_text(), "public diagnostic\n")
                self.assertEqual((records / "checkpoint.txt").read_bytes(),
                                 (checkpoint_dir / "metadata.txt").read_bytes())
                self.assertEqual((records / "arguments.txt").read_text().splitlines(),
                                 arguments)
                self.assertEqual((root / "test-arguments").read_text().splitlines(), [
                    "-I", "-B", "/environment/starter/optifine_public_tests/verifiers/run_public.py",
                    *arguments,
                ])
                self.assertIn("Checkpoint saved:", result.stderr)
                self.assertIn("public diagnostic", result.stderr)

    def test_public_wrapper_uses_a_real_verified_checkpoint(self):
        wrapper = self._wrapper()
        helper = ROOT.parent / "environment/checkpoint.sh"
        if wrapper is None or not helper.is_file():
            self.skipTest("task checkpoint script is not available")
        for command in ("bash", "zstd", "rsync", "flock"):
            if shutil.which(command) is None:
                self.skipTest(f"{command} is not available")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / "live/maze_task/candidate.py"
            live.parent.mkdir(parents=True)
            live.write_text("original")
            script = helper.read_text()
            for name, path in (("SOURCE_ROOT", live.parent.parent),
                               ("PROGRESS_ROOT", root / "progress"),
                               ("CACHE_ROOT", root / "cache")):
                original = next(line for line in script.splitlines()
                                if line.startswith(f"readonly {name}="))
                script = script.replace(original, f"readonly {name}={shlex.quote(str(path))}")
            checkpoint = root / "checkpoint.sh"
            checkpoint.write_text(script)
            checkpoint.chmod(0o755)
            python = root / "python"
            python.write_text(
                "#!/bin/sh\nset -eu\n"
                'before=$(cat "$STARTER_ROOT/maze_task/candidate.py")\n'
                f"printf modified > {shlex.quote(str(live))}\n"
                'after=$(cat "$STARTER_ROOT/maze_task/candidate.py")\n'
                'printf \'{"before":"%s","after":"%s"}\\n\' "$before" "$after"\n'
            )
            python.chmod(0o755)
            result = subprocess.run(
                [str(wrapper), "--mode", "full"], text=True, capture_output=True,
                env={"PATH": f"{temporary}:{os.environ['PATH']}"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout),
                             {"before": "original", "after": "original"})
            self.assertEqual(live.read_text(), "modified")
            records = list((root / "public-tests").glob("*"))
            self.assertEqual(len(records), 1)
            saved = root / "progress" / records[0].name
            self.assertEqual((records[0] / "checkpoint.txt").read_bytes(),
                             (saved / "metadata.txt").read_bytes())
            restored = root / "restored"
            restore = subprocess.run(
                [str(checkpoint), "--restore", str(saved), "--destination", str(restored)],
                text=True, capture_output=True,
            )
            self.assertEqual(restore.returncode, 0, restore.stdout + restore.stderr)
            self.assertEqual((restored / "maze_task/candidate.py").read_text(), "original")

    def test_public_runtime_sources_match_the_verifier_and_deployed_bundle(self):
        bundle = ROOT / "public_tests"
        deployed = ROOT.parent / "environment" / "starter" / "optifine_public_tests"
        for relative in ("utils/maze.py", "utils/contract.py", "utils/model.py", "utils/runner.py",
                         "verifiers/build_candidate.py", "verifiers/run_public.py",
                         "verifiers/run_training.py"):
            with self.subTest(relative=relative):
                expected = (ROOT / relative).read_text()
                if relative == "utils/maze.py":
                    expected = expected.replace(
                        'set(partition_ids("public_train")) | set(partition_ids("private_train"))',
                        'set(partition_ids("public_train"))',
                    ).replace(
                        '("initialization", "public_train", "public_eval", "private_train", "private_eval")',
                        'fixture_manifest()["partitions"]',
                    )
                self.assertEqual((bundle / relative).read_text(), expected)
                if deployed.is_dir():
                    self.assertEqual((deployed / relative).read_text(), expected)


if __name__ == "__main__":
    unittest.main()
