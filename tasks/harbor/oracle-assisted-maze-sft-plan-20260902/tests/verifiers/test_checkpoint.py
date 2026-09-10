import fcntl
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        helper = ROOT.parent / "environment/checkpoint.sh"
        if not helper.is_file():
            self.skipTest("task checkpoint script is not available")
        for command in ("bash", "rsync", "zstd", "flock"):
            if shutil.which(command) is None:
                self.skipTest(f"{command} is not available")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        (self.source / "maze_task").mkdir(parents=True)
        (self.source / "maze_task/candidate.py").write_text("original")
        self.progress = self.root / "progress"
        self.cache = self.root / "cache"
        self.cache.mkdir()
        script = helper.read_text()
        for name, path in (("SOURCE_ROOT", self.source),
                           ("PROGRESS_ROOT", self.progress),
                           ("CACHE_ROOT", self.cache)):
            original = next(line for line in script.splitlines()
                            if line.startswith(f"readonly {name}="))
            script = script.replace(original, f"readonly {name}={shlex.quote(str(path))}")
        self.helper = self.root / "checkpoint.sh"
        self.helper.write_text(script)
        self.helper.chmod(0o755)
        self.env = {"PATH": f"{self.root}:{os.environ['PATH']}"}

    def _run(self, *arguments):
        return subprocess.run([str(self.helper), *arguments], env=self.env,
                              text=True, capture_output=True, timeout=20)

    def _metadata(self):
        return sorted((dict(line.split("=", 1) for line in path.read_text().splitlines())
                       for path in self.progress.glob("*/metadata.txt")),
                      key=lambda row: int(row["sequence"]))

    def test_concurrent_saves_keep_one_chain_and_restore_every_snapshot(self):
        # Pause the first save after parent selection to force an overlap.
        ready = self.root / "ready"
        release = self.root / "release"
        rsync = self.root / "rsync"
        rsync.write_text(
            "#!/bin/sh\nset -eu\n"
            f"if mkdir {shlex.quote(str(self.root / 'first-save'))} 2>/dev/null; then\n"
            f"  touch {shlex.quote(str(ready))}\n"
            f"  while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.01; done\n"
            "fi\n"
            f"exec {shlex.quote(shutil.which('rsync'))} \"$@\"\n"
        )
        rsync.chmod(0o755)
        processes = []
        try:
            processes.append(subprocess.Popen(
                [str(self.helper), "--label", "parallel-0"], env=self.env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ))
            deadline = time.monotonic() + 10
            while not ready.exists():
                if processes[0].poll() is not None:
                    self.fail(str(processes[0].communicate()))
                self.assertLess(time.monotonic(), deadline, "first save did not reach rsync")
                time.sleep(0.01)
            descriptor = os.open(self.cache / "save.lock", os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
            for index in range(1, 4):
                processes.append(subprocess.Popen(
                    [str(self.helper), "--label", f"parallel-{index}"], env=self.env,
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ))
            release.touch()
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, stdout + stderr)
                self.assertEqual(stdout.count("Checkpoint saved:"), 1)
        finally:
            release.touch()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=20)

        rows = self._metadata()
        self.assertEqual([int(row["sequence"]) for row in rows], list(range(4)))
        self.assertEqual(len({row["checkpoint_id"] for row in rows}), 4)
        for index, row in enumerate(rows):
            self.assertEqual(row["kind"], "patch" if index else "keyframe")
            self.assertEqual(row["parent_checkpoint_id"],
                             rows[index - 1]["checkpoint_id"] if index else "")
            self.assertEqual(row["parent_snapshot_sha256"],
                             rows[index - 1]["snapshot_sha256"] if index else "")
            restored = self.root / f"restored-{index}"
            result = self._run("--restore", row["checkpoint_id"], "--destination", str(restored))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((restored / "maze_task/candidate.py").read_text(), "original")
        self.assertEqual((self.cache / "checkpoint_id").read_text().strip(),
                         rows[-1]["checkpoint_id"])

    def test_failed_save_releases_lock(self):
        rsync = self.root / "rsync"
        rsync.write_text("#!/bin/sh\nexit 7\n")
        rsync.chmod(0o755)
        failed = self._run("--label", "failed")
        self.assertEqual(failed.returncode, 7, failed.stdout + failed.stderr)
        self.assertEqual(self._metadata(), [])
        rsync.unlink()
        retry = self._run("--label", "retry")
        self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
        self.assertEqual(len(self._metadata()), 1)

    def test_restore_does_not_wait_for_save_lock(self):
        saved = self._run("--label", "restore-input")
        self.assertEqual(saved.returncode, 0, saved.stdout + saved.stderr)
        checkpoint = self._metadata()[0]["checkpoint_id"]
        descriptor = os.open(self.cache / "save.lock", os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            restored = self.root / "restored"
            result = self._run("--restore", checkpoint, "--destination", str(restored))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((restored / "maze_task/candidate.py").read_text(), "original")
        finally:
            os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
