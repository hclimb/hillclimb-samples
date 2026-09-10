"""Find the latest checkpoint of a run across all of its run-dir generations.

Unlike find_latest_ckpt.py (which emits dir/step = warm-start), this prints
    "<best_step> <n_run_dirs> <ckpt_dir>"
where ckpt_dir has NO step suffix -> `+trainer.resume_from=<ckpt_dir>` triggers a
FULL resume (optimizer state + step + stage) of the latest step in that dir.
n_run_dirs counts restarts (each relaunch creates a new {run_name}-{date}-{time}/
dir), used to derive a fresh dataloader shuffle seed per restart.

Prints "-1 0 " if no checkpoint exists yet.

Usage: python scripts/misc/sim_find_latest.py <run_name>
Env:   GCS_BUCKET, GCS_BUCKET_PROJECT, GOOGLE_APPLICATION_CREDENTIALS
"""
import os, re, sys
from google.cloud import storage

run_name = sys.argv[1]
bucket = os.environ["GCS_BUCKET"]
project = os.environ.get("GCS_BUCKET_PROJECT")
client = storage.Client(project=project) if project else storage.Client()

prefix = f"{run_name}-"
best_dir, best_step = None, -1
run_dirs = set()
for blob in client.list_blobs(bucket, prefix=prefix):
    parts = blob.name.split("/")
    if len(parts) < 3:
        continue
    run_dir, model, step = parts[0], parts[1], parts[2]
    # Guard: run_name must be followed by a -DATE-TIME suffix only (avoid
    # matching other runs that share this run_name as a prefix).
    suffix = run_dir[len(run_name):]
    if not re.fullmatch(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", suffix):
        continue
    run_dirs.add(run_dir)
    if not re.fullmatch(r"\d+", step):
        continue
    step = int(step)
    if step > best_step:
        best_step, best_dir = step, f"gs://{bucket}/{run_dir}/{model}"

print(f"{best_step} {len(run_dirs)} {best_dir or ''}")
