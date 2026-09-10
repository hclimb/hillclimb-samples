"""Print a hydra `+trainer.resume_from=...` arg for the latest checkpoint of a run,
or print nothing if none exists. Uses google.cloud.storage with GOOGLE_APPLICATION_CREDENTIALS
(same creds as training), so it works on a freshly-bootstrapped VM where gsutil auth isn't set up.

Usage: python find_latest_ckpt.py <run_name>   (env: GCS_BUCKET, GOOGLE_APPLICATION_CREDENTIALS)
Checkpoint layout: gs://{bucket}/{run_name}-{date}-{time}/{model_name}/{step}/
"""
import os, re, sys
from google.cloud import storage

run_name = sys.argv[1]
bucket = os.environ["GCS_BUCKET"]
project = os.environ.get("GCS_BUCKET_PROJECT")
client = storage.Client(project=project) if project else storage.Client()

# Find all "{run_name}-{date}-{time}/{model}/{step}/" prefixes and pick the max step.
prefix = f"{run_name}-"
best_dir, best_step = None, -1
seen_dirs = {}
for blob in client.list_blobs(bucket, prefix=prefix):
    # path: {run_name}-DATE-TIME/{model}/{step}/...
    parts = blob.name.split("/")
    if len(parts) < 3:
        continue
    run_dir, model, step = parts[0], parts[1], parts[2]
    if not re.fullmatch(r"\d+", step):
        continue
    step = int(step)
    ckpt_dir = f"gs://{bucket}/{run_dir}/{model}"
    # track the global max step across all run dirs
    if step > best_step:
        best_step, best_dir = step, ckpt_dir

if best_dir is not None and best_step >= 0:
    # resume_from=dir/step lets setup_checkpointing parse the explicit step
    sys.stdout.write(f"+trainer.resume_from={best_dir}/{best_step}")
