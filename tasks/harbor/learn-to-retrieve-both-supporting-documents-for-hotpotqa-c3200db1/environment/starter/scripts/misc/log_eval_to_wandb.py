"""Publish a one-shot eval's results: metrics + samples artifact into the TRAINING wandb run,
and the raw JSON to GCS beside the checkpoints.

Why this exists. `eval.py` writes its results JSON to LOCAL disk and opens its own wandb run in
`memory-layers-eval` — but that run carries only the config, never the numbers, because
`llm_judge_accuracy` / `lexical_grounding` are computed by `evals/shared.py::run_metrics_pipeline`
in eval.py's PARENT process, after the JAX worker that owns the wandb run has exited.
`scripts/misc/hard_neg_eval_box.py` normally closes that loop for watcher-style evals; a one-shot
eval bypasses it and the numbers exist nowhere durable. On a flex-start VM that is a hard
deadline — the box deletes itself, boot disk included.

Attaches to the training run the same way the eval box does: the wandb id is derived from the
checkpoint's run-dir via `utils.wandb_run_id_from_run_dir`, so wandb run and GCS folder stay 1:1
and it is impossible to write into a different launch's curve. Shared mode with
x_primary=False + x_update_finish_state=False means we are a SECONDARY writer — eval/* lands
beside train/* without our finish() ending the run.

    CKPT=gs://…/qwen3_mem_embed/<step> [EVAL_TASK=…] python scripts/misc/log_eval_to_wandb.py
"""
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import wandb
from utils import wandb_run_id_from_run_dir

CKPT = os.environ.get("CKPT") or sys.exit("set CKPT to the gs://…/qwen3_mem_embed/<step> path")
TASK = os.environ.get("EVAL_TASK", "eval")
RESULTS_GLOB = os.environ.get("RESULTS_GLOB", "outputs/**/eval_results/**/*.json")
# Explicit path wins over the glob, and may be gs://. Needed because multi-vm-tpu-setup.sh wipes
# everything but .venv/.git on every code sync — so a later launcher invocation DELETES the
# local outputs/ tree, and the GCS copy becomes the only surviving one.
RESULTS_FILE = os.environ.get("RESULTS_FILE")

# gs://<bucket>/<run_dir>/qwen3_mem_embed/<step>  ->  (run_dir, step)
m = re.match(r"(gs://[^/]+/([^/]+))/[^/]+/(\d+)/?$", CKPT.rstrip("/"))
if not m:
    sys.exit(f"could not parse a run-dir and step out of CKPT={CKPT!r}")
run_root, run_dir, step = m.group(1), m.group(2), int(m.group(3))

# Newest results file: a re-run leaves older hydra output dirs in place, and silently publishing a
# stale one would attribute the wrong numbers to this checkpoint.
if RESULTS_FILE:
    path = RESULTS_FILE
    if path.startswith("gs://"):
        local = os.path.join("/tmp", os.path.basename(path))
        subprocess.run(["gsutil", "-q", "cp", path, local], check=True)
        path = local
elif (files := sorted(glob.glob(RESULTS_GLOB, recursive=True), key=os.path.getmtime)):
    path = files[-1]
else:
    sys.exit(f"no results JSON matched {RESULTS_GLOB!r} and RESULTS_FILE unset — "
             "note a launcher code-sync wipes outputs/, so prefer the GCS copy")
data = json.load(open(path))
metrics = data.get("metrics") or {}
if not metrics:
    sys.exit(f"{path} has no 'metrics' block; refusing to log an empty eval")
print(f"[log_eval] {path}  ({len(data.get('samples', []))} samples, {len(metrics)} metrics)")

# 1. durable copy in GCS, beside the checkpoints
dst = f"{run_root}/eval/step_{step}/{TASK}.json"
if RESULTS_FILE and RESULTS_FILE.startswith("gs://"):
    print(f"[log_eval] GCS  -> already at {RESULTS_FILE}")
else:
    subprocess.run(["gsutil", "-q", "cp", path, dst], check=True)
    print(f"[log_eval] GCS  -> {dst}")

# 2. metrics + samples artifact into the TRAINING run
wid = wandb_run_id_from_run_dir(run_dir)
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "memory-layers"),
    id=wid,
    resume="allow",
    settings=wandb.Settings(mode="shared", x_primary=False, x_label="eval",
                            x_update_finish_state=False),
)
# train_step as the x-axis, matching hard_neg_eval_box.py: several tasks can report at the SAME
# checkpoint step, and wandb.log(step=...) would keep only the first and silently drop the rest.
wandb.define_metric("train_step")
wandb.define_metric("eval/*", step_metric="train_step")
payload = {f"eval/{TASK}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
payload["train_step"] = step
wandb.log(payload)
print(f"[log_eval] wandb {wid} <- {len(payload) - 1} metrics @ train_step={step}")

art = wandb.Artifact(f"eval-{TASK}-step{step}", type="eval_results",
                     metadata={"checkpoint": CKPT, "task": TASK, "step": step, **metrics})
art.add_file(path, name=f"{TASK}.json")
run.log_artifact(art)
run.finish()
print(f"[log_eval] artifact eval-{TASK}-step{step} uploaded")
for k, v in sorted(metrics.items()):
    print(f"    {k}: {v}")
