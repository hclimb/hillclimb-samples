"""Backfill all ground_eval GCS results into wandb (per-run '<run>_eval' run), using train_step
as the x-axis so every alias (msmarco/hotpotqa/musique x corpus/oracle) shows — the live logger
had dropped non-first aliases per step via wandb's monotonic-step rule. Run on the eval box (has
WANDB_API_KEY + GCS creds). Uses gcsfs (gsutil '**' glob returns nothing on the eval box).

  GCS_BUCKET=memory-layers-training .venv/bin/python scripts/misc/backfill_eval_wandb.py
"""
import json, os
import gcsfs
import wandb

bucket = os.environ.get("GCS_BUCKET", "memory-layers-training")
fs = gcsfs.GCSFileSystem()
# <bucket>/ground_eval/<run>/step<N>/<alias>.json
files = fs.glob(f"{bucket}/ground_eval/*/*/*.json")
print(f"found {len(files)} eval result files")

by_run = {}
for f in files:
    try:
        run, stepdir, aliasjson = f.split("/ground_eval/")[1].split("/")
        step = int(stepdir.replace("step", "")); alias = aliasjson.replace(".json", "")
    except Exception:
        continue
    by_run.setdefault(run, []).append((step, alias, f))

for run, items in sorted(by_run.items()):
    # Fresh id namespace (_geval): clean-slate-deleted {run}_eval ids hang on resume='allow'. Name stays.
    r = wandb.init(project="memory-layers", name=f"{run}_eval", id=f"{run}_geval",
                   resume="allow", reinit=True,
                   settings=wandb.Settings(init_timeout=300))
    wandb.define_metric("train_step")
    wandb.define_metric("eval/*", step_metric="train_step")
    n = 0
    for step, alias, f in sorted(items):
        try:
            acc = json.loads(fs.cat(f).decode())["metrics"]["llm_judge_accuracy"]
        except Exception:
            continue
        wandb.log({f"eval/{alias}/llm_judge_accuracy": acc, "train_step": step})
        n += 1
    print(f"{run}: logged {n} points")
    r.finish()
print("BACKFILL DONE")
