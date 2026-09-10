#!/bin/bash
# Runs ON a box. Pulls the training run's wandb history around the stage-3 boundary to locate the
# NaN source: is the loss diverging (LR too high) or is a specific aux loss going non-finite while
# total_loss stays healthy (a backward-only pathology)?
#   WID=<wandb run id> bash scripts/misc/wandb_nan_probe.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a
. .venv/bin/activate 2>/dev/null || true

WID="${WID:?set WID=<wandb run id>}"
PROJ="${WANDB_PROJECT:-memory-layers}"

WID="$WID" PROJ="$PROJ" python - <<'PY'
import os
import wandb

api = wandb.Api()
run = api.run(f"{os.environ.get('WANDB_ENTITY') or api.default_entity}/{os.environ['PROJ']}/{os.environ['WID']}")
print(f"run: {run.name}  state={run.state}")

keys = ["_step", "train/total_loss", "train/ce_loss", "train/ce_weight", "train/lr",
        "train/grad_norm", "train/grad_norm_main", "train/grad_norm_mem", "train/grad_norm_embed",
        "train/doc_access_loss", "train/doc_access_acc", "train/grad_nan_count",
        "train/loss_nan_count", "train/mem_pos_weight_mass/mean", "train/mem_top1_weight/mean"]
rows = list(run.scan_history(keys=keys, page_size=2000))
rows = [r for r in rows if r.get("_step") is not None]
rows.sort(key=lambda r: r["_step"])
print(f"history rows: {len(rows)}  step range: {rows[0]['_step']}..{rows[-1]['_step']}" if rows else "no rows")

def fmt(v):
    if v is None: return "    -   "
    if isinstance(v, float):
        return "   nan  " if v != v else f"{v:8.4g}"
    return f"{v:>8}"

hdr = ["step", "total_loss", "ce_loss", "lr", "grad_norm", "gn_main", "gn_mem", "gn_embed",
       "doc_acc_loss", "doc_acc", "grad_nan"]
print("  ".join(f"{h:>11s}" for h in hdr))
for r in rows:
    s = r["_step"]
    # NOTE: wandb's _step is its own log-call counter (it ignores our explicit step= under
    # shared mode), so with trainer.log_interval=10 the TRAINING step is ~10x _step. Convert.
    tstep = s * 10
    if not (14900 <= tstep <= 15400):
        continue
    vals = [tstep, r.get("train/total_loss"), r.get("train/ce_loss"), r.get("train/lr"),
            r.get("train/grad_norm"), r.get("train/grad_norm_main"), r.get("train/grad_norm_mem"),
            r.get("train/grad_norm_embed"), r.get("train/doc_access_loss"),
            r.get("train/doc_access_acc"), r.get("train/grad_nan_count")]
    print("  ".join(f"{fmt(v):>11s}" for v in vals))
PY
