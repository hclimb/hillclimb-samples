"""Log the manual 4-doc msmarco-oracle (g4eval) results into each training run's
'<run>_eval' wandb run, matching ground_eval_box.py's convention (name=<run>_eval,
id=<run>_geval fresh namespace, metric eval/<alias>/llm_judge_accuracy vs train_step).

The g4eval results live in local /tmp/g4eval_<tag>/eval_results/step_<N>/msmarco_g4/outputs/*.json
on the eval box (they were run with use_wandb=false). Tag encodes run + step, e.g.
  g4eval_da01_40k_n256  -> run ground_msmarco_iso_grp4_ft,      step 40000
  g4eval_da0_30k_n256   -> run ground_msmarco_iso_grp4_da0_ft,  step 30000
  g4eval_da003_20k_n256 -> run ground_msmarco_iso_grp4_da003_ft, step 20000

Per (run,step) the highest-sample-count (n256>n128>n64/none) json wins. A marker file
(~/.g4eval_logged) makes reruns idempotent: a (run,step,acc) already logged is skipped,
but a new acc for the same step (higher n) still logs.

  WANDB_API_KEY=... .venv/bin/python scripts/misc/log_g4eval_wandb.py
"""
import glob, json, os, re
import wandb

RUN_MAP = {  # tag-prefix -> training run name (longest-first; da0 is a prefix of da01/da003)
    "qhn64":    "qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16",
    "grp4frz":  "ground_grp4_freeze_from_iso1_20k",
    "grp4noda": "ground_grp4_noda_from_iso1_20k",
    "grp4iso": "ground_grp4_from_iso1_20k",
    "iso1":  "msmarco_iso1_valce",
    "da003": "ground_msmarco_iso_grp4_da003_ft",
    "da01":  "ground_msmarco_iso_grp4_ft",
    "da0":   "ground_msmarco_iso_grp4_da0_ft",
}
# alias for the oracle metric: 4-doc grouped runs vs the 1-doc isolation run
def alias_for(run):
    return "msmarco_g4" if "grp4" in run else "msmarco_oracle"
MARKER = os.path.expanduser("~/.g4eval_logged")


def parse_tag(tag):
    """g4eval_iso1_12k_bs1 -> (run, step, n, alias). bs=1 tags get a distinct '_bs1' alias
    so the clean single-doc oracle curve never mixes with contaminated bs>1 points."""
    body = tag[len("g4eval_"):] if tag.startswith("g4eval_") else tag
    run = None
    for pref in ("qhn64", "grp4frz", "grp4noda", "grp4iso", "iso1", "da003", "da01", "da0"):   # longest-first
        if body == pref or body.startswith(pref + "_"):
            run = RUN_MAP[pref]; break
    if run is None:
        return None
    ms = re.search(r"_(\d+)k(?:_|$)", body)
    if not ms:
        return None
    step = int(ms.group(1)) * 1000
    mn = re.search(r"_n(\d+)", body)
    n = int(mn.group(1)) if mn else 64
    mbs = re.search(r"_bs(\d+)", body)  # explicit batch-size marker -> distinct alias (bs1 != bs16)
    alias = alias_for(run) + (f"_bs{mbs.group(1)}" if mbs else "")
    return run, step, n, alias


def main():
    # gather (run, step, alias) -> best (n, acc, jsonfile)
    best = {}
    for jf in glob.glob("/tmp/g4eval_*/eval_results/step_*/msmarco_g4/outputs/*.json"):
        tag = jf.split("/tmp/")[1].split("/")[0]
        parsed = parse_tag(tag)
        if not parsed:
            continue
        run, step, n, alias = parsed
        try:
            acc = json.load(open(jf))["metrics"]["llm_judge_accuracy"]
        except Exception:
            continue
        key = (run, step, alias)
        if key not in best or n > best[key][0]:
            best[key] = (n, acc, jf)

    logged = set()
    if os.path.exists(MARKER):
        logged = set(l.strip() for l in open(MARKER))

    by_run = {}
    for (run, step, alias), (n, acc, jf) in best.items():
        sig = f"{run}\t{step}\t{alias}\t{acc}"
        if sig in logged:
            continue
        by_run.setdefault(run, []).append((step, n, acc, jf, sig, alias))

    if not by_run:
        print("nothing new to log"); return

    new_sigs = []
    for run, items in sorted(by_run.items()):
        rid = f"{run}_geval"
        r = wandb.init(project="memory-layers", name=f"{run}_eval", id=rid,
                       resume="allow", reinit=True,
                       settings=wandb.Settings(init_timeout=300))
        wandb.define_metric("train_step")
        wandb.define_metric("eval/*", step_metric="train_step")
        for step, n, acc, jf, sig, alias in sorted(items):
            wandb.log({f"eval/{alias}/llm_judge_accuracy": acc,
                       f"eval/{alias}/n_samples": n, "train_step": step})
            # generation samples as an artifact, matching generation_embed.py's naming
            art = wandb.Artifact(name=f"{rid}-eval-{alias}-step-{step}-results",
                                 type="evaluation_results")
            art.add_file(jf)
            wandb.log_artifact(art)
            new_sigs.append(sig)
            print(f"{run} step{step} n{n} {alias} acc={acc} +artifact")
        r.finish()

    with open(MARKER, "a") as f:
        for sig in new_sigs:
            f.write(sig + "\n")
    print(f"LOGGED {len(new_sigs)} new points")


if __name__ == "__main__":
    main()
