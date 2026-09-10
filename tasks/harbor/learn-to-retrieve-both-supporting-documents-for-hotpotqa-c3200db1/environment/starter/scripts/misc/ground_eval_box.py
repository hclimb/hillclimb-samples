"""Dedicated eval box for the GROUNDING experiments — runs ON a rohun* v6e-8, evaluates
saved GCS checkpoints, never touches training. Adapted from sim_eval_box.py.

Self-driving loop: every SCAN_S, for each tracked run, find its latest checkpoint; for each
milestone multiple (MILESTONE) with no result yet, evaluate it on the selected datasets and
upload gs://.../ground_eval/<run>/step<N>/<task>.json. Idempotent (skips existing results),
so 3 boxes can run concurrently over the same runs sharded by dataset with no coordination.

Per dataset we run BOTH channels (grounding_experiments_plan.md):
  corpus  — full retrieval into the memory layer (gen_large_mem_msa_*)
  oracle  — memory bank from the gold pos_doc only (gen_embed_msa_*_oracle)

Eval load = 4 runs × 6 tasks per cycle; that's ~3× the sim box, so shard across 3 boxes by
dataset (GROUND_EVAL_DATASETS=msmarco / hotpotqa / musique per box). Keep MILESTONE and
SAMPLES tuned so a milestone is evaluated inside the orbax keep-N rotation window.

Usage (per box, detached):
    GROUND_EVAL_DATASETS=msmarco PYTHONPATH=. .venv/bin/python scripts/misc/ground_eval_box.py \
        ground_control ground_s1_zeroinit_4layer ground_s2_kv_split ground_s3_span

Env: GCS creds, GCS_BUCKET, GCS_BUCKET_PROJECT; SIM_EVAL_SAMPLES (128), SIM_EVAL_MILESTONE
     (2000), SIM_EVAL_SCAN_S (900); GROUND_EVAL_DATASETS (default all 3);
     GROUND_EVAL_WANDB=1 to also log accuracy to wandb keyed by step.
"""
import os, re, json, subprocess, sys, time

BUCKET = os.environ["GCS_BUCKET"]
PROJECT = os.environ.get("GCS_BUCKET_PROJECT")
SAMPLES = int(os.environ.get("SIM_EVAL_SAMPLES", "128"))
MILESTONE = int(os.environ.get("SIM_EVAL_MILESTONE", "2000"))
SCAN_S = int(os.environ.get("SIM_EVAL_SCAN_S", "900"))
WANDB = os.environ.get("GROUND_EVAL_WANDB", "0") == "1"
RESULT_PREFIX = f"gs://{BUCKET}/ground_eval"

# (alias, task-config-name) per dataset. Corpus tasks carry eval.type generation_large_mem_msa;
# oracle tasks carry generation_embed — each task yaml sets its own type, so no type override.
DATASET_TASKS = {
    # Oracle FIRST (fast ~2-3min -> quick wandb points), corpus SECOND (slow ~30min full-corpus gen).
    # The step-done sentinel below is EVAL_TASKS[-1] (the LAST/corpus task), so a step is only marked
    # complete once corpus finishes -> a slow/interrupted corpus is retried, not skipped.
    "msmarco":  [("msmarco_oracle", "gen_embed_msa_msmarco_oracle"), ("msmarco_corpus", "gen_large_mem_msa_msmarco_v1")],
    "hotpotqa": [("hotpotqa_oracle", "gen_embed_msa_hotpotqa_oracle"), ("hotpotqa_corpus", "gen_large_mem_msa_hotpotqa")],
    "musique":  [("musique_oracle", "gen_embed_msa_musique_oracle"), ("musique_corpus", "gen_large_mem_msa_musique")],
}
_selected = os.environ.get("GROUND_EVAL_DATASETS", "msmarco,hotpotqa,musique").split(",")
EVAL_TASKS = [t for ds in _selected if ds.strip() in DATASET_TASKS for t in DATASET_TASKS[ds.strip()]]
# Corpus evals are ~10x slower (~30min full-corpus gen) and starve the fast oracle sweep. Default OFF
# so all runs' oracle curves populate quickly; set GROUND_EVAL_CORPUS=1 for the slow corpus pass.
CORPUS = os.environ.get("GROUND_EVAL_CORPUS", "0") == "1"
if not CORPUS:
    EVAL_TASKS = [(a, t) for (a, t) in EVAL_TASKS if not a.endswith("_corpus")]

# Runs to evaluate oracle tasks at batch_size=1 for a CLEAN per-query isolated oracle: at bs>1 the
# memory bank is shared across the batch (grounding-exps has no per_query_isolation), so a query can
# read other batch-mates' docs. bs=1 -> bank = only that query's own gold doc. Logged to a separate
# '<run>_eval_bs1' wandb run so it doesn't mix with the contaminated bs=16 curve.
BS1_RUNS = {"4B_msmarco_triplets_topk32_per_query_isolation"}


def _run_prefix(run_name):
    """GCS result prefix for a run. bs=1 runs use a separate '<run>_bs1' dir so the clean-isolation
    results don't collide with (or get idempotency-skipped by) the contaminated bs=16 results."""
    tag = "_bs1" if run_name in BS1_RUNS else ""
    return f"{RESULT_PREFIX}/{run_name}{tag}"

from google.cloud import storage
_client = storage.Client(project=PROJECT) if PROJECT else storage.Client()


def _gcs_exists(gs_path):
    assert gs_path.startswith("gs://")
    _, _, rest = gs_path.partition("gs://")
    bkt, _, key = rest.partition("/")
    return storage.Blob(bucket=_client.bucket(bkt), name=key).exists(_client)


# A run can have MULTIPLE run-dirs (each launch/preemption-restart makes a new timestamped dir),
# and step numbers repeat across them. Resolve each step to the NEWEST run-dir containing it
# (timestamp sorts chronologically) — otherwise step 4000 from two dirs = two different accuracies.
def _steps_by_dir(run_name):
    """{run_dir: (model, set(steps))} across all timestamped run-dirs of run_name."""
    dirs = {}
    for blob in _client.list_blobs(BUCKET, prefix=f"{run_name}-"):
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue
        run_dir, model, step = parts[0], parts[1], parts[2]
        if not re.fullmatch(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", run_dir[len(run_name):]):
            continue
        if not re.fullmatch(r"\d+", step):
            continue
        d = dirs.setdefault(run_dir, [model, set()])
        d[1].add(int(step))
    return dirs


def latest_ckpt(run_name):
    dirs = _steps_by_dir(run_name)
    if not dirs:
        return -1, None
    # global highest step; the dir is resolved per-step by ckpt_dir_for_step (newest containing it)
    best_step = max(max(s) for _, s in dirs.values())
    return best_step, ckpt_dir_for_step(run_name, best_step)


def ckpt_dir_for_step(run_name, step):
    # Newest run-dir (max timestamp) that CONTAINS this step. Resolves both the restart-collision
    # (shared step numbers across dirs -> pick the newest model) and resume-split runs (a step only
    # present in an older dir -> still found).
    dirs = _steps_by_dir(run_name)
    for run_dir in sorted(dirs, reverse=True):
        model, steps = dirs[run_dir]
        if int(step) in steps:
            return f"gs://{BUCKET}/{run_dir}/{model}"
    return None


def _log_wandb(run_name, step, alias, result_json_path):
    if not WANDB:
        return
    try:
        import wandb
        with open(result_json_path) as f:
            data = json.load(f)
        # pull the llm-judge accuracy out of the eval result (best-effort across shapes)
        acc = None
        def _find(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if "judge" in k.lower() and "acc" in k.lower() and isinstance(v, (int, float)):
                        return v
                    r = _find(v)
                    if r is not None:
                        return r
            return None
        acc = _find(data)
        if acc is None:
            return
        # Creating a brand-new run (first point for s1/s2/s3_eval) is slower than resuming an existing
        # one and blew the default 90s init timeout -> point dropped, run never created. Give it 300s
        # and one retry so new runs actually come into being.
        # bs=1 (clean-isolation) runs log to a SEPARATE '<run>_eval_bs1' wandb run so the clean curve
        # doesn't mix with the contaminated bs=16 points already logged under '<run>_eval'.
        suffix = "_bs1" if run_name in BS1_RUNS else ""
        wname = f"{run_name}_eval{suffix}"
        wid = f"{run_name}_geval{suffix}"
        run = None
        for attempt in range(2):
            try:
                # id uses a FRESH namespace (_geval) not the old {run}_eval: the clean-slate deleted the
                # old s1/s2/s3 _eval run ids, and resume='allow' on a DELETED id hangs (300s timeout) ->
                # a never-used id creates cleanly. Display name stays {run}_eval.
                run = wandb.init(project="memory-layers", name=wname, id=wid,
                                 resume="allow", reinit=True,
                                 settings=wandb.Settings(init_timeout=300))
                break
            except Exception as ie:
                print(f"[{run_name}@{step}] wandb init attempt {attempt} failed: {ie}", flush=True)
        if run is None:
            return
        # Use train_step as the x-axis, NOT wandb's monotonic step: all 6 aliases (msmarco/hotpotqa/
        # musique x corpus/oracle) log at the SAME checkpoint step, and wandb.log(step=step) would
        # keep only the first alias per step and DROP the rest (that's why musique was missing).
        wandb.define_metric("train_step")
        wandb.define_metric("eval/*", step_metric="train_step")
        wandb.log({f"eval/{alias}/llm_judge_accuracy": acc, "train_step": step})
        # Also save the full sample JSON as a wandb artifact so outputs are inspectable off-GCS.
        try:
            art = wandb.Artifact(f"{run_name}-{alias}-step{step}", type="eval_samples")
            art.add_file(result_json_path)
            run.log_artifact(art)
        except Exception as ae:
            print(f"[{run_name}@{step}] wandb artifact warn: {ae}", flush=True)
        run.finish()
    except Exception as e:
        print(f"[{run_name}@{step}] wandb warn: {e}", flush=True)


def eval_ckpt(run_name, step):
    ckpt_dir = ckpt_dir_for_step(run_name, step)
    if ckpt_dir is None:
        print(f"[{run_name}] step {step} ckpt vanished before eval", flush=True)
        return
    ckpt = f"{ckpt_dir}/{step}"
    for alias, task in EVAL_TASKS:
        dst = f"{_run_prefix(run_name)}/step{step}/{alias}.json"
        if _gcs_exists(dst):
            print(f"[{run_name}@{step}] {alias} exists, skip", flush=True)
            continue
        # Free the TPU: the previous eval's vLLM judge lingers on /dev/vfio; free_tpu_devices()
        # fusers the device nodes and SIGKILLs every holder (a04fc00 / 560b8de).
        try:
            from evals.vllm import VLLMInference
            for _ in range(3):
                n = VLLMInference.free_tpu_devices()
                if n == 0:
                    break
                time.sleep(3)
        except Exception as e:
            print(f"[{run_name}@{step}] free_tpu warn: {e}", flush=True)
        time.sleep(5)
        out = os.path.expanduser(f"~/evalbox/{run_name}_{step}_{alias}")
        subprocess.run(["rm", "-rf", out])
        print(f"[{run_name}@{step}] eval {alias} ({task}) -> {dst}", flush=True)
        is_corpus = alias.endswith("_corpus")
        # tp stays 1: the mem-model checkpoint loader isn't TP-aware (a head-split reshape
        # bf16[4096@model,...] -> (4,1024,...) can't split a model-sharded axis -> ShardingTypeError),
        # so tp>1 breaks weight loading. Corpus OOM (msmarco +2.1G over HBM at bs=16) is generation
        # activations, NOT the bank (~29k slots, ~120MB) -> shave it with a smaller gen batch instead.
        cmd = ["python", "eval.py", f"checkpoint_dir={ckpt}",
               "~eval_set@evals=pretraining",
               f"+eval/tasks@evals.{alias}={task}",
               f"evals.{alias}.eval.num_samples={SAMPLES}",
               "tp_devices=1", "use_wandb=false", f"hydra.run.dir={out}"]
        # Corpus tasks are the gen_large_mem_msa_* configs, but the *_msa evaluator wants an MSA
        # 'msa' cfg block our memory-layer model lacks (KeyError 'msa'). Override to the memory-layer
        # evaluator generation_large_mem (same as the sim-pairs eval box did). Halve the gen batch
        # (16 -> 8) so the full-corpus forward fits HBM; same corpus + same num_samples.
        if is_corpus:
            cmd.append(f"evals.{alias}.eval.type=generation_large_mem")
            cmd.append(f"evals.{alias}.dataset.batch_size=8")
        elif run_name in BS1_RUNS:
            # bs=1 -> each query's bank holds only its own gold doc (clean oracle isolation).
            cmd.append(f"evals.{alias}.dataset.batch_size=1")
        rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "."}).returncode
        if rc != 0:
            print(f"[{run_name}@{step}] {alias} FAILED rc={rc}", flush=True)
            continue
        import glob
        res = glob.glob(f"{out}/eval_results/step_{step}/{alias}/outputs/*.json")
        if not res:
            print(f"[{run_name}@{step}] {alias} no result file", flush=True)
            continue
        import gcsfs
        gcsfs.GCSFileSystem().put(res[0], dst)
        _log_wandb(run_name, step, alias, res[0])
        print(f"[{run_name}@{step}] {alias} UPLOADED", flush=True)


def main(run_names):
    print(f"=== GROUND_EVAL_BOX runs={run_names} tasks={[a for a,_ in EVAL_TASKS]} "
          f"milestone={MILESTONE} samples={SAMPLES} scan={SCAN_S}s ===", flush=True)
    # Breadth-first: each pass, evaluate at most ONE (newest unevaluated) checkpoint PER run, so all
    # runs' curves fill together instead of one run draining its whole history first (which starved
    # s1/s2/s3 behind control). Sleep only when a full pass found nothing to do. Sentinel = the LAST
    # task in EVAL_TASKS (a step counts done only when every task for it exists); with gaps handled by
    # `continue` (not `break`) so a missing older step is still backfilled.
    sentinel = EVAL_TASKS[-1][0]
    while True:
        did_work = False
        for run in run_names:
            try:
                step, _ = latest_ckpt(run)
                if step < MILESTONE:
                    continue
                m = (step // MILESTONE) * MILESTONE
                for cand in range(m, 0, -MILESTONE):
                    if _gcs_exists(f"{_run_prefix(run)}/step{cand}/{sentinel}.json"):
                        continue
                    if ckpt_dir_for_step(run, cand) is None:
                        continue
                    eval_ckpt(run, cand)
                    did_work = True
                    break  # one checkpoint per run per pass -> round-robin across runs
            except Exception as e:
                print(f"[{run}] scan error: {e}", flush=True)
        if not did_work:
            time.sleep(SCAN_S)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ground_control", "ground_s1_zeroinit_4layer", "ground_s2_kv_split", "ground_s3_span"])
