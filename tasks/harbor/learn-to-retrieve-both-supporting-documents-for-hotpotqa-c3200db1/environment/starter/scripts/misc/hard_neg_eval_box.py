"""Dedicated eval box for the hard-neg (think) SFT4B run — runs ON a rohun* v6e-8, evaluates
saved GCS checkpoints, never touches training. Adapted from ground_eval_box.py.

TODO — ABSTRACT THIS. This file is ~80% identical to ground_eval_box.py (itself forked from
sim_eval_box.py): the GCS discovery, the scan loop, the sentinel, the eval.py invocation and the
train_step wandb axis are all generic; only the task list, result/env prefixes, wandb target and
per-task cmd extras differ. Forking again means re-inheriting every subtlety and fixing every bug
N times. Plan + the full config surface: wiki/evaluation/eval-boxes.md ("TODO — abstract the box").
Do it before a third box gets forked.

WHY A BOX AND NOT IN-LOOP: llm_judge_accuracy and lexical_grounding are produced by
evals/shared.py::run_metrics_pipeline, which runs in eval.py's PARENT process after the JAX
worker exits and frees the TPU for the vLLM judge. trainer.py::_run_evals only keeps
evaluator.evaluate()'s inference_metrics and never calls run_metrics, so a task's `metrics:`
block is inert during training. Judged accuracy therefore has to come from a box like this one.

Self-driving loop: every SCAN_S, for each tracked run, find its latest checkpoint; for each
milestone multiple (MILESTONE) with no result yet, evaluate every task and upload
gs://.../hard_neg_eval/<run>/step<N>/<alias>.json. Idempotent (skips existing results), so
several boxes can run concurrently over the same runs sharded by dataset, no coordination.

TARGET = ONE RUN-DIR, NOT A RUN NAME. You pass the full run-dir basename
(`<run_name>-<YYYY-MM-DD>-<HH-MM-SS>`), which is what train.py checkpoints into. This is the
whole safety property: a run_name is NOT unique — every launch mints its own dir, and re-using a
name months later re-uses the prefix. A box that scanned by name would take the max step across
ALL of them; the hard-neg name already had a COMPLETE April run at steps 40k-100k, so a fresh
launch would have made the box evaluate the APRIL model and log it into the NEW wandb run at
train_step=100000 — a plausible-looking curve from the wrong model.

WANDB: points are written INTO THE TRAINING RUN (not a companion '<run>_eval' run) as a
shared-mode SECONDARY writer, so eval/* lands beside train/* on one run. The id is derived from
the SAME run-dir via utils.wandb_run_id_from_run_dir, so the wandb run and the GCS folder are 1:1
and this box cannot write into a different launch's run. Requires the training run to have been
launched with trainer.wandb_run_id=auto. x_primary=False + x_update_finish_state=False mean our
per-point run.finish() never marks the still-running training run as finished.
The x-axis is the custom `train_step` metric, NOT wandb's monotonic step: every task logs at the
SAME checkpoint step, and wandb.log(step=step) would keep only the first task per step and drop
the rest (the bug that lost musique in the ground box).

A preemption-resume mints a NEW run-dir (and so a new wandb run); point a box at that dir too.

Usage (per box, detached):
    HARD_NEG_EVAL_DATASETS=msmarco PYTHONPATH=. .venv/bin/python scripts/misc/hard_neg_eval_box.py \
        qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-16-15-14-02

Env: GCS creds, GCS_BUCKET, GCS_BUCKET_PROJECT; HARD_NEG_EVAL_SAMPLES (128),
     HARD_NEG_EVAL_MILESTONE (10000), HARD_NEG_EVAL_SCAN_S (900), HARD_NEG_EVAL_DATASETS
     (default all 4), HARD_NEG_EVAL_WANDB (1), HARD_NEG_EVAL_PROJECT (memory-layers).
"""
import os, re, json, subprocess, sys, time, glob

BUCKET = os.environ["GCS_BUCKET"]
PROJECT = os.environ.get("GCS_BUCKET_PROJECT")
SAMPLES = int(os.environ.get("HARD_NEG_EVAL_SAMPLES", "128"))
MILESTONE = int(os.environ.get("HARD_NEG_EVAL_MILESTONE", "10000"))
SCAN_S = int(os.environ.get("HARD_NEG_EVAL_SCAN_S", "900"))
WANDB = os.environ.get("HARD_NEG_EVAL_WANDB", "1") == "1"
WANDB_PROJECT = os.environ.get("HARD_NEG_EVAL_PROJECT", "memory-layers")
RESULT_PREFIX = f"gs://{BUCKET}/hard_neg_eval"

# (alias, task-config-name). Order matters: the step-done sentinel is TASKS[-1], so a step counts
# as complete only once the LAST task lands -> a slow/interrupted corpus task is retried, not
# skipped. ScienceQA NLL is cheap and goes first for a quick point; the corpus generations are
# ~10x slower and follow.
DATASET_TASKS = {
    "science":  [("science_qa_nll", "nll_science_qa_hard_neg_think_n128")],
    "msmarco":  [("msmarco_c512", "gen_large_mem_msmarco_c512")],
    "hotpotqa": [("hotpotqa_c512", "gen_large_mem_hotpotqa_c512")],
    "musique":  [("musique_c512", "gen_large_mem_musique_c512")],
}
_selected = os.environ.get("HARD_NEG_EVAL_DATASETS", "science,msmarco,hotpotqa,musique").split(",")
TASKS = [t for ds in _selected if ds.strip() in DATASET_TASKS for t in DATASET_TASKS[ds.strip()]]

# Force the read-channel telemetry on at eval regardless of how the checkpoint was trained:
# evals/nll.py gates mem_pos_weight_mass on the aux config, and eval_worker seeds that from the
# TRAIN cfg. Passing it here keeps the eval self-contained.
AUX_OVERRIDES = [
    "+aux_losses.mem_pos_weight_mass.enabled=true",
    "+aux_losses.mem_pos_weight_mass.weight=0.0",
    "+aux_losses.doc_access_acc.enabled=true",
    "+aux_losses.doc_access_acc.weight=0.0",
]

# Extra Hydra overrides appended to every eval.py call, space-separated.
# The reason this hook exists: the judge defaults in evals/metrics/llm_judge.py are Qwen3-8B at
# tensor_parallel_size=8 — a v6e-8 assumption. TP must divide the local chip count, so on a
# 4-chip box (v5p-4 / v6e-4) the judge's vLLM cannot start at all and every generation task dies
# after doing the expensive part. Pass e.g.
#   HARD_NEG_EVAL_EXTRA_OVERRIDES="+evals.{alias}.eval.metrics.llm_judge_accuracy.model_id=Qwen/Qwen3-4B ..."
# {alias} is substituted per task so one setting covers every task in the run.
EXTRA_OVERRIDES = [o for o in os.environ.get("HARD_NEG_EVAL_EXTRA_OVERRIDES", "").split() if o]

from google.cloud import storage
_client = storage.Client(project=PROJECT) if PROJECT else storage.Client()


def _gcs_exists(gs_path):
    assert gs_path.startswith("gs://")
    _, _, rest = gs_path.partition("gs://")
    bkt, _, key = rest.partition("/")
    return storage.Blob(bucket=_client.bucket(bkt), name=key).exists(_client)


def steps_in_run_dir(run_dir):
    """(model_name, sorted[steps]) for THE ONE run-dir we were pointed at.

    Scoped to a single dir on purpose — see the module docstring. There is no cross-dir
    resolution here because there is nothing to resolve: one launch, one dir, one wandb run.
    """
    model, steps = None, set()
    for blob in _client.list_blobs(BUCKET, prefix=f"{run_dir}/"):
        parts = blob.name.split("/")
        if len(parts) < 3 or parts[0] != run_dir:
            continue
        if not re.fullmatch(r"\d+", parts[2]):
            continue
        model = parts[1]
        steps.add(int(parts[2]))
    return model, sorted(steps)


def latest_ckpt(run_dir):
    _model, steps = steps_in_run_dir(run_dir)
    return steps[-1] if steps else -1


def ckpt_path_for_step(run_dir, step):
    """gs:// path of this run-dir's checkpoint at `step`, or None if it isn't there (yet, or
    rotated away by orbax's max_to_keep)."""
    model, steps = steps_in_run_dir(run_dir)
    if model is None or int(step) not in steps:
        return None
    return f"gs://{BUCKET}/{run_dir}/{model}/{step}"


def _metrics_from_result(path):
    """Scalar metrics out of an eval result JSON. The generation evaluators write
    {"metrics": {...}} (run_metrics merges llm_judge_accuracy/lexical_grounding into it); the NLL
    evaluator writes {"stats": {...}} instead. Accept both and keep only real scalars."""
    with open(path) as f:
        data = json.load(f)
    block = data.get("metrics") or data.get("stats") or {}
    out = {}
    for k, v in block.items():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _log_wandb(run_dir, step, alias, result_json_path):
    """Log every scalar from the result into THIS run-dir's training run, keyed on train_step."""
    if not WANDB:
        return
    try:
        import wandb
        sys.path.insert(0, os.getcwd())
        from utils import wandb_run_id_from_run_dir

        metrics = _metrics_from_result(result_json_path)
        if not metrics:
            print(f"[{run_dir}@{step}] {alias}: no scalar metrics, skip wandb", flush=True)
            return
        # Derived from the run-dir we are evaluating — the same string train.py derived its id
        # from — so we cannot write into a different launch's run even if the run_name is shared.
        wid = wandb_run_id_from_run_dir(run_dir)
        run = None
        for attempt in range(2):
            try:
                # resume="allow" attaches to the training run. mode="shared" + x_primary=False is
                # what makes a second writer safe; x_update_finish_state=False keeps our finish()
                # from marking the live training run as finished.
                run = wandb.init(
                    project=WANDB_PROJECT, id=wid, resume="allow", reinit=True,
                    settings=wandb.Settings(
                        mode="shared", x_primary=False, x_label="eval",
                        x_update_finish_state=False, init_timeout=300,
                    ),
                )
                break
            except Exception as ie:
                print(f"[{run_dir}@{step}] wandb init attempt {attempt} failed: {ie}", flush=True)
        if run is None:
            return
        run.define_metric("train_step")
        run.define_metric("eval/*", step_metric="train_step")
        payload = {f"eval/{alias}/{k}": v for k, v in metrics.items()}
        payload["train_step"] = step
        run.log(payload)
        print(f"[{run_dir}@{step}] wandb {alias}: {sorted(metrics)}", flush=True)
        try:
            art = wandb.Artifact(f"{run_dir}-{alias}-step{step}", type="eval_samples")
            art.add_file(result_json_path)
            run.log_artifact(art)
        except Exception as ae:
            print(f"[{run_dir}@{step}] wandb artifact warn: {ae}", flush=True)
        run.finish()
    except Exception as e:
        print(f"[{run_dir}@{step}] wandb warn: {e}", flush=True)


def eval_ckpt(run_dir, step):
    ckpt = ckpt_path_for_step(run_dir, step)
    if ckpt is None:
        print(f"[{run_dir}] step {step} ckpt vanished before eval", flush=True)
        return
    for alias, task in TASKS:
        # Results are keyed by run-dir (not run_name), so two launches sharing a name keep
        # separate result trees and can't idempotency-skip each other's work.
        dst = f"{RESULT_PREFIX}/{run_dir}/step{step}/{alias}.json"
        if _gcs_exists(dst):
            print(f"[{run_dir}@{step}] {alias} exists, skip", flush=True)
            continue
        # Free the TPU: the previous eval's vLLM judge lingers on /dev/vfio; free_tpu_devices()
        # fusers the device nodes and SIGKILLs every holder (a04fc00 / 560b8de).
        try:
            from evals.vllm import VLLMInference
            for _ in range(3):
                if VLLMInference.free_tpu_devices() == 0:
                    break
                time.sleep(3)
        except Exception as e:
            print(f"[{run_dir}@{step}] free_tpu warn: {e}", flush=True)
        time.sleep(5)
        out = os.path.expanduser(f"~/evalbox/{run_dir}_{step}_{alias}")
        subprocess.run(["rm", "-rf", out])
        print(f"[{run_dir}@{step}] eval {alias} ({task}) -> {dst}", flush=True)
        # tp stays 1: the mem-model checkpoint loader isn't TP-aware (a head-split reshape
        # bf16[4096@model,...] -> (4,1024,...) can't split a model-sharded axis -> ShardingTypeError),
        # so tp>1 breaks weight loading.
        # use_wandb=false: eval.py would otherwise open its OWN run in memory-layers-eval; we log
        # into the training run ourselves below.
        cmd = ["python", "eval.py", f"checkpoint_dir={ckpt}",
               "~eval_set@evals=pretraining",
               f"+eval/tasks@evals.{alias}={task}",
               "tp_devices=1", "use_wandb=false", f"hydra.run.dir={out}"] + AUX_OVERRIDES
        cmd += [o.replace("{alias}", alias) for o in EXTRA_OVERRIDES]
        # num_samples is a generation knob; the NLL task sizes n with eval_steps x batch_size.
        if not alias.startswith("science_qa"):
            cmd.append(f"evals.{alias}.eval.num_samples={SAMPLES}")
        rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "."}).returncode
        if rc != 0:
            print(f"[{run_dir}@{step}] {alias} FAILED rc={rc}", flush=True)
            continue
        res = glob.glob(f"{out}/eval_results/step_{step}/{alias}/outputs/*.json")
        if not res:
            print(f"[{run_dir}@{step}] {alias} no result file", flush=True)
            continue
        import gcsfs
        gcsfs.GCSFileSystem().put(res[0], dst)
        _log_wandb(run_dir, step, alias, res[0])
        print(f"[{run_dir}@{step}] {alias} UPLOADED", flush=True)


def main(run_dirs):
    sys.path.insert(0, os.getcwd())
    from utils import wandb_run_id_from_run_dir

    print(f"=== HARD_NEG_EVAL_BOX tasks={[a for a, _ in TASKS]} milestone={MILESTONE} "
          f"samples={SAMPLES} scan={SCAN_S}s wandb={WANDB} ===", flush=True)
    for rd in run_dirs:
        # Fail fast on a run_name passed where a run-dir belongs: that is the exact mistake this
        # design exists to prevent, and wandb_run_id_from_run_dir raises on a missing timestamp.
        wid = wandb_run_id_from_run_dir(rd)
        model, steps = steps_in_run_dir(rd)
        rng = f"{steps[0]}..{steps[-1]}" if steps else "no checkpoints yet"
        print(f"[{rd}]\n    wandb id : {wid}\n    model    : {model}\n    steps    : {rng}",
              flush=True)

    # Breadth-first: each pass, evaluate at most ONE (newest unevaluated) checkpoint PER run-dir,
    # so several runs' curves fill together instead of one draining its whole history first. Sleep
    # only when a full pass found nothing to do. Sentinel = the LAST task (a step counts done only
    # when every task for it exists); gaps use `continue` (not `break`) so older steps still backfill.
    sentinel = TASKS[-1][0]
    while True:
        did_work = False
        for rd in run_dirs:
            try:
                step = latest_ckpt(rd)
                if step < MILESTONE:
                    continue
                m = (step // MILESTONE) * MILESTONE
                for cand in range(m, 0, -MILESTONE):
                    if _gcs_exists(f"{RESULT_PREFIX}/{rd}/step{cand}/{sentinel}.json"):
                        continue
                    if ckpt_path_for_step(rd, cand) is None:
                        continue
                    eval_ckpt(rd, cand)
                    did_work = True
                    break  # one checkpoint per run-dir per pass -> round-robin
            except Exception as e:
                print(f"[{rd}] scan error: {e}", flush=True)
        if not did_work:
            time.sleep(SCAN_S)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: hard_neg_eval_box.py <run_dir> [<run_dir> ...]\n"
              "  <run_dir> is the GCS run-dir BASENAME, not a run_name:\n"
              "    qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-16-15-14-02\n"
              "  train.py prints it at startup next to the wandb id.")
        sys.exit(2)
    main(sys.argv[1:])
