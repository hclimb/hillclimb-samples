"""Dedicated eval box — runs ON b-1, evaluates saved GCS checkpoints, never touches training.

Self-driving loop: every SCAN_INTERVAL, for each tracked run, find its latest checkpoint;
if its step is a MILESTONE multiple (default 8000) and no result exists yet, evaluate it on
popqa/nq/hotpotqa and upload to gs://.../simpair_eval/<run>/step<N>/<ds>.json. Idempotent
(skips existing results). This decouples eval from training so training runs are never paused.

Because orbax keeps only the last few checkpoints (max_to_keep=4, save_interval=2000 => last
~8000 steps), a milestone must be evaluated within ~8000 steps of being written; a 15-min scan
vs ~2h-per-8000-steps cadence catches every milestone comfortably.

Usage (on b-1, detached):
    PYTHONPATH=. .venv/bin/python scripts/misc/sim_eval_box.py \
        simpair_ctrl_v2 simpair_sim40_v2 simpair_sim40_nce simpair_sim15_nce

Env: GCS creds (GOOGLE_APPLICATION_CREDENTIALS), GCS_BUCKET, GCS_BUCKET_PROJECT;
     SIM_EVAL_SAMPLES (default 128), SIM_EVAL_MILESTONE (default 8000), SIM_EVAL_SCAN_S (900).
"""
import os, re, subprocess, sys, time

BUCKET = os.environ["GCS_BUCKET"]
PROJECT = os.environ.get("GCS_BUCKET_PROJECT")
SAMPLES = int(os.environ.get("SIM_EVAL_SAMPLES", "128"))
MILESTONE = int(os.environ.get("SIM_EVAL_MILESTONE", "8000"))
SCAN_S = int(os.environ.get("SIM_EVAL_SCAN_S", "900"))
RESULT_PREFIX = f"gs://{BUCKET}/simpair_eval"
# Trimmed to 2 datasets: at ~12 min/eval the box couldn't finish 4 runs × 3 datasets
# within the ~2h checkpoint-rotation window. popqa (single-hop) + natural_questions
# (single-hop) keep the box inside the window; hotpotqa (multi-hop) dropped for speed.
EVAL_DATASETS = [("popqa", "popqa"), ("natural_questions", "natural_questions")]

from google.cloud import storage
_client = storage.Client(project=PROJECT) if PROJECT else storage.Client()


def _gcs_exists(gs_path):
    assert gs_path.startswith("gs://")
    _, _, rest = gs_path.partition("gs://")
    bkt, _, key = rest.partition("/")
    return storage.Blob(bucket=_client.bucket(bkt), name=key).exists(_client)


def latest_ckpt(run_name):
    """Return (step, ckpt_dir_without_step) for the highest step across run-dir generations."""
    best_step, best_dir = -1, None
    for blob in _client.list_blobs(BUCKET, prefix=f"{run_name}-"):
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue
        run_dir, model, step = parts[0], parts[1], parts[2]
        if not re.fullmatch(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", run_dir[len(run_name):]):
            continue
        if not re.fullmatch(r"\d+", step):
            continue
        step = int(step)
        if step > best_step:
            best_step, best_dir = step, f"gs://{BUCKET}/{run_dir}/{model}"
    return best_step, best_dir


def ckpt_dir_for_step(run_name, step):
    """Find the run-dir generation whose {step}/ exists (checkpoints may span generations)."""
    for blob in _client.list_blobs(BUCKET, prefix=f"{run_name}-"):
        parts = blob.name.split("/")
        if len(parts) >= 3 and parts[2] == str(step) and \
           re.fullmatch(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", parts[0][len(run_name):]):
            return f"gs://{BUCKET}/{parts[0]}/{parts[1]}"
    return None


def eval_ckpt(run_name, step):
    ckpt_dir = ckpt_dir_for_step(run_name, step)
    if ckpt_dir is None:
        print(f"[{run_name}] step {step} ckpt vanished before eval", flush=True)
        return
    ckpt = f"{ckpt_dir}/{step}"
    for alias, ds in EVAL_DATASETS:
        dst = f"{RESULT_PREFIX}/{run_name}/step{step}/{ds}.json"
        if _gcs_exists(dst):
            print(f"[{run_name}@{step}] {ds} exists, skip", flush=True)
            continue
        # Free the TPU: the previous eval's vLLM judge server (and its worker
        # subprocesses) linger and hold /dev/vfio, so the next eval's generation
        # phase fails with "Device or resource busy". free_tpu_devices() uses fuser
        # on the TPU device nodes to SIGKILL every holder — more reliable than a
        # name-based pkill (vLLM's TPU workers don't match "vllm serve").
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
        out = os.path.expanduser(f"~/evalbox/{run_name}_{step}_{ds}")
        subprocess.run(["rm", "-rf", out])
        print(f"[{run_name}@{step}] eval {ds} -> {dst}", flush=True)
        rc = subprocess.run(
            ["python", "eval.py", f"checkpoint_dir={ckpt}",
             "~eval_set@evals=pretraining",
             f"+eval/tasks@evals.{alias}=gen_large_mem_msa_{ds}",
             f"evals.{alias}.eval.type=generation_large_mem",
             f"evals.{alias}.eval.num_samples={SAMPLES}",
             f"evals.{alias}.eval.doc_access_acc=false",
             "tp_devices=1", "use_wandb=false", f"hydra.run.dir={out}"],
            env={**os.environ, "PYTHONPATH": "."},
        ).returncode
        if rc != 0:
            print(f"[{run_name}@{step}] {ds} FAILED rc={rc}", flush=True)
            continue
        import glob
        res = glob.glob(f"{out}/eval_results/step_{step}/{alias}/outputs/*.json")
        if not res:
            print(f"[{run_name}@{step}] {ds} no result file", flush=True)
            continue
        import gcsfs
        gcsfs.GCSFileSystem().put(res[0], dst)
        print(f"[{run_name}@{step}] {ds} UPLOADED", flush=True)


def main(run_names):
    print(f"=== SIM_EVAL_BOX runs={run_names} milestone={MILESTONE} samples={SAMPLES} scan={SCAN_S}s ===", flush=True)
    while True:
        for run in run_names:
            try:
                step, _ = latest_ckpt(run)
                if step < MILESTONE:
                    continue
                # highest milestone <= latest available checkpoint
                m = (step // MILESTONE) * MILESTONE
                # evaluate any milestone from m down that isn't done yet and still on disk
                for cand in range(m, 0, -MILESTONE):
                    dst0 = f"{RESULT_PREFIX}/{run}/step{cand}/{EVAL_DATASETS[0][0]}.json"
                    if _gcs_exists(dst0):
                        break  # this and all lower milestones already evaluated
                    if ckpt_dir_for_step(run, cand) is None:
                        continue  # rotated out before we caught it
                    eval_ckpt(run, cand)
            except Exception as e:
                print(f"[{run}] scan error: {e}", flush=True)
        time.sleep(SCAN_S)


if __name__ == "__main__":
    main(sys.argv[1:] or ["simpair_ctrl_v2", "simpair_sim40_v2", "simpair_sim40_nce", "simpair_sim15_nce"])
