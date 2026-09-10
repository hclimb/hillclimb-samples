"""Dump eval-box result samples to readable markdown.

Usage:
  python scripts/misc/read_eval_samples.py <run> <step> <dataset> [--only wrong|right|all] [--n N]

The gsutil account defaults to the GCS identity from .env (GCS_USER_EMAIL); override with
CLOUDSDK_CORE_ACCOUNT=... if you need a different one.

Example: python scripts/misc/read_eval_samples.py simpair_sim40nce_v3 24000 natural_questions --only wrong
Result JSON: gs://$GCS_BUCKET/simpair_eval/<run>/step<step>/<dataset>.json
"""
import json, subprocess, sys, os

run, step, ds = sys.argv[1], sys.argv[2], sys.argv[3]
only = "all"; n = 999
if "--only" in sys.argv: only = sys.argv[sys.argv.index("--only")+1]
if "--n" in sys.argv: n = int(sys.argv[sys.argv.index("--n")+1])
bucket = os.environ.get("GCS_BUCKET", "memory-layers-training")
# Grounding eval box writes to ground_eval/<run>/step<step>/<alias>.json (aliases: msmarco_oracle,
# msmarco_corpus, musique_oracle, ...). Override prefix with EVAL_PREFIX for the old simpair_eval.
prefix = os.environ.get("EVAL_PREFIX", "ground_eval")
path = f"gs://{bucket}/{prefix}/{run}/step{step}/{ds}.json"
raw = subprocess.run(["gsutil", "cat", path], capture_output=True, text=True,
                     env={**os.environ, "CLOUDSDK_CORE_ACCOUNT": os.environ.get(
                         "CLOUDSDK_CORE_ACCOUNT",
                         os.environ.get("GCS_USER_EMAIL") or "rohunagrawal@gmail.com")}).stdout
d = json.loads(raw)
acc = d["metrics"].get("llm_judge_accuracy")
print(f"# {run} @ {step} — {ds}   (judge acc = {acc})\n")
for i, s in enumerate(d.get("samples", [])):
    j = s.get("llm_judge_accuracy", 0)
    if only == "wrong" and j != 0: continue
    if only == "right" and j != 1: continue
    if i >= n: break
    q = s.get("prompt", "").replace("<|im_start|>user", "").replace("<|im_end|>", "").replace("<|im_start|>assistant", "").strip()
    hit = s.get('doc_token_hit_rate')          # corpus evals only; absent for oracle-memory
    mass = s.get('pos_slot_weight_mass')       # answer-slot diagnostic (grounding oracle evals)
    extra = f"doc_token_hit={hit:.2f}" if isinstance(hit, (int, float)) else (
            f"pos_slot_mass={mass:.3f}" if isinstance(mass, (int, float)) else "")
    print(f"## [{i}] judge={'RIGHT' if j==1 else 'WRONG'}  {extra}")
    print(f"**Q:** {q}")
    print(f"**Model answer:** {s.get('generated_answer','').strip()[:600]}")
    print(f"**Ground truth:** {s.get('ground_truth','')}")
    print(f"**Retrieved doc:** {s.get('doc','')}")
    print()
