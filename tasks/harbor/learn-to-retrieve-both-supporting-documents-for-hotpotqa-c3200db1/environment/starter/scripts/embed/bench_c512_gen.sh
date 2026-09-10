#!/bin/bash
# Runs ON a box. A/B the 512-doc corpus-eval generation speed. Uses gen_large_mem's built-in
# MEMBENCH hook: iter 0 is JIT warmup, then BENCH_REPS timed reps on the compiled program, then
# it stops — so these are steady-state numbers, not compile time.
#
# WHY: the live eval box measured ~14.4 s/sample (~225 ms/decode step, ~36 tok/s aggregate at
# bs=8). Suspected cause: with MEM_REPLICATE_BANK unset, gen_large_mem shards mem_k and leaves
# mem_v on the HOST, so retrieval_ops.py::_lookup_cpu_mem_v does a jax.pure_callback (a blocking
# device->host->device round trip + numpy gather) on EVERY decode step. At 512 docs the bank is
# only ~0.27 GB with ~24 GB HBM free, so the offload buys nothing.
#
#   CKPT=gs://... bash scripts/embed/bench_c512_gen.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

# REQUIRED: unlike train.py (which calls setup_gcs_credentials() at import), eval.py does NOT
# configure GCS creds — every caller must export them, as the box run scripts do. Without this,
# gcsfs silently falls back to the VM's attached compute SA, which has no bucket access, and the
# eval dies reading the checkpoint's .hydra config with a Forbidden on storage.objects.list.
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?GCS_USER_EMAIL not set in ~/.env}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT
[ -f "$GOOGLE_APPLICATION_CREDENTIALS" ] || { echo "ERROR: ADC missing at $GOOGLE_APPLICATION_CREDENTIALS" >&2; exit 1; }

CKPT="${CKPT:?set CKPT=gs://.../qwen3_mem_embed/<step>}"
REPS="${BENCH_REPS:-3}"
OUT=$HOME/bench_c512

run_arm () {  # name, extra-env..., then hydra overrides via $ARM_OVERRIDES
  local name="$1"; shift
  local dir="$OUT/$name"
  rm -rf "$dir"; mkdir -p "$dir"
  echo
  echo "############ ARM: $name ############"
  echo "  env: $*"
  env "$@" MEMBENCH="$dir" BENCH_REPS="$REPS" PYTHONPATH=. \
    uv run python eval.py \
      checkpoint_dir="$CKPT" \
      '~eval_set@evals=pretraining' \
      +eval/tasks@evals.msmarco_c512=gen_large_mem_msmarco_c512 \
      evals.msmarco_c512.eval.num_samples=16 \
      ${ARM_OVERRIDES:-} \
      tp_devices=1 use_wandb=false \
      hydra.run.dir="$dir/hydra" 2>&1 \
    | grep -viE "^Generating|it/s\]" | tail -25
  echo "--- bench_membed.json ---"
  cat "$dir/bench_membed.json" 2>/dev/null || echo "  (no bench file — arm failed)"
}

# A: exactly what the live eval box runs today (bank sharded, mem_v on CPU, bs=8).
ARM_OVERRIDES=""
run_arm baseline MEM_APPROX_TOPK=1

# B: replicate the bank on-device (kills the per-decode-step host callback) + bs=16.
ARM_OVERRIDES="evals.msmarco_c512.dataset.batch_size=16"
run_arm replicated_bs16 MEM_REPLICATE_BANK=1 MEM_APPROX_TOPK=1

echo
echo "############ SUMMARY (tokens/s = B * max_new_tokens / gen_e2e_s) ############"
uv run python - <<'PY'
import json, os, glob
for name in ("baseline", "replicated_bs16"):
    p = os.path.expanduser(f"~/bench_c512/{name}/bench_membed.json")
    if not os.path.exists(p):
        print(f"{name:16s} (no data)"); continue
    b = json.load(open(p)).get("batches", [])
    if not b:
        print(f"{name:16s} (no batches)"); continue
    B = b[0]["B"]; mnt = b[0]["max_new_tokens"]
    gen = sorted(x["gen_e2e_s"] for x in b)
    med = gen[len(gen)//2]
    print(f"{name:16s} B={B:<3d} max_new_tokens={mnt:<4d} gen_e2e={med:7.2f}s  "
          f"-> {B*mnt/med:7.1f} tok/s aggregate, {mnt/med:6.2f} tok/s/seq, "
          f"{med/mnt*1000:6.1f} ms/decode-step, {med/B:6.2f} s/sample")
PY
