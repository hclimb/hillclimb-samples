#!/bin/bash
# Runs ON a box. Why does the 512-doc corpus eval cost ~83 s/batch when _gen() benches at 6.42 s?
#
# ESTABLISHED SO FAR (scripts/embed/bench_c512_gen.sh + an earlier aux_on/aux_off arm):
#   _gen()                     6.42 s/batch   (steady-state, JIT excluded)
#   prompt prefill (:822)      0.32 s/batch
#   numpy + allgathers         ~0             (doc_access_acc=false removes them; no speedup)
#   => ~76 s/batch is the SECOND aux forward at gen_large_mem.py:849 (`aux_gen`), by subtraction.
# NOTE doc_access_acc=false only nulls pos_sets (:797); BOTH aux forwards (:822, :849) still run,
# which is why that A/B showed nothing. aux_gen cannot simply be gated away either: in production
# doc_access_acc=true, so pos_sets IS set and its consumer at :852 needs it.
#
# So attack WHY aux_gen is expensive. It is one forward over [prompt|512 gen] with collect_aux=True:
#   * mem_collect_full_scores=true -> materialises scores for every position against ALL 131,072
#     slots (the corpus path only reads mem_top_k_indices, so the full matrix may be dead weight)
#   * mem_v on CPU -> its value read goes through jax.pure_callback for ~1.1M rows in ONE shot.
#     (_gen()'s decode steps each fetch only top-64, which is why MEM_REPLICATE_BANK barely moved
#     the generation bench but could matter enormously here.)
#
# Metric: the pbar's s/it (tqdm counts SAMPLES; x8 = per batch) + total wall-clock. Same checkpoint,
# same task, same n — one variable per arm.
#
#   CKPT=gs://.../qwen3_mem_embed/16000 bash scripts/embed/bench_c512_loop.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT

CKPT="${CKPT:?set CKPT=gs://.../qwen3_mem_embed/<step>}"
N="${N_SAMPLES:-32}"

# doc_access_acc stays ON in every arm: that is the production config (pos_sets set -> aux_gen's
# output is consumed), and it keeps the arms comparable.
AUX="+aux_losses.mem_pos_weight_mass.enabled=true +aux_losses.mem_pos_weight_mass.weight=0.0 \
+aux_losses.doc_access_acc.enabled=true +aux_losses.doc_access_acc.weight=0.0"

run_arm () {
  local name="$1"; shift
  local env_kv="$1"; shift
  local out=$HOME/loopbench/$name
  rm -rf "$out"; mkdir -p "$out"
  echo
  echo "############ ARM: $name ############"
  echo "  env: ${env_kv:-<none>}   overrides: $*"
  local t0=$(date +%s)
  env $env_kv PYTHONPATH=. uv run python eval.py \
      checkpoint_dir="$CKPT" \
      '~eval_set@evals=pretraining' \
      +eval/tasks@evals.msmarco_c512=gen_large_mem_msmarco_c512 \
      evals.msmarco_c512.eval.num_samples="$N" \
      $AUX "$@" \
      tp_devices=1 use_wandb=false hydra.run.dir="$out/hydra" 2>&1 \
    | tr '\r' '\n' | grep -iE "Generating \(large mem\)|mem_v on CPU|REPLICATED|mem_k sharded" | tail -3
  local t1=$(date +%s)
  echo "  ARM $name total wall-clock: $((t1-t0))s"
}

# A: exactly what the eval box runs today.
run_arm baseline ""

# B: bank replicated on-device -> no jax.pure_callback for mem_v. Bank is ~0.54 GB vs ~24 GB free.
run_arm replicate_bank "MEM_REPLICATE_BANK=1"

# C: don't materialise the full [B,N,T,M] score matrix. The corpus path reads mem_top_k_indices,
# not mem_scores, so this may be pure waste at 131k slots x 531 positions.
run_arm no_full_scores "" model.memory.mem_collect_full_scores=false

# D: both.
run_arm replicate_no_scores "MEM_REPLICATE_BANK=1" model.memory.mem_collect_full_scores=false

echo
echo "############ READ ############"
echo "baseline should reproduce ~8-10 s/it (~83 s/batch). Any arm that collapses toward"
echo "~1 s/it (~7 s/batch = _gen + prefill) identifies aux_gen's real cost."
