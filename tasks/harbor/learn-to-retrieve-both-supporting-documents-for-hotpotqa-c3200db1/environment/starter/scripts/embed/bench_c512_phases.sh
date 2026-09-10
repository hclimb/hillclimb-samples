#!/bin/bash
# Runs ON a box. Attribute the 512-doc corpus eval's per-batch wall-clock to its phases, MEASURED
# per batch at real prompt lengths — not inferred by subtraction.
#
# WHY THIS EXISTS (the previous attempts and why they failed):
#   * bench_c512_gen.sh timed _gen() at 6.42 s/batch, but MEMBENCH break()s after batch 0, and with
#     shuffle=false batch 0 has the SHORTEST prompt (prompt_len=19) -> that number is the cheapest
#     batch in the run, not a representative one.
#   * an aux_on/aux_off A/B showed nothing — because doc_access_acc=false only nulls pos_sets
#     (:797); BOTH aux forwards (:822, :849) still run.
#   * MEM_REPLICATE_BANK / mem_collect_full_scores=false both made it SLOWER, not faster.
#   * run-to-run variance is ~30% (an identical config measured 6.60 and 8.57 s/it), so small
#     deltas between arms are noise.
# MEMBENCH_PHASES=1 times gen / aux_prefill / aux_gen / other for EVERY batch and prints a
# steady-state median (batch 0 excluded — it carries JIT).
#
#   CKPT=gs://.../qwen3_mem_embed/16000 bash scripts/embed/bench_c512_phases.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json
export GCLOUD_PROJECT=$GCS_BUCKET_PROJECT

CKPT="${CKPT:?set CKPT=gs://.../qwen3_mem_embed/<step>}"
N="${N_SAMPLES:-48}"          # 6 batches at bs=8: enough steady-state batches for a median
OUT=$HOME/phasebench; rm -rf "$OUT"; mkdir -p "$OUT"

# Production config: doc_access_acc ON (pos_sets set -> aux_gen's output is consumed).
MEMBENCH="$OUT" MEMBENCH_PHASES=1 PYTHONPATH=. uv run python eval.py \
    checkpoint_dir="$CKPT" \
    '~eval_set@evals=pretraining' \
    +eval/tasks@evals.msmarco_c512=gen_large_mem_msmarco_c512 \
    evals.msmarco_c512.eval.num_samples="$N" \
    +aux_losses.mem_pos_weight_mass.enabled=true +aux_losses.mem_pos_weight_mass.weight=0.0 \
    +aux_losses.doc_access_acc.enabled=true +aux_losses.doc_access_acc.weight=0.0 \
    tp_devices=1 use_wandb=false hydra.run.dir="$OUT/hydra" 2>&1 \
  | tr '\r' '\n' | grep -E "^\[PHASES\]|prompt_len|median|^    (gen_s|aux_|other_s|batch_total)"
