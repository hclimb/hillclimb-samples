#!/usr/bin/env bash
# Stage-1 grounding + DOC-CODE + slim heads: the ground_s1_zeroinit_4layer recipe
# (staged_ground, qa_hard_neg_think_sft4b, frozen main, zero-init mem_o_proj, 4 layers)
# with the qwen3_mem_embed_g4l_doccode architecture (k=128, v=256, doc_code=128 — see the
# model config header). Motivated by the 2026-07-22 hopping/grounding findings: give
# value-position slots a subject binding via a per-chunk identity code in the bank key.
#
#   [SMOKE=1] RUN_START_TIME=... bash scripts/embed/train_ground_s1_doccode.sh
#   SMOKE=1 -> 600 steps, checkpoint every 200 (validates the save path early — the
#   2026-07-22 spec64 smoke OOM'd at its FIRST save on this slice; see the
#   FAILED_PRECONDITION note's second addendum).
#
# MEM_MASKED_OPTIMIZER tried and REVERTED (2026-07-22): optax.masked crashes in real
# training — TypeError ("can't multiply sequence by non-int") inside adamw's update_moment
# on MaskedNode leaves; the optimizer-moment note's "incomplete" is load-bearing. Running
# with full moments; the slimmer arch (k=128/v=256 vs spec64's 1024s) may clear the ~39 MB
# save-time shortfall that killed spec64 — the SMOKE's step-400 save answers this early.
# tp_devices=2, batch_size at recipe default 16 (see SINGLE_HOST note above for the
# save-OOM history: bs and tp sweeps proved the multi-host save can never fit).

# SINGLE_HOST=1 (2026-07-22, smoke v8): the multi-host checkpoint save is architecturally
# oversized for v6e — save_checkpoint's unshard replicates the FULL state per chip
# (weights+moments ~24 GB transient, async-enqueued), so it OOMs at ANY tp on 31 GB chips
# (v4 tp=2: 47M free; v7 tp=4: 1.5M free — more sharding just queues more gathers before
# dying). Single-host runs never take the unshard path — all July ground_s1-family runs
# were single-host and saved fine. Run on ONE worker's 4 chips (standalone-TPU env, same
# trick as eval_msa_rag.sh); ~2x wall clock accepted until a sharded save lands.
if [ "${SINGLE_HOST:-0}" = "1" ]; then
  export TPU_SKIP_MDS_QUERY=1
  export TPU_PROCESS_BOUNDS=1,1,1
  export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
  export JAX_FORCE_SINGLE_HOST=1   # utils.py::init_jax_distributed override (metadata still advertises the slice)
  # tp=1 (v10): save_checkpoint's unshard gathers ANY sharded state — tp=2 single-host
  # still OOM'd at the post-stage-flip save (886K free, v9). July's runs saved fine because
  # they ran tp=1: everything already replicated, no gather spike. Frozen 4B at tp=1 =
  # 8G weights + 16G moments + bs16/data=4 activations (~4 rows/chip) ~= 27G < 31G. Full
  # recipe fidelity restored (bs=16, global batch matches control).
  # bs=8 (v11): tp=1/bs=16 compile-OOMs by 627M — July's v6e-8 ran 8 chips = 2 rows/chip;
  # this 4-chip host at bs=16 is 4 rows/chip (~2G activations/row). bs=8 = July's exact
  # per-chip load, ~3.5G margin. RECIPE DEVIATION: global batch 8 vs control's 16.
  TP=1
  BS=8
else
  TP=2
  BS=16
fi

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  # >= stage-0 boundary (1000) or parse_training_stages rejects; 1200 also exercises the
  # stage-0 -> stage-1 flip. Saves at 400/800/1200 validate the save path early.
  STEPS=1200; CKPT_INT=400; RUN_NAME="ground_s1_doccode_4layer_smoke"
else
  STEPS=40000; CKPT_INT=2000; RUN_NAME="ground_s1_doccode_4layer"
fi

HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed_g4l_doccode \
    dataset=qa_hard_neg_think_sft4b \
    dataset.batch_size=$BS \
    dataset.num_workers=0 \
    trainer=staged_ground \
    trainer.steps=$STEPS \
    trainer.checkpoint_interval=$CKPT_INT \
    trainer.tp_devices=$TP \
    trainer.log_interval=10 \
    trainer.eval_interval=1000000000 \
    'eval_set@trainer.evals=none' \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="$RUN_NAME" \
    ${RESUME_FROM:+ trainer.resume_from="$RESUME_FROM"}
