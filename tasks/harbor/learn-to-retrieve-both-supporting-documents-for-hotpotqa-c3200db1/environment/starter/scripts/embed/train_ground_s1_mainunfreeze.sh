#!/usr/bin/env bash
# Resume ground_s1_zeroinit_4layer @ 38000 with the MAIN 4B UNFROZEN (stage C, LR 1e-5).
#
# Launch (multi-host v6e-8 flex slice => run on EVERY worker; runbook §2.3):
#     TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers \
#     bash scripts/infrastructure/multi-tpu-box-run.sh \
#       tpu-v6e-slice-mig-1wjb=scripts/embed/train_ground_s1_mainunfreeze.sh \
#       tpu-v6e-slice-mig-1z9d=scripts/embed/train_ground_s1_mainunfreeze.sh
#
# SMOKE=1 -> stop 30 steps past the resume point, no wandb. Do this FIRST (see checklist below).
#
# PREREQUISITE — full parquet staged on EACH host (both disks are independent):
#     GROUND_DATA_FRAC=1.0 bash scripts/misc/precache_hf.sh
# FRAC=1.0 is deliberate: the original run streamed live HF, i.e. the WHOLE dataset. The default
# 0.5 would stage a different shard set, which changes the deterministic stream order and so
# silently invalidates the saved dataloader position (see "resume caveat" below).
#
# ─────────────────────────────────────────────────────────────────────────────────────────
# ⚠️ THE model.memory.* OVERRIDES ARE REQUIRED, NOT COSMETIC. train.py builds the model from the
# Hydra config and only then restores weights with partial_restore=True — so a 4-layer checkpoint
# loaded under the default 1-layer config would SILENTLY take layer 14 and drop 9/20/27, with no
# error and a plausible loss curve. This checkpoint's architecture (from its .hydra/config.yaml):
#     mem_layers [9,14,20,27]   mem_top_k 128   mem_o_proj_zero_init true
#
# RESUME_FROM has NO trailing step — that is load-bearing. utils.py::setup_checkpointing parses a
# trailing "/38000" into resume_step and takes the weights-only branch (fresh optimizer, step 0,
# NO dataloader restore). The bare model dir gives a FULL resume: weights + optimizer + dataloader
# position, continuing at 38000.
CKPT_BUCKET="${CKPT_BUCKET:-memory-layers-training}"
RESUME_FROM="${RESUME_FROM:-gs://${CKPT_BUCKET}/ground_s1_zeroinit_4layer-2026-07-04-09-42-55/qwen3_mem_embed}"
# The bucket is EUROPE-WEST4 and this slice is europe-west4-a — same region, so no cross-region
# checkpoint traffic. (Contrast the MuSiQue runs, which forced a us-central1 bucket for a us box.)
#
# ⚠️ RESUME CAVEAT — verify, don't assume. The dataloader state is a per-worker raw-item COUNT into
# the deterministic shuffled/interleaved stream; the seed itself is not in it (shuffle_seed 42 comes
# from the dataset config, and matches). Exactness therefore needs the same shard set AND the same
# stream order, and the original run streamed live-HF while this one reads local parquet — an
# equivalence that is NOT verified (runbook §5 flags resume as untested under offline parquet).
# Restore is warn-only on both sides: if it fails you get "stream starts from 0" and training
# quietly restarts the data, which is a correctness issue, not a crash. CHECK THE LOG for
# "Restored dataloader state" before walking away.
#
# MEM_DONATE is deliberately left at its default 'none'. Donation is a known NaN trigger and the
# hazard only fires ONCE main_model IS TRAINABLE — exactly this run's regime. See the header of
# trainer/trainer.py and wiki/implementations/2026-07-17-stage3-grad-nan-donation.md.
#
# SMOKE CHECKLIST — what the 30-step run must show before committing days of compute:
#   1. "Restored dataloader state from …/38000/dataloader_state.json"  (not "stream starts from 0")
#   2. "=== Transitioning to Stage 2 …" and a trainable list CONTAINING main_model weights
#   3. no NaN warnings (main-trainable is the documented NaN regime)
#   4. train/ce_loss near its 38000 value and NOT climbing
#   5. it fits HBM at all — stage C adds Adam mu/nu for 4B params that stages A/B never allocated
#   6. ms/step, to size the real run against the box's 7-day lifetime
# ── Fresh-stream resume (the replay is not affordable; runbook §5) ───────────────────────────
# SKIP_LOADER_RESTORE=1 resumes weights+optimizer at 38000 but starts the data stream at 0
# instead of replaying ~13M items to the saved cursor (measured: >2 h, no first batch, headed
# for OOM). SHUFFLE_SEED must then DIFFER from the original run's 42, or "start from 0" means
# re-showing exactly the samples this checkpoint already trained on, in the same order.
#
# How good is the approximation? The original run consumed only ~10% of one epoch, so a reseed
# genuinely resamples. Working (the state file is easy to misread — do NOT sum the 16 cursors):
#   each grain worker holds its OWN iterator over the FULL stream and takes a strided slice of it
#   (data/qa.py::_StreamingIterator — `_count` increments on EVERY item pulled, including ones the
#   stride skips). So the 16 counts are 16 positions in ONE shared stream, not disjoint shares.
#   stream position   ~800k source items  (the cursors, 670k-960k; they drift by worker speed)
#   emitted samples    608,000            (38000 steps x batch 16) -> filter drops ~24%, not 20:1
#   staged rows      8,105,497            -> ~10% of one epoch consumed
# Expected overlap between the new seed's draw and what this checkpoint already saw is therefore
# ~10%. (Note ~800k x 16 workers ~= 13M item-READS is the REPLAY cost — work, not unique data.)
export SKIP_LOADER_RESTORE="${SKIP_LOADER_RESTORE:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-20260720}"   # original run used 42; any value != 42 works
# num_workers 16 -> 8: each grain worker holds its own pipeline + a 100k-item shuffle buffer,
# measured at ~45 GB RSS. 16 x 45 = 720 GB exceeds this host's 708 GB (the TRC v6e-8 box this
# recipe was written for has ~1.4 TB, which is why 16 was fine there). 8 x 45 ~ 360 GB leaves
# headroom. Data was never the bottleneck at ~1-2 s/step, so fewer workers should not cost
# throughput — but WATCH the first steps' timing to confirm.
NUM_WORKERS="${NUM_WORKERS:-8}"
# ⚠️ THE UNFROZEN 4B DOES NOT FIT A v6e AT ANY BATCH SIZE. Measured 2026-07-20, stage C:
#     batch 16 -> RESOURCE_EXHAUSTED: allocate 47.50M, 36.65M free
#     batch  8 -> RESOURCE_EXHAUSTED: allocate 47.50M, 37.26M free   (halving bought 0.6 MB)
# Batch controls ACTIVATIONS, which are not what is exhausting HBM. The binding term is
# batch-independent and, at tp_devices=1, REPLICATED ON EVERY CHIP:
#     4B params bf16 ~8 GB + grads ~8 GB + Adam mu&nu bf16 ~16 GB  ~=  32 GB PER CHIP
# (mu/nu are bf16, NOT fp32: weights are created bfloat16 and utils.py calls optax.adamw with no
# mu_dtype, so scale_by_adam's tree_zeros_like inherits the param dtype. "Make the optimizer state
# bf16" is not an available saving — it already is.)
# Stages A/B fit easily only because a frozen main carries no optimizer state at all.
#
# So the constraint is PER-CHIP HBM, not the box total (an earlier note here said total — wrong):
#     v6e  ~31 GB/chip  -> ~32 GB replicated is JUST over; it died short by ~47 MB, not by GBs
#     v5p  ~95 GB/chip  -> fits with room, which is why the full-unfreeze ran at batch 16 on v5p-4
# Because the v6e gap is ~1 GB, tp_devices=2 is the cheap fix: sharding two ways puts the per-chip
# cost near ~16 GB and an 8-chip slice still leaves 4-way data parallelism. LoRA (optimizer state
# for adapters only) or a v5p also work. Adding chips at tp_devices=1 does nothing.
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-150000}"
RUN_NAME="${RUN_NAME:-ground_s1_zeroinit_4layer_mainunfreeze}"
EXTRA=()

# LORA=1 — give the main 4B LoRA adapters instead of a full unfreeze. This is the way to fit a
# v6e: adamw allocates mu/nu only for the *matched* params, so adapters cost ~MBs of optimizer
# state instead of the ~16 GB/chip that full-unfreeze moments need (see wiki/training/optimizer.md).
# Adapters on the MLP only (gate/up/down_proj), matching configs/model/qwen3_mem_embed_copyinit_lora.yaml.
# LoRA B is zero-init so the branch starts as an exact no-op — the model at step 38000 is unchanged
# on the first step, which is what makes this a clean continuation rather than a perturbation.
# NOTE this is NOT the same experiment as a full unfreeze: capacity is confined to a rank-R update
# of the MLPs, attention stays frozen. Report it as LoRA, never as "main unfrozen".
# The 38000 checkpoint has no adapter weights; partial_restore=True initialises them fresh.
MODEL_CFG="qwen3_mem_embed"
if [ "${LORA:-0}" = "1" ]; then
  # Switch the whole model config, don't override into it: the base `main_model` has no `lora`
  # key, so `model.main_model.lora.enabled=true` dies with
  # "ConfigKeyError: Key 'lora' is not in struct". qwen3_mem_embed_mainlora adds the block.
  MODEL_CFG="qwen3_mem_embed_mainlora"
  RUN_NAME="${RUN_NAME/mainunfreeze/mainlora}"
  EXTRA+=(
    # Stage C trains adapters + mem/embed, NOT the 4B base.
    "trainer.training_stages.2.trainable_params=['.*mem_.*','.*embed_model.*','.*value_model.*','.*main_model.*_a_proj','.*main_model.*_b_proj']"
  )
  echo "[lora] model=$MODEL_CFG (rank/alpha from that config); base 4B frozen; run=$RUN_NAME"
fi
if [ "${SMOKE:-0}" = "1" ]; then
  STEPS="${SMOKE_STEPS:-38030}"    # default 30 steps past the 38000 resume point
                                   # SMOKE_STEPS=38005 for the quickest possible pass (5 steps:
                                   # enough for HBM fit + NaN + a steady-state ms/step reading)
  RUN_NAME="smoke_${RUN_NAME}"
  EXTRA+=(trainer.use_wandb=false) # don't pollute the project with smoke runs
  echo "[smoke] steps=$STEPS run_name=$RUN_NAME (checkpoint_interval 2000 > 30, so no ckpt write)"
fi

HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model="$MODEL_CFG" \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    model.memory.mem_o_proj_zero_init=true \
    'model.memory.mem_layers=[9,14,20,27]' \
    dataset=qa_hard_neg_think_sft4b \
    dataset.shuffle_seed="$SHUFFLE_SEED" \
    dataset.num_workers="$NUM_WORKERS" \
    dataset.batch_size="$BATCH_SIZE" \
    trainer=staged_ground_mainunfreeze \
    trainer.resume_from="$RESUME_FROM" \
    trainer.steps="$STEPS" \
    trainer.checkpoint_interval=2000 \
    trainer.log_interval=10 \
    'eval_set@trainer.evals=none' \
    +trainer.run_name="$RUN_NAME" \
    "${EXTRA[@]}"
