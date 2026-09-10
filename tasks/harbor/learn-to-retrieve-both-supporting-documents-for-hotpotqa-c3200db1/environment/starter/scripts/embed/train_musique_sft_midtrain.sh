# MuSiQue-only midtraining run: fine-tune the hard-neg-think 4B checkpoint on the MuSiQue
# multi-hop SFT set (ragrawal36/musique-sft, 13,153 rows with gold-labelled hard negatives).
# See wiki/implementations/2026-07-18-musique-sft-dataset.md for how the dataset was built.
#
# TARGET BOX: a v6e-8 (8 chips). The checkpoint was trained on one, and the batch-32 config below
# will not fit a v6e-4.
#
# Default: a flex-start (DWS) box — a Compute Engine VM with an attached TPU, in project
# `memory-layers`. It is invisible to the Cloud TPU API, so it needs TRANSPORT=gce; the default
# tpu-vm ssh path returns NOT_FOUND. Provisioning recipe:
# wiki/infrastructure/experiment-launch-instructions.md §2.2.
#     TPU_NAME=tpu-v5p-4-uscentral1a ZONE=us-central1-a PROJECT_ID=memory-layers TRANSPORT=gce \
#     RUN_SCRIPT_PATH=scripts/embed/train_musique_sft_midtrain.sh \
#       bash scripts/infrastructure/multi-vm-tpu-run.sh
#
# v6e-8 flex capacity has never been granted in europe-west4-a (3 attempts, 3 durations), so the
# working box is a 4-chip v5p in us-central1-a. That is FINE for memory — v5p is ~95 GB HBM/chip
# vs v6e's ~31 GB, so 4 v5p chips give ~380 GB against a v6e-8's ~248 GB — but it is roughly 4x
# slower in bf16 FLOPs, so expect the run to take proportionally longer.
#
# On a TRC/tpunanny box instead (§2.1) it is a Cloud TPU node, so drop TRANSPORT and switch
# project:  TPU_NAME=rohun-v6e-8-0 PROJECT_ID=memorylayers ZONE=europe-west4-a ...
#
# ⚠️ A flex box self-deletes at --max-run-duration, boot disk included. The staged
# $GROUND_HF_PARQUET cache and the venv go with it, so re-run the download step on a fresh box.
# Checkpoints are unaffected — they are written to gs:// as the run goes.
#
# CHECKPOINT I/O IS REGION-SENSITIVE. .env's GCS_BUCKET (memory-layers-training) is a REGIONAL
# bucket in EUROPE-WEST4. Running from us-central1 against it means reading ~20 GiB across the
# Atlantic at warm start and writing every checkpoint back the same way — the write latency lands
# inside the training loop, not just on the invoice. So both the warm-start source and the
# checkpoint destination are pointed at an in-region copy. Set CKPT_BUCKET=memory-layers-training
# CKPT_REGION=europe-west4 to go back to the European bucket (correct for a europe-west4 box).
CKPT_BUCKET="${CKPT_BUCKET:-memory-layers-training-usc1}"
CKPT_REGION="${CKPT_REGION:-us-central1}"
# Forced, not defaulted: setup_shell.sh has already exported .env's GCS_BUCKET by this point, so
# `${GCS_BUCKET:-...}` would silently keep the European one.
export GCS_BUCKET="$CKPT_BUCKET"
export GCS_REGION="$CKPT_REGION"
#
# PREREQUISITE — stage the parquets first, on the box:
#     STEPS=1250 BATCH_SIZE=32 bash scripts/misc/download_musique_sft_data.sh
# BOTH env vars below are required. GROUND_HF_PARQUET alone does NOTHING (data/qa.py:313 gates
# the offline branch on HF_HUB_OFFLINE == "1"), and live-HF is a livelock, not a fallback
# (wiki/data/hf-rate-limits.md).
#
# ─────────────────────────────────────────────────────────────────────────────────────────
# WARM START, NOT RESUME. RESUME_FROM ends in "/100000" and that trailing step is LOAD-BEARING:
# utils.py::setup_checkpointing parses a trailing digit group into resume_step, and
# load_checkpoint then takes the "Warm start: only restore weights, keep fresh optimizer state
# and step=0" branch. Drop the "/100000" and it takes the OTHER branch — full resume of weights
# + optimizer + dataloader position at step 100000 — which, with trainer.steps=2500, would exit
# immediately having trained nothing.
RESUME_FROM="${RESUME_FROM:-gs://${CKPT_BUCKET}/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09/qwen3_mem_embed/100000}"
#
# trainer=midtraining_telemetry: single stage, cosine, unfreezing mem + embed_model + main
# layers 13/14/15 (the memory layer is 14, so this is the memory layer and its neighbours).
# NOT `staged`/`staged_telemetry`: that recipe's stage 0 re-freezes the main model and sets
# ce_weight=0 for 10k steps, which is a warmup schedule for a FRESH model. Applying it to an
# already-trained checkpoint at step 0 would undo the thing we are starting from.
#
# learning_rate=1e-4 (the `standard` default) — chosen deliberately, against the more cautious
# 1e-5. For contrast, staged_sim.yaml lowers stage-3 peak LR to 1e-5 "to protect the 4B" when the
# main model is trainable, and this recipe does unfreeze main layers 13/14/15. The risk at 1e-4 is
# drift in the 4B over ~3 epochs on 13k rows; the out-of-domain tasks (msmarco/hotpotqa) in
# hard_neg_think_c512 are where that would show up first. Drop to 1e-5 if those regress.
#
# warmup_frac=0.1 (250 steps) is added because `midtraining` ships with none. A warm start
# restores weights but NOT the optimizer — Adam's moments begin at zero, so the first updates are
# effectively unscaled and land on a converged 4B. Every `staged` stage uses 0.1 for the same
# reason. Drop it only if you want the LR at peak from step 1.
#
# batch_size=32 (the dataset config's default is 16), so steps=1250 to hold ~3 epochs:
# 1250 x 32 = 40,000 samples over 13,153 rows = 3.04 epochs, the same sample budget as the
# batch-16 x 2500-step version. Steps and batch move together here — leaving steps at 2500 with
# batch 32 would be 6.1 epochs, which on a 13k-row set is memorisation. Adjust both or neither.
#
# ⚠️ MEMORY: batch 32 x 20 chunks x 256 = 163,840 bank slots at seq_len 1024, vs 65,536 at
# seq_len 512 / batch 16 for the run that produced the checkpoint — 2.5x the bank and 2x the
# sequence. This needs a v6e-8 and may still OOM. Fallback ladder, cheapest first:
#   dataset.batch_size=16 trainer.steps=2500     (same sample budget)
#   dataset.num_chunks_per_doc=16                (drops 4 distractors/row; gold always survives,
#                                                 since pack_docs packs positives first)
#
# model.memory.mem_top_k=64 MUST match the checkpoint (.hydra/config.yaml has mem_top_k: 64;
# the config default is 128). Every other memory/embed field already matches the default.
#
# seq_len=1024 comes from configs/dataset/musique_sft.yaml and must match the budget the CoT was
# generated at (think+answer <= 950). Lowering it does NOT shorten the data — it makes
# _StreamingQAFilter silently DROP every over-length row (~35%, disproportionately 3/4-hop).
#
# EVALS: in-loop eval is off, same rationale as train_hard_neg_think.sh — llm_judge_accuracy is
# computed in eval.py's parent process after the JAX worker frees the TPU, so a task's `metrics:`
# block is inert in-loop. Score checkpoints from a SEPARATE eval box.
#   ⚠️ MuSiQue evals are now IN-DOMAIN for this run. eval_set=hard_neg_think_c512 includes
#   musique@512; its numbers are no longer measuring zero-shot generalisation and are not
#   comparable to the same metric on the pre-midtraining checkpoint. msmarco/hotpotqa in that
#   suite stay out-of-domain and are the honest read on whether this helped or hurt.
#
# checkpoint_interval=250 + max_to_keep=12 => the last 3000 steps stay evaluable, i.e. the whole
# run. These two must be set together (orbax rotates all but max_to_keep).
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet_musique}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    dataset=musique_sft \
    dataset.batch_size=32 \
    trainer=midtraining_telemetry \
    trainer.resume_from="$RESUME_FROM" \
    trainer.steps=1250 \
    trainer.learning_rate=1e-4 \
    +trainer.training_stages.0.warmup_frac=0.1 \
    trainer.checkpoint_interval=250 \
    trainer.max_to_keep=12 \
    trainer.log_interval=10 \
    trainer.eval_interval=1000000000 \
    'eval_set@trainer.evals=none' \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="musique_sft_midtrain_topk64_seq1024_chunks20_bs32"
