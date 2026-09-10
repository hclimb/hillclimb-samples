# MuSiQue midtraining on the GROUNDING 4-layer checkpoint
# (ground_s1_zeroinit_4layer-2026-07-04-09-42-55 @ 38000), as opposed to the hard-neg-think
# checkpoint used by scripts/embed/train_musique_sft_midtrain.sh. Same dataset, different model.
#
# Launch (flex-start v5p is GCE-attached => TRANSPORT=gce; runbook §2.2):
#     TPU_NAME=tpu-v5p-4-uscentral1a ZONE=us-central1-a PROJECT_ID=memory-layers TRANSPORT=gce \
#     RUN_SCRIPT_PATH=scripts/embed/train_musique_ground4layer_midtrain.sh \
#       bash scripts/infrastructure/multi-vm-tpu-run.sh
#
# PREREQUISITE — stage the parquets first, on the box:
#     STEPS=1500 BATCH_SIZE=16 bash scripts/misc/download_musique_sft_data.sh
#
# ─────────────────────────────────────────────────────────────────────────────────────────
# ⚠️ THE model.memory.* OVERRIDES BELOW ARE REQUIRED, NOT COSMETIC.
# eval.py resolves the checkpoint's saved .hydra/config.yaml and makes it authoritative
# (evals/shared.py::apply_checkpoint_model_cfg). train.py does NOT — it builds the model from the
# Hydra config via get_model(cfg.model, ...) and only then restores weights, with
# partial_restore=True. So a 4-layer checkpoint loaded into the default 1-layer config would
# SILENTLY take only the layer-14 weights and drop layers 9/20/27, with no error and a plausible
# loss curve. This checkpoint's architecture (from its .hydra/config.yaml):
#     mem_layers [9,14,20,27]   mem_top_k 128   mem_o_proj_zero_init true
#
# mem_approx_topk=true is a DELIBERATE departure: this checkpoint predates the option and trained
# with exact top-k. Approx is ~1.6x faster at ~98% recall@k
# (wiki/experiments/2026-07-15-approx-topk-training.md). Note it is also the prime suspect for
# eval non-reproducibility — 44/128 greedy generations diverged between runs — so set it false for
# any comparison that turns on a small gap.
CKPT_BUCKET="${CKPT_BUCKET:-memory-layers-training-usc1}"
CKPT_REGION="${CKPT_REGION:-us-central1}"
# Forced over .env's EUROPE-WEST4 bucket: this box is us-central1, and setup_shell.sh has already
# exported the European value by the time this runs, so `${GCS_BUCKET:-...}` would keep it.
export GCS_BUCKET="$CKPT_BUCKET"
export GCS_REGION="$CKPT_REGION"
#
# WARM START — the trailing "/38000" is load-bearing (utils.py::setup_checkpointing parses it into
# resume_step, selecting the weights-only/fresh-optimizer/step-0 branch). 38000 is the LATEST
# surviving checkpoint: the run was configured for 150k steps but only 32000-38000 remain after
# orbax's max_to_keep rotation.
RESUME_FROM="${RESUME_FROM:-gs://${CKPT_BUCKET}/ground_s1_zeroinit_4layer-2026-07-04-09-42-55/qwen3_mem_embed/38000}"
#
# trainer=midtraining_frozen_telemetry keeps the MAIN MODEL FROZEN — mem_* + embed_model only.
# This matches the regime the checkpoint was built under: the staged_ground line never trained the
# main model in either stage, so its Qwen3-4B is pristine.
#
# A full unfreeze at this LR was tried first and REGRESSED THE LM. From the logged curve of
# musique_ground4layer_..._bs16-2026-07-19-00-18-11: train/doc_access_loss fell 2.68 -> ~0.7 and
# doc_access_acc rose 0.095 -> 0.26 (retrieval improving as intended), while train/ce_loss
# bottomed at ~0.357 near step 50 and then climbed past its own 0.560 starting value — i.e. the
# language model degraded while the memory pathway improved. The turn coincided with warmup
# ending (warmup_frac 0.1 = step 150) and LR reaching peak 1e-4 on a 4B that had never been
# trained. Zero NaNs; this was drift, not divergence.
#   -> If you want the main model trainable, drop the LR with it (staged_sim uses 1e-5 for
#      exactly this reason) rather than reusing the 1e-4 that suits a memory-only run.
# (midtraining's own "unfreeze main 13/14/15" was never an option here: that is the
# neighbourhood of a single memory layer at 14, meaningless for memory at 9/14/20/27.)
#
# batch_size 16, half the MuSiQue-on-hard-neg run's 32: four memory layers at top_k 128 is ~8x the
# retrieval work of that config (1 layer at 64), each reading a 16x20x256 = 81,920-slot bank.
# Verified to fit even with the main model unfrozen (which adds optimizer state for all 4B
# params), so it has headroom now that main is frozen — 32 is probably reachable.
#
# steps 1500 x 16 = 24,000 samples = 1.82 epochs over 13,153 rows — the point where the
# hard-neg MuSiQue run peaked before overfitting
# (wiki/experiments/2026-07-18-musique-midtraining-vs-rag.md).
#
# learning_rate 1e-4 matches that run so the comparison isolates the MODEL. With the main model
# frozen this is a safe value — the 1e-4 forgetting risk applies to the 4B, and nothing here
# updates it. The grounding run itself used 2e-4 for its own frozen-main stage.
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet_musique}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    'model.memory.mem_layers=[9,14,20,27]' \
    model.memory.mem_top_k=128 \
    model.memory.mem_o_proj_zero_init=true \
    model.memory.mem_approx_topk=true \
    dataset=musique_sft \
    dataset.batch_size=16 \
    trainer=midtraining_frozen_telemetry \
    trainer.resume_from="$RESUME_FROM" \
    trainer.steps=1500 \
    trainer.learning_rate=1e-4 \
    +trainer.training_stages.0.warmup_frac=0.1 \
    trainer.checkpoint_interval=250 \
    trainer.max_to_keep=12 \
    trainer.log_interval=10 \
    trainer.eval_interval=1000000000 \
    'eval_set@trainer.evals=none' \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="musique_ground4layer_midtrain_topk128_seq1024_chunks20_bs16"
