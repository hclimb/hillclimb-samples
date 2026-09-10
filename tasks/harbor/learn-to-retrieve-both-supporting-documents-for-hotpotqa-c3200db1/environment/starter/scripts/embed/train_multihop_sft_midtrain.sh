# Midtraining on ragrawal36/multihop_qa_sft (1.34M synthetic multi-hop QA rows with CoT),
# warm-started from the same hard-neg-think 4B checkpoint the MuSiQue run used — so the two
# midtraining datasets are directly comparable against one shared baseline.
# See wiki/experiments/2026-07-18-musique-midtraining-vs-rag.md for that run's results.
#
# TARGET BOX / launch (flex-start v5p, GCE-attached => TRANSPORT=gce; §2.2 of the launch runbook):
#     TPU_NAME=tpu-v5p-4-uscentral1a ZONE=us-central1-a PROJECT_ID=memory-layers TRANSPORT=gce \
#     RUN_SCRIPT_PATH=scripts/embed/train_multihop_sft_midtrain.sh \
#       bash scripts/infrastructure/multi-vm-tpu-run.sh
#
# PREREQUISITE — stage the parquets first, on the box:
#     STEPS=10000 BATCH_SIZE=32 GROUND_HF_PARQUET=$HOME/hf_parquet_multihop \
#       bash scripts/misc/download_multihop_sft_data.sh
#
# ─────────────────────────────────────────────────────────────────────────────────────────
# THIS IS A DIFFERENT REGIME FROM THE MUSIQUE RUN. That dataset had 13,153 rows, so 1250 steps
# was 3.04 epochs and the run overfit — accuracy peaked at ~1.8 epochs and fell after. Here one
# epoch is 1,341,045/32 = 41,907 steps, so 10,000 steps is 0.24 EPOCHS. Repetition is not the
# constraint; compute is. Expect the failure mode to be under-training, not memorisation.
CKPT_BUCKET="${CKPT_BUCKET:-memory-layers-training-usc1}"
CKPT_REGION="${CKPT_REGION:-us-central1}"
# Forced over .env's value: setup_shell.sh has already exported the EUROPE-WEST4 bucket by now,
# and a us-central1 box writing checkpoints there pays a transatlantic round trip per save.
export GCS_BUCKET="$CKPT_BUCKET"
export GCS_REGION="$CKPT_REGION"
#
# WARM START, NOT RESUME — the trailing "/100000" is load-bearing. utils.py::setup_checkpointing
# parses a trailing digit group into resume_step, which selects load_checkpoint's "restore weights
# only, fresh optimizer, step 0" branch. Without it you get a full resume at step 100000 and the
# run exits immediately against trainer.steps.
RESUME_FROM="${RESUME_FROM:-gs://${CKPT_BUCKET}/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09/qwen3_mem_embed/100000}"
#
# seq_len 512 (the dataset config's value), NOT the 1024 the MuSiQue run needed. This data was
# generated with MAX_THINK_ANS_TOKENS=512 (datagen/generate_multihop_sft.py) and its think+answer
# is ~283 tokens at the median, so most rows fit comfortably; _StreamingQAFilter drops the long
# tail (think+answer + ~50-token prefix > 512).
#   ⚠️ That tail-drop is a deliberate trade, and the opposite call from MuSiQue. There, 13k rows
#   were scarce and losing ~35% was unacceptable, so seq_len went to 1024. Here rows are
#   effectively unlimited (we consume 0.24 of one epoch), and seq_len 512 gives 2x the samples
#   per unit compute with a median sequence that is only ~33% padding instead of ~67%. The cost
#   is a bias against long-reasoning examples — set dataset.seq_len=1024 to keep them.
#
# steps=10000 x batch 32 = 320,000 samples = 0.24 epochs. checkpoint_interval=1000 with
# max_to_keep=12 keeps the last 12,000 steps evaluable, i.e. the whole run, so the accuracy-vs-
# steps curve can be reconstructed afterwards the way it was for MuSiQue.
#
# learning_rate 1e-4 and warmup_frac 0.1 match the MuSiQue run so the comparison isolates the
# DATASET. midtraining_telemetry = single cosine stage over mem_* + embed_model + main layers
# 13/14/15, plus the weight-0 read-channel block (mem_pos_weight_mass etc.), which is the metric
# that actually moved on the MuSiQue run.
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet_multihop}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    dataset=multihop_qa_sft_midtraining \
    dataset.batch_size=32 \
    trainer=midtraining_telemetry \
    trainer.resume_from="$RESUME_FROM" \
    trainer.steps=10000 \
    trainer.learning_rate=1e-4 \
    +trainer.training_stages.0.warmup_frac=0.1 \
    trainer.checkpoint_interval=1000 \
    trainer.max_to_keep=12 \
    trainer.log_interval=10 \
    trainer.eval_interval=1000000000 \
    'eval_set@trainer.evals=none' \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="multihop_sft_midtrain_topk64_seq512_chunks16_bs32"
